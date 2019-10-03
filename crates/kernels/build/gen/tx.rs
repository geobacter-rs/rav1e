// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::collections::BTreeMap;
use std::fmt;

use crate::gen::*;
use crate::loops::*;
use crate::plane::Plane;

fn tx_blocks_iter() -> impl Iterator<Item = Block> {
  Block::tx_sizes_iter()
    .filter(|b| {
      b.rect_log_ratio() <= 2
    })
}
fn tx_inv_block_shift(b: Block) -> u32 {
  match b {
    Block(4, 4) | Block(4, 8) | Block(8, 4) => 0,

    Block(8, 8)
    | Block(8, 16)
    | Block(16, 8)
    | Block(4, 16)
    | Block(16, 4)
    | Block(16, 32)
    | Block(32, 16)
    | Block(32, 64)
    | Block(64, 32) => 1,

    Block(16, 16)
    | Block(16, 64)
    | Block(64, 16)
    | Block(64, 64)
    | Block(32, 32)
    | Block(8, 32)
    | Block(32, 8) => 2,

    _ => unreachable!(),
  }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TxType {
  Id,
  Dct,
  Adst { flip: bool },
}
impl TxType {
  pub fn types() -> &'static [TxType] {
    const C: &'static [TxType] = &[
      TxType::Id,
      TxType::Dct,
      TxType::Adst { flip: false },
      TxType::Adst { flip: true },
    ];
    C
  }
  fn fn_suffix(&self) -> &'static str {
    match self {
      TxType::Id => "id",
      TxType::Dct => "dct",
      TxType::Adst { flip: false } => "adst",
      TxType::Adst { flip: true } => "flip_adst",
    }
  }
  pub fn flip(&self) -> bool {
    match self {
      TxType::Adst { flip } => *flip,
      _ => false,
    }
  }

  fn table_idx(&self) -> usize {
    match self {
      TxType::Id => 0,
      TxType::Dct => 1,
      TxType::Adst { flip: false } => 2,
      TxType::Adst { flip: true } => 3,
    }
  }

  fn inv_disable(&self, size: usize) -> bool {
    match (self, size) {
      (TxType::Adst { .. }, s) if s >= 32 => true,
      (TxType::Adst { flip: true }, _) => true,
      (TxType::Id, s) if s >= 64 => true,
      _ => false,
    }
  }
}
impl fmt::Display for TxType {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    f.pad(self.fn_suffix())
  }
}
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Tx2dType {
  pub col: TxType,
  pub row: TxType,
}
impl Tx2dType {
  pub fn types() -> impl Iterator<Item = Self> {
    let tys = TxType::types();
    tys
      .iter()
      .flat_map(move |&col| tys.iter().map(move |&row| Tx2dType { col, row }))
  }
  fn fn_suffix(&self) -> String {
    format!("{}_{}", self.row, self.col)
  }
  fn module_name(&self) -> Ident {
    let s = format!("x_{}_{}", self.row, self.col);
    Ident::new(&s, Span::call_site())
  }
  fn inv_disable(&self, b: Block) -> bool {
    self.row.inv_disable(b.w()) || self.col.inv_disable(b.h())
  }
}
impl fmt::Display for Tx2dType {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    f.pad(&self.fn_suffix())
  }
}

#[derive(Clone, Debug)]
enum IdxMap {
  /// Read/write.
  Idx(usize),
  /// The rest are read only
  Neg(usize),
  HalfBtf(NegIdx, usize, NegIdx, usize),
  AddClamp(NegIdx, NegIdx),
  Clamp(NegIdx),
  Const(SimdValue),
}
use self::IdxMap::{AddClamp, HalfBtf};

impl From<usize> for IdxMap {
  fn from(v: usize) -> IdxMap {
    IdxMap::Idx(v)
  }
}
impl From<NegIdx> for IdxMap {
  fn from(v: NegIdx) -> IdxMap {
    match v {
      Pos(idx) => idx.into(),
      Neg(idx) => IdxMap::Neg(idx),
    }
  }
}
struct Stage<T>
where
  T: Array + ?Sized,
{
  idx_map: Vec<IdxMap>,
  prev: T,
}
impl<T> Stage<T>
where
  T: Array,
{
  fn next(prev: T, map: Vec<IdxMap>) -> Self {
    Stage { prev, idx_map: map }
  }
  fn const_fill(prev: T, to_len: usize, v: SimdValue) -> Self {
    assert_eq!(prev.ty(), v.ty());
    let len = prev.len();
    let mut map = Vec::with_capacity(to_len);
    for i in 0..len.min(to_len) {
      map.push(i.into());
    }
    if to_len > len {
      for _ in 0..(to_len - len) {
        map.push(IdxMap::Const(v.clone()));
      }
    }
    let o = Stage { prev, idx_map: map };

    o
  }
}
impl<T> Array for Stage<T>
where
  T: Array + ?Sized,
{
  fn ty(&self) -> SimdType {
    self.prev.ty()
  }
  fn len(&self) -> usize {
    self.prev.len().max(self.idx_map.len())
  }
  fn get(&self, idx: usize) -> SimdValue {
    assert!(self.len() > idx);
    match self.idx_map.get(idx) {
      None => self.prev.get(idx), // passthrough
      Some(IdxMap::Idx(idx)) => self.prev.get(*idx),
      Some(IdxMap::Neg(idx)) => self.prev.get(*idx).neg(),
      Some(IdxMap::HalfBtf(w0, in0, w1, in1)) => {
        let in0 = self.prev.get(*in0);
        let in1 = self.prev.get(*in1);
        half_btf(*w0, &in0, *w1, &in1)
      }
      Some(IdxMap::AddClamp(l, r)) => match (*l, *r) {
        (Pos(l), Pos(r)) => {
          let l = self.prev.get(l);
          let r = self.prev.get(r);
          (l + r).clamp(quote!(min_value), quote!(max_value))
        }
        (Pos(l), Neg(r)) => {
          let l = self.prev.get(l);
          let r = self.prev.get(r);
          (l - r).clamp(quote!(min_value), quote!(max_value))
        }
        (Neg(l), Pos(r)) => {
          let l = self.prev.get(l);
          let r = self.prev.get(r);
          (r - l).clamp(quote!(min_value), quote!(max_value))
        }
        (Neg(l), Neg(r)) => {
          let l = self.prev.get(l);
          let r = self.prev.get(r);
          (l.neg() - r).clamp(quote!(min_value), quote!(max_value))
        }
      },
      Some(IdxMap::Clamp(idx)) => {
        let v = self.prev.get_neg(*idx);
        clamp(&v)
      }
      Some(IdxMap::Const(v)) => {
        assert_eq!(self.ty(), v.ty());
        v.clone()
      }
    }
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    assert!(self.len() > idx);
    assert_eq!(self.ty(), v.ty());
    let idx = match self.idx_map.get(idx) {
      None => idx,
      Some(IdxMap::Idx(idx)) => *idx,
      Some(map) => {
        panic!("{:?} is read only; can't write", map);
      }
    };

    self.prev.set(dst, idx, v);
  }
}

impl VarArr {
  fn add_clamp_merge<P, T, U>(
    dst: &mut TokenStream, prefix: P, lhs: T, rhs: U, map: &[(NegIdx, NegIdx)],
  ) -> VarArr
  where
    P: Display,
    T: Array,
    U: Array,
  {
    assert_eq!(lhs.ty(), rhs.ty());

    let mut names = Vec::with_capacity(map.len());
    for (i, &(l, r)) in map.iter().enumerate() {
      let l = lhs.get_neg(l);
      let r = rhs.get_neg(r);
      let v = clamp_pair(&l, &r)
        .let_(dst, format_args!("{}_{}", prefix, i))
        .unwrap_value();
      names.push(v);
    }

    VarArr { ty: lhs.ty(), prefix: prefix.to_string(), names }
  }
}

fn cospi_inv(ty: SimdType, idx: NegIdx) -> SimdValue {
  let v = match idx {
    Neg(idx) => quote!(-COSPI_INV[#idx]),
    Pos(idx) => quote!(COSPI_INV[#idx]),
  };

  ty.splat(v)
}

fn half_btf(
  w0: NegIdx, in0: &SimdValue, w1: NegIdx, in1: &SimdValue,
) -> SimdValue {
  assert_eq!(in0.ty(), in1.ty());
  let w0 = cospi_inv(in0.ty(), w0);
  let w1 = cospi_inv(in0.ty(), w1);
  let t = (w0 * in0) + (w1 * in1);
  SimdValue::from(
    t.ty(),
    quote! {
      { #t.round_shift(INV_COS_BIT as u32) }
    },
  )
}
fn clamp_pair(l: &SimdValue, r: &SimdValue) -> SimdValue {
  let t = (l + r).clamp(quote!(min_value), quote!(max_value));
  // wrap in a block for readability:
  SimdValue::from(t.ty(), quote!({ #t }))
}
fn clamp(v: &SimdValue) -> SimdValue {
  let t = v.clamp(quote!(min_value), quote!(max_value));
  // wrap in a block for readability:
  SimdValue::from(t.ty(), quote!({ #t }))
}

#[derive(Clone, Default)]
struct Functions {
  transpose: BTreeMap<Block, Func>,
  /// Block width is the size of the FFT, height is the SIMD lanes
  inv_id: BTreeMap<Block, Func>,
  inv_dct: BTreeMap<Block, Func>,
  inv_adst: BTreeMap<(Block, bool), Func>,

  initial_shift: BTreeMap<Block, Func>,
  middle_shift: BTreeMap<Block, Func>,
  final_shift_add: BTreeMap<Block, Func>,
  inv_tx_add: BTreeMap<(Tx2dType, Block), TableEntry>,
}

type Ctx<'a> = super::Ctx<'a, Functions>;
impl<'a> Ctx<'a> {
  fn tx_calc(&self) -> PrimType {
    PrimType::I32
  }
  fn tx_calc_lanes(&self) -> usize {
    self.simd_width(self.tx_calc())
  }
  fn transpose(&mut self, b: Block) -> Ident {
    let w = b.w();
    let h = b.h();
    let simd_lanes = w.min(8);
    let key = b;
    if let Some(f) = self.funcs.transpose.get(&key) {
      return f.name();
    }

    let calc = self.tx_calc();
    let calc_ty = SimdType::new(calc, simd_lanes);
    let offsets = calc_ty.indices(|idx| quote!(#idx) );

    let args = quote!{(
      input: *mut #calc,
      w_stride: i32,
      h_stride: i32,
    )};
    let params = vec![];
    let mut func = self
      .new_func(format_args!("transpose_{}", b.fn_suffix()),
                args, params, true);
    func.extend(quote! {
      debug_assert!(w_stride as usize >= #w);
      debug_assert!(h_stride as usize >= #h);

      let all_mask = <Simd<[m8; #simd_lanes]>>::splat(true);
      let offsets = #offsets * (h_stride as u32);
      let offsets = <Simd<[usize; #simd_lanes]>>::from_cast(offsets);
    });

    let input = Plane::new_named("input", quote!((w_stride as usize)));
    let output = Plane::new_named("input", quote!((h_stride as usize)));

    for r in 0..h {
      let rows = (0..w).step_by(simd_lanes)
        .map(|c| calc_ty.aload(input.add_rc(r, c)) )
        .fold(None, |acc, v| {
          if let Some(acc) = acc {
            Some(quote! {
              #acc, #v
            })
          } else {
            Some(quote! { #v })
          }
        });
      let name = Ident::new(&format!("row{}", r), Span::call_site());
      func.extend(quote! {
        let #name = [#rows];
      });
    }

    for r in 0..h {
      let name = Ident::new(&format!("row{}", r), Span::call_site());
      for c in (0..w).step_by(simd_lanes) {
        let c_chunk = c / simd_lanes;
        let dst_ptr = output.add_rc(c, r);
        let v = SimdValue::from(calc_ty, quote!(#name[#c_chunk]));
        func.extend(quote! {{
          let ptr = <Simd<[*mut #calc; #simd_lanes]>>::splat(#dst_ptr);
          let ptr = ptr.add(offsets);
          ptr.write(all_mask, #v);
        }});
      }
    }

    func.to_tokens(&mut **self.out);
    let name = func.name();
    assert!(self.funcs.transpose.insert(key, func).is_none());
    name
  }
  fn inv_tx_args(&self) -> TokenStream {
    let calc = self.tx_calc();
    quote!{(
      input: *mut #calc,
      input_stride: i32,
      range: Range<#calc>,
      total_lanes: u16,
    )}
  }
  fn inv_tx_gen<F>(&mut self, b: Block, tx: TxType, f: F) -> Func
    where F: FnOnce(&mut TokenStream, Box<dyn Array>) -> Box<dyn Array>,
  {
    let name = format!("inv_{}_{}", tx.fn_suffix(), b.fn_suffix());
    let args = self.inv_tx_args();
    let mut func = self.new_func(name, args,
                                 vec![], true);
    let calc = self.tx_calc();
    let simd_lanes = self.simd_width(calc);
    assert!(b.h() <= simd_lanes);
    let calc_ty = SimdType::new(calc, b.h());

    let h = b.h();
    func.extend(quote! {
      debug_assert_eq!(total_lanes as usize % #h, 0);
      let min_value = range.start;
      let max_value = range.end;
    });

    let input = Ident::new("input", Span::call_site());
    // so we iterate over the columns of the plane, not the rows
    let input = Plane::new_stride(&input, 1i32);

    let mut l = Loop::new("l", quote!(total_lanes),
                          PrimType::U16);
    l.add_loop_var(input);

    l.gen(&mut *func, b.h() as _, move |body, _l, vars| {
      let input = &vars[0];
      // restore the original stride:
      let input = Plane::new_stride(input.name(),
                                    quote!(input_stride));

      let inputs = (0..b.w())
        .map(|idx| {
          let ptr = input.add_r(idx as i32);
          let v = if b.h() <= b.w() {
            calc_ty.aload(ptr)
          } else {
            calc_ty.uload(ptr)
          };
          v.let_(body, format_args!("iv_{}", idx))
        })
        .collect::<Vec<_>>();

      let inputs = Box::new(Vector::new(calc_ty, inputs));

      let outputs = f(body, inputs);
      for (idx, output) in outputs.iter().enumerate() {
        if b.h() <= b.w() {
          output.astore(body, input.add_r(idx as i32))
        } else {
          output.ustore(body, input.add_r(idx as i32))
        }
      }
    });

    func
  }
  fn inv_tx_id(&mut self, b: Block) -> Ident {
    let simd_lanes = b.h().min(self.tx_calc_lanes());
    let simd_b = Block(b.w(), simd_lanes);
    let key = simd_b;
    if let Some(f) = self.funcs.inv_id.get(&key) {
      return f.name();
    }

    let func = self.inv_tx_gen(simd_b, TxType::Id, |body, arr| {
      match b.w() {
        4 => Box::new(iidentity4(body, "row", arr)),
        8 => Box::new(iidentity8(body, "row", arr)),
        16 => Box::new(iidentity16(body, "row", arr)),
        32 => Box::new(iidentity32(body, "row", arr)),
        _ => unimplemented!("id width: {}", b.w()),
      }
    });

    func.to_tokens(&mut **self.out);
    let name = func.name();
    assert!(self.funcs.inv_id.insert(key, func).is_none());
    name
  }
  fn inv_tx_dct(&mut self, b: Block) -> Ident {
    let simd_lanes = b.h().min(self.tx_calc_lanes());
    let simd_b = Block(b.w(), simd_lanes);
    let key = simd_b;
    if let Some(f) = self.funcs.inv_dct.get(&key) {
      return f.name();
    }

    let func = self.inv_tx_gen(simd_b, TxType::Dct, |body, arr| {
      match b.w() {
        4 => Box::new(idct4(body, "row", arr)),
        8 => Box::new(idct8(body, "row", arr)),
        16 => Box::new(idct16(body, "row", arr)),
        32 => Box::new(idct32(body, "row", arr)),
        64 => Box::new(idct64(body, "row", arr)),
        _ => unimplemented!("dct width: {}", b.w()),
      }
    });

    func.to_tokens(&mut **self.out);
    let name = func.name();
    assert!(self.funcs.inv_dct.insert(key, func).is_none());
    name
  }
  fn inv_tx_adst(&mut self, b: Block, flip: bool) -> Ident {
    let simd_lanes = b.h().min(self.tx_calc_lanes());
    let simd_b = Block(b.w(), simd_lanes);
    let key = (simd_b, flip);
    if let Some(f) = self.funcs.inv_adst.get(&key) {
      return f.name();
    }

    let func = self.inv_tx_gen(simd_b, TxType::Adst { flip, }, |body, arr| {
      match b.w() {
        4 => Box::new(iadst4(body, "row", arr)),
        8 => Box::new(iadst8(body, "row", arr)),
        16 => Box::new(iadst16(body, "row", arr)),
        _ => unimplemented!("adst width: {}", b.w()),
      }
    });

    func.to_tokens(&mut **self.out);
    let name = func.name();
    assert!(self.funcs.inv_adst.insert(key, func).is_none());
    name
  }
  fn inv_tx(&mut self, b: Block, tx: TxType) -> Ident {
    match tx {
      TxType::Id => self.inv_tx_id(b),
      TxType::Dct => self.inv_tx_dct(b),
      TxType::Adst { flip, } => self.inv_tx_adst(b, flip),
    }
  }
  fn initial_shift(&mut self, original_b: Block) -> Ident  {
    let b = Block(original_b.w().min(32),
                  original_b.h().min(32));
    let w = b.w();
    let h = b.h();
    let calc = self.tx_calc();
    let calc_lanes = w.min(h)
      .min(self.tx_calc_lanes());
    let calc_ty = SimdType::new(calc, calc_lanes);

    if let Some(f) = self.funcs.initial_shift.get(&original_b) {
      return f.name();
    }

    let name = format!("initial_inv_shift_{}", original_b.fn_suffix());
    let args = quote!{(
      input: *const #calc,
      output: *mut #calc,
      output_stride: i32,
      bd: u8,
    )};
    let mut func = self.new_func(name, args,
                                 vec![], true);
    if self.px == PixelType::U8 {
      func.extend(quote! {
        debug_assert_eq!(bd, 8,
                         "inv_tx_add expects 8bit depth on u8 pixels; got {}",
                         bd);
        let bd = 8;
      });
    }
    func.extend(quote! {
      let range = bd + 8;
      let max_value = ((1i64 << (range - 1)) - 1) as #calc;
      let min_value = (-(1i64 << (range - 1))) as #calc;
      let input_stride = #w as i32;
    });
    let input = Ident::new("input", Span::call_site());
    let input = Plane::new(&input);
    let output = Ident::new("output", Span::call_site());
    let output = Plane::new(&output);

    let mut l = BlockLoop::new(quote!(#w),
                               quote!(#h),
                               PrimType::U16);
    l.add_loop_var(input);
    l.add_loop_var(output);
    l.gen(&mut *func, 1, calc_lanes as _,
          |body, _, _, vars: &[_]| {
            let input = &vars[0];
            let output = &vars[1];

            let i = calc_ty.aload(input)
              .let_(body, "t");
            let i = if original_b.rect_log_ratio() == 1 {
              let i = SimdValue::from(calc_ty, quote!(#i * (INV_SQRT2 as #calc)));
              i.round_shift(quote!(SQRT2_BITS as u32))
            } else {
              i
            };

            clamp(&i)
              .let_(body, "t")
              .astore(body, output);
          });

    func.to_tokens(&mut **self.out);
    let name = func.name();
    assert!(self.funcs.initial_shift.insert(original_b, func).is_none());
    name
  }
  fn middle_shift(&mut self, b: Block) -> Ident  {
    let w = b.w();
    let h = b.h();
    let calc = self.tx_calc();
    let calc_lanes = w.min(h)
      .min(self.tx_calc_lanes());
    let calc_ty = SimdType::new(calc, calc_lanes);
    let shift = tx_inv_block_shift(b);

    if let Some(f) = self.funcs.middle_shift.get(&b) {
      return f.name();
    }

    let name = format!("middle_inv_shift_{}", b.fn_suffix());
    let args = quote!{(
      io: *mut #calc,
      bd: u8,
    )};
    let mut func = self.new_func(name, args,
                                 vec![], true);
    if self.px == PixelType::U8 {
      func.extend(quote! {
        debug_assert_eq!(bd, 8,
                         "inv_tx_add expects 8bit depth on u8 pixels; got {}",
                         bd);
        let bd = 8;
      });
    }
    func.extend(quote! {
      let range = bd + 8;
      let max_value = ((1i64 << (range - 1)) - 1) as #calc;
      let min_value = (-(1i64 << (range - 1))) as #calc;
    });
    let io = Plane::new_named("io", quote!(#w as i32));

    let mut l = BlockLoop::new(w, h, PrimType::U16);
    l.add_loop_var(io);
    l.gen(&mut *func, 1, calc_lanes as _,
          |body, _, _, vars: &[_]| {
            let io = &vars[0];

            let i = if b.h() <= b.w() {
              calc_ty.aload(io)
                .let_(body, "t")
            } else {
              calc_ty.uload(io)
                .let_(body, "t")
            };
            let i = clamp(&i.round_shift(quote!(#shift)))
              .let_(body, "t");
            i.astore(body, io);
          });

    func.to_tokens(&mut **self.out);
    let name = func.name();
    assert!(self.funcs.middle_shift.insert(b, func).is_none());
    name
  }
  fn final_shift(&mut self, b: Block) -> Ident  {
    let w = b.w();
    let h = b.h();
    let px = self.px;
    let calc = self.tx_calc();
    let calc_lanes = w.min(h)
      .min(self.tx_calc_lanes());
    let calc_ty = SimdType::new(calc, calc_lanes);
    let px_ty = SimdType::new(px.into(), calc_lanes);

    if let Some(f) = self.funcs.final_shift_add.get(&b) {
      return f.name();
    }

    let name = format!("final_inv_shift_add_{}", b.fn_suffix());
    let args = quote!{(
      input: *const #calc,
      output: *mut #px,
      output_stride: i32,
      bd: u8,
    )};
    let mut func = self.new_func(name, args,
                                 vec![], true);
    if self.px == PixelType::U8 {
      func.extend(quote! {
        debug_assert_eq!(bd, 8,
                         "inv_tx_add expects 8bit depth on u8 pixels; got {}",
                         bd);
        let bd = 8;
      });
    }
    let add_min = quote!(0);
    let add_max = quote!(((1 << bd) - 1) as #calc);
    let input = Plane::new_named("input", quote!(#w as i32));
    let output = Ident::new("output", Span::call_site());
    let output = Plane::new(&output);

    let mut l = BlockLoop::new(quote!(#w),
                               quote!(#h),
                               PrimType::U16);
    l.add_loop_var(input);
    l.add_loop_var(output);
    l.gen(&mut *func, 1, calc_lanes as _,
          |body, _, _, vars: &[_]| {
            let input = &vars[0];
            let output = &vars[1];

            let px_v = px_ty.uload(output)
              .cast(calc_ty)
              .let_(body, "px_v");
            let in_v = if b.h() <= b.w() {
              calc_ty.aload(input)
            } else {
              calc_ty.uload(input)
            };
            let in_v = in_v.let_(body, "in_v")
              .round_shift(4u32)
              .let_(body, "in_v");
            let v = (px_v + in_v)
              .clamp(&add_min, &add_max)
              .cast(px_ty)
              .let_(body, "out_v");
            v.ustore(body, output);
          });

    func.to_tokens(&mut **self.out);
    let name = func.name();
    assert!(self.funcs.final_shift_add.insert(b, func).is_none());
    name
  }

  fn inv_tx_add_args(&self) -> TokenStream {
    let calc = self.tx_calc();
    let px = self.px;
    quote!{(
      input: *const #calc,
      output: *mut #px,
      output_stride: i32,
      bd: u8,
      width: u16, height: u16,
    )}
  }
  fn inv_tx_add(&mut self, b: Block, tx: Tx2dType) {
    let key = (tx, b);
    if let Some(_) = self.funcs.inv_tx_add.get(&key) {
      return;
    }

    let initial_transpose = self.transpose(Block(b.w().min(32),
                                                 b.h().min(32)));
    let row_col_transpose = self.transpose(b.transpose());
    let row_tx = self.inv_tx(b, tx.row);
    let col_tx = self.inv_tx(b.transpose(), tx.col);
    let initial_shift = self.initial_shift(b);
    let middle_shift = self.middle_shift(b.transpose());
    let final_shift = self.final_shift(b);

    let calc = self.tx_calc();

    let args = self.inv_tx_add_args();
    let params = vec![];
    let mut func = self
      .new_func(format_args!("inv_tx_add_{}_{}",
                             tx.fn_suffix(), b.fn_suffix()),
                args, params, true);

    Self::block_size_check(&mut *func, b);

    if self.px == PixelType::U8 {
      func.extend(quote! {
        debug_assert_eq!(bd, 8,
                         "inv_tx_add expects 8bit depth on u8 pixels; got {}",
                         bd);
        let bd = 8;
      });
    }

    let w = b.w();
    let h = b.h();
    let area = b.area();

    func.extend(quote! {
      let range = bd + 8;
      let max_value = ((1i64 << (range - 1)) - 1) as #calc;
      let min_value = (-(1i64 << (range - 1))) as #calc;
      let value_range = min_value..max_value;
      let mut row_buf = AlignedArray::new([0; #area]);
      let row_ptr = row_buf.as_mut_ptr();

      #initial_shift(input, row_ptr, #w as _, bd);
      // TODO: make the forward tx write transposed values, so this isn't needed.
      #initial_transpose(row_ptr, #w as _, #h as _);

      #row_tx(row_ptr, #h as _, value_range, #h as _);

      #middle_shift(row_ptr, bd);

      let range = ::std::cmp::max(bd + 6, 16);
      let max_value = ((1i64 << (range - 1)) - 1) as #calc;
      let min_value = (-(1i64 << (range - 1))) as #calc;
      let value_range = min_value..max_value;

      #row_col_transpose(row_ptr, #h as _, #w as _);

      #col_tx(row_ptr, #w as _, value_range, #w as _);

      #final_shift(row_ptr, output, output_stride, bd);
    });


    func.to_tokens(&mut **self.out);

    let feature_idx = self.isa.index();
    let b_enum = b.table_idx();
    let row_idx = tx.row.table_idx();
    let col_idx = tx.col.table_idx();
    let indices = quote! {
      [#feature_idx][#b_enum][#row_idx][#col_idx]
    };
    let entry = TableEntry {
      indices: Rc::new(indices),
      name: func.name(),
      path: func.path(),
      func: None,
    };
    assert!(self.funcs.inv_tx_add.insert(key, entry).is_none());
  }
  fn inv_tx_adds(&mut self) {
    for b in tx_blocks_iter() {
      for tx in Tx2dType::types() {
        if tx.inv_disable(b) {
          continue;
        }

        self.inv_tx_add(b, tx);
      }
    }
  }
  fn push_kernels(&self, set: &mut KernelDispatch) {
    for k in self.funcs.inv_tx_add.values() {
      set.push_kernel(self.px, k.idx(), k.path());
    }
  }
}

fn iidentity4<T, U>(_: &mut TokenStream, _disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  input.map(|_idx, v| {
    let v = v.ty().splat(quote!(SQRT2)) * v;
    v.round_shift(12u32)
  })
}
fn iidentity8<T, U>(_: &mut TokenStream, _disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  input.map(|_idx, v| v.ty().splat(quote!(2)) * v)
}
fn iidentity16<T, U>(_: &mut TokenStream, _disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  input.map(|_idx, v| {
    let v = v.ty().splat(quote!(SQRT2 * 2)) * v;
    v.round_shift(12u32)
  })
}
fn iidentity32<T, U>(_: &mut TokenStream, _disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  input.map(|_idx, v| v.ty().splat(quote!(4)) * v)
}

fn iadst4<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  let ty = input.ty();
  let sinpi_inv = |i: usize| -> SimdValue { ty.splat(quote!(SINPI_INV[#i])) };

  let x0 = input.get(0);
  let x1 = input.get(1);
  let x2 = input.get(2);
  let x3 = input.get(3);

  let s0 = sinpi_inv(1) * &x0;
  let s1 = sinpi_inv(2) * &x0;
  let s2 = sinpi_inv(3) * &x1;
  let s3 = sinpi_inv(4) * &x2;
  let s4 = sinpi_inv(1) * &x2;
  let s5 = sinpi_inv(2) * &x3;
  let s6 = sinpi_inv(4) * &x3;

  let s7 = (&x0 - &x2) + &x3;

  let s0 = s0 + s3;
  let s1 = s1 - s4;
  let s3 = s2;
  let s2 = sinpi_inv(3) * s7;

  let s0 = s0 + s5;
  let s1 = s1 - s6;

  let x0 = &s0 + &s3;
  let x1 = &s1 + &s3;
  let x2 = s2;
  let x3 = s0 + s1;

  let x3 = x3 - s3;

  let bit = 12u32;
  let out = vec![
    x0.round_shift(bit),
    x1.round_shift(bit),
    x2.round_shift(bit),
    x3.round_shift(bit),
  ];
  let out = Vector::new(ty, out);

  VarArr::let_(dst, format_args!("{}_iadst4", disc), &out)
}
fn iadst8<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  let stg1 = vec![
    7usize.into(),
    0usize.into(),
    5usize.into(),
    2usize.into(),
    3usize.into(),
    4usize.into(),
    1usize.into(),
    6usize.into(),
  ];
  let stg1 = Stage::next(input, stg1);

  let stg2 = vec![
    HalfBtf(Pos(4), 0, Pos(60), 1),
    HalfBtf(Pos(60), 0, Neg(4), 1),
    HalfBtf(Pos(20), 2, Pos(44), 3),
    HalfBtf(Pos(44), 2, Neg(20), 3),
    HalfBtf(Pos(36), 4, Pos(28), 5),
    HalfBtf(Pos(28), 4, Neg(36), 5),
    HalfBtf(Pos(52), 6, Pos(12), 7),
    HalfBtf(Pos(12), 6, Neg(52), 7),
  ];
  let stg2 = Stage::next(stg1, stg2);

  let stg3 = vec![
    AddClamp(Pos(0), Pos(4)),
    AddClamp(Pos(1), Pos(5)),
    AddClamp(Pos(2), Pos(6)),
    AddClamp(Pos(3), Pos(7)),
    AddClamp(Pos(0), Neg(4)),
    AddClamp(Pos(1), Neg(5)),
    AddClamp(Pos(2), Neg(6)),
    AddClamp(Pos(3), Neg(7)),
  ];
  let stg3 = Stage::next(stg2, stg3);

  let stg4 = vec![
    0usize.into(),
    1usize.into(),
    2usize.into(),
    3usize.into(),
    HalfBtf(Pos(16), 4, Pos(48), 5),
    HalfBtf(Pos(48), 4, Neg(16), 5),
    HalfBtf(Neg(48), 6, Pos(16), 7),
    HalfBtf(Pos(16), 6, Pos(48), 7),
  ];
  let stg4 = Stage::next(stg3, stg4);

  let stg5 = vec![
    AddClamp(Pos(0), Pos(2)),
    AddClamp(Pos(1), Pos(3)),
    AddClamp(Pos(0), Neg(2)),
    AddClamp(Pos(1), Neg(3)),
    AddClamp(Pos(4), Pos(6)),
    AddClamp(Pos(5), Pos(7)),
    AddClamp(Pos(4), Neg(6)),
    AddClamp(Pos(5), Neg(7)),
  ];
  let stg5 = Stage::next(stg4, stg5);

  let stg6 = vec![
    0usize.into(),
    1usize.into(),
    HalfBtf(Pos(32), 2, Pos(32), 3),
    HalfBtf(Pos(32), 2, Neg(32), 3),
    4usize.into(),
    5usize.into(),
    HalfBtf(Pos(32), 6, Pos(32), 7),
    HalfBtf(Pos(32), 6, Neg(32), 7),
  ];
  let stg6 = Stage::next(stg5, stg6);

  let stg7 = vec![
    0usize.into(),
    Neg(4).into(),
    6usize.into(),
    Neg(2).into(),
    3usize.into(),
    Neg(7).into(),
    5usize.into(),
    Neg(1).into(),
  ];
  let stg7 = Stage::next(stg6, stg7);

  VarArr::let_(dst, format_args!("{}_iadst8", disc), &stg7)
}
fn iadst16<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
  where T: Array, U: Display,
{
  let stg1 = vec![
    15usize.into(),
    0usize.into(),
    13usize.into(),
    2usize.into(),
    11usize.into(),
    4usize.into(),
    9usize.into(),
    6usize.into(),
    7usize.into(),
    8usize.into(),
    5usize.into(),
    10usize.into(),
    3usize.into(),
    12usize.into(),
    1usize.into(),
    14usize.into(),
  ];
  let stg1 = Stage::next(input, stg1);
  let stg1 = VarArr::let_(dst, format_args!("{}_iadst16_stg1", disc), &stg1);
  let stg2 = vec![
    HalfBtf(Pos ( 2usize ), 0usize, Pos ( 62usize ), 1usize, ),
    HalfBtf(Pos ( 62usize ), 0usize, Neg ( 2usize ), 1usize, ),
    HalfBtf(Pos ( 10usize ), 2usize, Pos ( 54usize ), 3usize, ),
    HalfBtf(Pos ( 54usize ), 2usize, Neg ( 10usize ), 3usize, ),
    HalfBtf(Pos ( 18usize ), 4usize, Pos ( 46usize ), 5usize, ),
    HalfBtf(Pos ( 46usize ), 4usize, Neg ( 18usize ), 5usize, ),
    HalfBtf(Pos ( 26usize ), 6usize, Pos ( 38usize ), 7usize, ),
    HalfBtf(Pos ( 38usize ), 6usize, Neg ( 26usize ), 7usize, ),
    HalfBtf(Pos ( 34usize ), 8usize, Pos ( 30usize ), 9usize, ),
    HalfBtf(Pos ( 30usize ), 8usize, Neg ( 34usize ), 9usize, ),
    HalfBtf(Pos ( 42usize ), 10usize, Pos ( 22usize ), 11usize, ),
    HalfBtf(Pos ( 22usize ), 10usize, Neg ( 42usize ), 11usize, ),
    HalfBtf(Pos ( 50usize ), 12usize, Pos ( 14usize ), 13usize, ),
    HalfBtf(Pos ( 14usize ), 12usize, Neg ( 50usize ), 13usize, ),
    HalfBtf(Pos ( 58usize ), 14usize, Pos ( 6usize ), 15usize, ),
    HalfBtf(Pos ( 6usize ), 14usize, Neg ( 58usize ), 15usize, ),
  ];
  let stg2 = Stage::next(stg1, stg2);
  let stg2 = VarArr::let_(dst, format_args!("{}_iadst16_stg2", disc), &stg2);
  let stg3 = vec![
    AddClamp(Pos ( 0usize ), Pos ( 8usize )),
    AddClamp(Pos ( 1usize ), Pos ( 9usize )),
    AddClamp(Pos ( 2usize ), Pos ( 10usize )),
    AddClamp(Pos ( 3usize ), Pos ( 11usize )),
    AddClamp(Pos ( 4usize ), Pos ( 12usize )),
    AddClamp(Pos ( 5usize ), Pos ( 13usize )),
    AddClamp(Pos ( 6usize ), Pos ( 14usize )),
    AddClamp(Pos ( 7usize ), Pos ( 15usize )),
    AddClamp(Pos ( 0usize ), Neg ( 8usize )),
    AddClamp(Pos ( 1usize ), Neg ( 9usize )),
    AddClamp(Pos ( 2usize ), Neg ( 10usize )),
    AddClamp(Pos ( 3usize ), Neg ( 11usize )),
    AddClamp(Pos ( 4usize ), Neg ( 12usize )),
    AddClamp(Pos ( 5usize ), Neg ( 13usize )),
    AddClamp(Pos ( 6usize ), Neg ( 14usize )),
    AddClamp(Pos ( 7usize ), Neg ( 15usize )),
  ];
  let stg3 = Stage::next(stg2, stg3);
  let stg3 = VarArr::let_(dst, format_args!("{}_iadst16_stg3", disc), &stg3);
  let stg4 = vec![
    0usize.into(),
    1usize.into(),
    2usize.into(),
    3usize.into(),
    4usize.into(),
    5usize.into(),
    6usize.into(),
    7usize.into(),
    HalfBtf(Pos ( 8usize ), 8usize, Pos ( 56usize ), 9usize, ),
    HalfBtf(Pos ( 56usize ), 8usize, Neg ( 8usize ), 9usize, ),
    HalfBtf(Pos ( 40usize ), 10usize, Pos ( 24usize ), 11usize, ),
    HalfBtf(Pos ( 24usize ), 10usize, Neg ( 40usize ), 11usize, ),
    HalfBtf(Neg ( 56usize ), 12usize, Pos ( 8usize ), 13usize, ),
    HalfBtf(Pos ( 8usize ), 12usize, Pos ( 56usize ), 13usize, ),
    HalfBtf(Neg ( 24usize ), 14usize, Pos ( 40usize ), 15usize, ),
    HalfBtf(Pos ( 40usize ), 14usize, Pos ( 24usize ), 15usize, ),
  ];
  let stg4 = Stage::next(stg3, stg4);
  let stg4 = VarArr::let_(dst, format_args!("{}_iadst16_stg4", disc), &stg4);
  let stg5 = vec![
    AddClamp(Pos ( 0usize ), Pos ( 4usize )),
    AddClamp(Pos ( 1usize ), Pos ( 5usize )),
    AddClamp(Pos ( 2usize ), Pos ( 6usize )),
    AddClamp(Pos ( 3usize ), Pos ( 7usize )),
    AddClamp(Pos ( 0usize ), Neg ( 4usize )),
    AddClamp(Pos ( 1usize ), Neg ( 5usize )),
    AddClamp(Pos ( 2usize ), Neg ( 6usize )),
    AddClamp(Pos ( 3usize ), Neg ( 7usize )),
    AddClamp(Pos ( 8usize ), Pos ( 12usize )),
    AddClamp(Pos ( 9usize ), Pos ( 13usize )),
    AddClamp(Pos ( 10usize ), Pos ( 14usize )),
    AddClamp(Pos ( 11usize ), Pos ( 15usize )),
    AddClamp(Pos ( 8usize ), Neg ( 12usize )),
    AddClamp(Pos ( 9usize ), Neg ( 13usize )),
    AddClamp(Pos ( 10usize ), Neg ( 14usize )),
    AddClamp(Pos ( 11usize ), Neg ( 15usize )),
  ];
  let stg5 = Stage::next(stg4, stg5);
  let stg5 = VarArr::let_(dst, format_args!("{}_iadst16_stg5", disc), &stg5);
  let stg6 = vec![
    0usize.into(),
    1usize.into(),
    2usize.into(),
    3usize.into(),
    HalfBtf(Pos ( 16usize ), 4usize, Pos ( 48usize ), 5usize, ),
    HalfBtf(Pos ( 48usize ), 4usize, Neg ( 16usize ), 5usize, ),
    HalfBtf(Neg ( 48usize ), 6usize, Pos ( 16usize ), 7usize, ),
    HalfBtf(Pos ( 16usize ), 6usize, Pos ( 48usize ), 7usize, ),
    8usize.into(),
    9usize.into(),
    10usize.into(),
    11usize.into(),
    HalfBtf(Pos ( 16usize ), 12usize, Pos ( 48usize ), 13usize, ),
    HalfBtf(Pos ( 48usize ), 12usize, Neg ( 16usize ), 13usize, ),
    HalfBtf(Neg ( 48usize ), 14usize, Pos ( 16usize ), 15usize, ),
    HalfBtf(Pos ( 16usize ), 14usize, Pos ( 48usize ), 15usize, ),
  ];
  let stg6 = Stage::next(stg5, stg6);
  let stg6 = VarArr::let_(dst, format_args!("{}_iadst16_stg6", disc), &stg6);
  let stg7 = vec![
    AddClamp(Pos ( 0usize ), Pos ( 2usize )),
    AddClamp(Pos ( 1usize ), Pos ( 3usize )),
    AddClamp(Pos ( 0usize ), Neg ( 2usize )),
    AddClamp(Pos ( 1usize ), Neg ( 3usize )),
    AddClamp(Pos ( 4usize ), Pos ( 6usize )),
    AddClamp(Pos ( 5usize ), Pos ( 7usize )),
    AddClamp(Pos ( 4usize ), Neg ( 6usize )),
    AddClamp(Pos ( 5usize ), Neg ( 7usize )),
    AddClamp(Pos ( 8usize ), Pos ( 10usize )),
    AddClamp(Pos ( 9usize ), Pos ( 11usize )),
    AddClamp(Pos ( 8usize ), Neg ( 10usize )),
    AddClamp(Pos ( 9usize ), Neg ( 11usize )),
    AddClamp(Pos ( 12usize ), Pos ( 14usize )),
    AddClamp(Pos ( 13usize ), Pos ( 15usize )),
    AddClamp(Pos ( 12usize ), Neg ( 14usize )),
    AddClamp(Pos ( 13usize ), Neg ( 15usize )),
  ];
  let stg7 = Stage::next(stg6, stg7);
  let stg7 = VarArr::let_(dst, format_args!("{}_iadst16_stg7", disc), &stg7);
  let stg8 = vec![
    0usize.into(),
    1usize.into(),
    HalfBtf(Pos ( 32usize ), 2usize, Pos ( 32usize ), 3usize, ),
    HalfBtf(Pos ( 32usize ), 2usize, Neg ( 32usize ), 3usize, ),
    4usize.into(),
    5usize.into(),
    HalfBtf(Pos ( 32usize ), 6usize, Pos ( 32usize ), 7usize, ),
    HalfBtf(Pos ( 32usize ), 6usize, Neg ( 32usize ), 7usize, ),
    8usize.into(),
    9usize.into(),
    HalfBtf(Pos ( 32usize ), 10usize, Pos ( 32usize ), 11usize, ),
    HalfBtf(Pos ( 32usize ), 10usize, Neg ( 32usize ), 11usize, ),
    12usize.into(),
    13usize.into(),
    HalfBtf(Pos ( 32usize ), 14usize, Pos ( 32usize ), 15usize, ),
    HalfBtf(Pos ( 32usize ), 14usize, Neg ( 32usize ), 15usize, ),
  ];
  let stg8 = Stage::next(stg7, stg8);
  let stg8 = VarArr::let_(dst, format_args!("{}_iadst16_stg8", disc), &stg8);
  let stg9 = vec![
    0usize.into(),
    Neg(8).into(),
    12usize.into(),
    Neg(4).into(),
    6usize.into(),
    Neg(14).into(),
    10usize.into(),
    Neg(2).into(),
    3usize.into(),
    Neg(11).into(),
    15usize.into(),
    Neg(7).into(),
    5usize.into(),
    Neg(13).into(),
    9usize.into(),
    Neg(1).into(),
  ];
  let stg9 = Stage::next(stg8, stg9);
  let stg9 = VarArr::let_(dst, format_args!("{}_iadst16_stg9", disc), &stg9);
  stg9
}

fn idct4<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  assert!(input.len() >= 4, "{} < {}", input.len(), 4);

  let stg1 = vec![0usize.into(), 2usize.into(), 1usize.into(), 3usize.into()];
  let stg1 = Stage::next(input, stg1).trunc(4);

  let stg2 = vec![
    IdxMap::HalfBtf(Pos(32), 0, Pos(32), 1),
    IdxMap::HalfBtf(Pos(32), 0, Neg(32), 1),
    IdxMap::HalfBtf(Pos(48), 2, Neg(16), 3),
    IdxMap::HalfBtf(Pos(16), 2, Pos(48), 3),
  ];
  let stg2 = Stage::next(stg1, stg2);
  let stg2 = VarArr::let_(dst, format_args!("{}_idct4_stg2", disc), &stg2);

  let stg3 = vec![
    AddClamp(Pos(0), Pos(3)),
    AddClamp(Pos(1), Pos(2)),
    AddClamp(Pos(1), Neg(2)),
    AddClamp(Pos(0), Neg(3)),
  ];
  let stg3 = Stage::next(stg2, stg3);
  VarArr::let_(dst, format_args!("{}_idct4_stg3", disc), &stg3)
}
fn idct8<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  assert!(input.len() >= 8, "{} < {}", input.len(), 8);

  let idct4 = vec![0usize.into(), 2usize.into(), 4usize.into(), 6usize.into()];
  let idct4 = Stage::next(&input, idct4);
  // XXX can't use format_args! here??? Just here??
  let idct4 = self::idct4(dst, format!("{}_idct8", disc), idct4);

  let stg1 = vec![1usize.into(), 5usize.into(), 3usize.into(), 7usize.into()];
  let stg1 = Stage::next(&input, stg1).trunc(4);

  let stg2 = vec![
    HalfBtf(Pos(56), 0, Neg(8), 3),
    HalfBtf(Pos(24), 1, Neg(40), 2),
    HalfBtf(Pos(40), 1, Pos(24), 2),
    HalfBtf(Pos(8), 0, Pos(56), 3),
  ];
  let stg2 = Stage::next(stg1, stg2);
  let stg2 = VarArr::let_(dst, format_args!("{}_idct8_stg2", disc), &stg2);

  let stg3 = vec![
    AddClamp(Pos(0), Pos(1)),
    AddClamp(Pos(0), Neg(1)),
    AddClamp(Neg(2), Pos(3)),
    AddClamp(Pos(2), Pos(3)),
  ];
  let stg3 = Stage::next(stg2, stg3);
  let stg3 = VarArr::let_(dst, format_args!("{}_idct8_stg3", disc), &stg3);

  let stg4 = vec![
    0usize.into(),
    HalfBtf(Neg(32), 1, Pos(32), 2),
    HalfBtf(Pos(32), 1, Pos(32), 2),
    3usize.into(),
  ];
  let stg4 = Stage::next(stg3, stg4);
  let stg4 = VarArr::let_(dst, format_args!("{}_idct8_stg4", disc), &stg4);

  let mut stg5 = Vec::new();
  for i in 0..4 {
    stg5.push((Pos(i), Pos(3 - i)));
  }
  for i in 0..4 {
    stg5.push((Pos(3 - i), Neg(i)));
  }

  VarArr::add_clamp_merge(
    dst,
    format_args!("{}_idct8", disc),
    idct4,
    stg4,
    &stg5,
  )
}
fn idct16<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  assert!(input.len() >= 16, "{} < {}", input.len(), 16);

  let idct8 = (0..8usize).map(|i| (i * 2).into()).collect::<Vec<_>>();
  let idct8 = Stage::next(&input, idct8);
  // XXX can't use format_args! here??? Just here??
  let idct8 = self::idct8(dst, format!("{}_idct16", disc), idct8);

  let stg1 = vec![
    1usize.into(),
    9usize.into(),
    5usize.into(),
    13usize.into(),
    3usize.into(),
    11usize.into(),
    7usize.into(),
    15usize.into(),
  ];
  let stg1 = Stage::next(&input, stg1).trunc(8);

  let stg2 = vec![
    HalfBtf(Pos(60), 0, Neg(4), 7),
    HalfBtf(Pos(28), 1, Neg(36), 6),
    HalfBtf(Pos(44), 2, Neg(20), 5),
    HalfBtf(Pos(12), 3, Neg(52), 4),
    HalfBtf(Pos(52), 3, Pos(12), 4),
    HalfBtf(Pos(20), 2, Pos(44), 5),
    HalfBtf(Pos(36), 1, Pos(28), 6),
    HalfBtf(Pos(4), 0, Pos(60), 7),
  ];
  let stg2 = Stage::next(stg1, stg2);
  let stg2 = VarArr::let_(dst, format_args!("{}_idct16_stg2", disc), &stg2);

  let stg3 = vec![
    AddClamp(Pos(0), Pos(1)),
    AddClamp(Pos(0), Neg(1)),
    AddClamp(Neg(2), Pos(3)),
    AddClamp(Pos(2), Pos(3)),
    AddClamp(Pos(4), Pos(5)),
    AddClamp(Pos(4), Neg(5)),
    AddClamp(Neg(6), Pos(7)),
    AddClamp(Pos(6), Pos(7)),
  ];
  let stg3 = Stage::next(stg2, stg3);
  let stg3 = VarArr::let_(dst, format_args!("{}_idct16_stg3", disc), &stg3);

  let stg4 = vec![
    0usize.into(),
    HalfBtf(Neg(16), 1, Pos(48), 6),
    HalfBtf(Neg(48), 2, Neg(16), 5),
    3usize.into(),
    4usize.into(),
    HalfBtf(Neg(16), 2, Pos(48), 5),
    HalfBtf(Pos(48), 1, Pos(16), 6),
    7usize.into(),
  ];
  let stg4 = Stage::next(stg3, stg4);
  let stg4 = VarArr::let_(dst, format_args!("{}_idct16_stg4", disc), &stg4);

  let stg5 = vec![
    AddClamp(Pos(0), Pos(3)),
    AddClamp(Pos(1), Pos(2)),
    AddClamp(Pos(1), Neg(2)),
    AddClamp(Pos(0), Neg(3)),
    AddClamp(Neg(4), Pos(7)),
    AddClamp(Neg(5), Pos(6)),
    AddClamp(Pos(5), Pos(6)),
    AddClamp(Pos(4), Pos(7)),
  ];
  let stg5 = Stage::next(stg4, stg5);
  let stg5 = VarArr::let_(dst, format_args!("{}_idct16_stg5", disc), &stg5);

  let stg6 = vec![
    0usize.into(),
    1usize.into(),
    HalfBtf(Neg(32), 2, Pos(32), 5),
    HalfBtf(Neg(32), 3, Pos(32), 4),
    HalfBtf(Pos(32), 3, Pos(32), 4),
    HalfBtf(Pos(32), 2, Pos(32), 5),
    6usize.into(),
    7usize.into(),
  ];
  let stg6 = Stage::next(stg5, stg6);
  let stg6 = VarArr::let_(dst, format_args!("{}_idct16_stg6", disc), &stg6);

  let mut stg7 = Vec::new();
  for i in 0..8 {
    stg7.push((Pos(i), Pos(7 - i)));
  }
  for i in 0..8 {
    stg7.push((Pos(7 - i), Neg(i)));
  }

  VarArr::add_clamp_merge(
    dst,
    format_args!("{}_idct16", disc),
    idct8,
    stg6,
    &stg7,
  )
}
fn idct32<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
  where
    T: Array,
    U: Display,
{
  assert!(input.len() >= 32, "{} < {}", input.len(), 32);

  let stg1 = vec![
    0usize.into(),
    16usize.into(),
    8usize.into(),
    24usize.into(),
    4usize.into(),
    20usize.into(),
    12usize.into(),
    28usize.into(),
    2usize.into(),
    18usize.into(),
    10usize.into(),
    26usize.into(),
    6usize.into(),
    22usize.into(),
    14usize.into(),
    30usize.into(),
    1usize.into(),
    17usize.into(),
    9usize.into(),
    25usize.into(),
    5usize.into(),
    21usize.into(),
    13usize.into(),
    29usize.into(),
    3usize.into(),
    19usize.into(),
    11usize.into(),
    27usize.into(),
    7usize.into(),
    23usize.into(),
    15usize.into(),
    31usize.into(),
  ];
  let stg1 = Stage::next(input, stg1);
  let stg1 = VarArr::let_(dst, format_args!("{}_idct16_stg1", disc), &stg1);
  let stg2 = vec![
    0usize.into(),
    1usize.into(),
    2usize.into(),
    3usize.into(),
    4usize.into(),
    5usize.into(),
    6usize.into(),
    7usize.into(),
    8usize.into(),
    9usize.into(),
    10usize.into(),
    11usize.into(),
    12usize.into(),
    13usize.into(),
    14usize.into(),
    15usize.into(),
    HalfBtf(Pos ( 62usize ), 16usize, Neg ( 2usize ), 31usize, ),
    HalfBtf(Pos ( 30usize ), 17usize, Neg ( 34usize ), 30usize, ),
    HalfBtf(Pos ( 46usize ), 18usize, Neg ( 18usize ), 29usize, ),
    HalfBtf(Pos ( 14usize ), 19usize, Neg ( 50usize ), 28usize, ),
    HalfBtf(Pos ( 54usize ), 20usize, Neg ( 10usize ), 27usize, ),
    HalfBtf(Pos ( 22usize ), 21usize, Neg ( 42usize ), 26usize, ),
    HalfBtf(Pos ( 38usize ), 22usize, Neg ( 26usize ), 25usize, ),
    HalfBtf(Pos ( 6usize ), 23usize, Neg ( 58usize ), 24usize, ),
    HalfBtf(Pos ( 58usize ), 23usize, Pos ( 6usize ), 24usize, ),
    HalfBtf(Pos ( 26usize ), 22usize, Pos ( 38usize ), 25usize, ),
    HalfBtf(Pos ( 42usize ), 21usize, Pos ( 22usize ), 26usize, ),
    HalfBtf(Pos ( 10usize ), 20usize, Pos ( 54usize ), 27usize, ),
    HalfBtf(Pos ( 50usize ), 19usize, Pos ( 14usize ), 28usize, ),
    HalfBtf(Pos ( 18usize ), 18usize, Pos ( 46usize ), 29usize, ),
    HalfBtf(Pos ( 34usize ), 17usize, Pos ( 30usize ), 30usize, ),
    HalfBtf(Pos ( 2usize ), 16usize, Pos ( 62usize ), 31usize, ),
  ];
  let stg2 = Stage::next(stg1, stg2);
  let stg2 = VarArr::let_(dst, format_args!("{}_idct16_stg2", disc), &stg2);
  let stg3 = vec![
    0usize.into(),
    1usize.into(),
    2usize.into(),
    3usize.into(),
    4usize.into(),
    5usize.into(),
    6usize.into(),
    7usize.into(),
    HalfBtf(Pos ( 60usize ), 8usize, Neg ( 4usize ), 15usize, ),
    HalfBtf(Pos ( 28usize ), 9usize, Neg ( 36usize ), 14usize, ),
    HalfBtf(Pos ( 44usize ), 10usize, Neg ( 20usize ), 13usize, ),
    HalfBtf(Pos ( 12usize ), 11usize, Neg ( 52usize ), 12usize, ),
    HalfBtf(Pos ( 52usize ), 11usize, Pos ( 12usize ), 12usize, ),
    HalfBtf(Pos ( 20usize ), 10usize, Pos ( 44usize ), 13usize, ),
    HalfBtf(Pos ( 36usize ), 9usize, Pos ( 28usize ), 14usize, ),
    HalfBtf(Pos ( 4usize ), 8usize, Pos ( 60usize ), 15usize, ),
    AddClamp(Pos ( 16usize ), Pos ( 17usize )),
    AddClamp(Pos ( 16usize ), Neg ( 17usize )),
    AddClamp(Neg ( 18usize ), Pos ( 19usize )),
    AddClamp(Pos ( 18usize ), Pos ( 19usize )),
    AddClamp(Pos ( 20usize ), Pos ( 21usize )),
    AddClamp(Pos ( 20usize ), Neg ( 21usize )),
    AddClamp(Neg ( 22usize ), Pos ( 23usize )),
    AddClamp(Pos ( 22usize ), Pos ( 23usize )),
    AddClamp(Pos ( 24usize ), Pos ( 25usize )),
    AddClamp(Pos ( 24usize ), Neg ( 25usize )),
    AddClamp(Neg ( 26usize ), Pos ( 27usize )),
    AddClamp(Pos ( 26usize ), Pos ( 27usize )),
    AddClamp(Pos ( 28usize ), Pos ( 29usize )),
    AddClamp(Pos ( 28usize ), Neg ( 29usize )),
    AddClamp(Neg ( 30usize ), Pos ( 31usize )),
    AddClamp(Pos ( 30usize ), Pos ( 31usize )),
  ];
  let stg3 = Stage::next(stg2, stg3);
  let stg3 = VarArr::let_(dst, format_args!("{}_idct16_stg3", disc), &stg3);
  let stg4 = vec![
    0usize.into(),
    1usize.into(),
    2usize.into(),
    3usize.into(),
    HalfBtf(Pos ( 56usize ), 4usize, Neg ( 8usize ), 7usize, ),
    HalfBtf(Pos ( 24usize ), 5usize, Neg ( 40usize ), 6usize, ),
    HalfBtf(Pos ( 40usize ), 5usize, Pos ( 24usize ), 6usize, ),
    HalfBtf(Pos ( 8usize ), 4usize, Pos ( 56usize ), 7usize, ),
    AddClamp(Pos ( 8usize ), Pos ( 9usize )),
    AddClamp(Pos ( 8usize ), Neg ( 9usize )),
    AddClamp(Neg ( 10usize ), Pos ( 11usize )),
    AddClamp(Pos ( 10usize ), Pos ( 11usize )),
    AddClamp(Pos ( 12usize ), Pos ( 13usize )),
    AddClamp(Pos ( 12usize ), Neg ( 13usize )),
    AddClamp(Neg ( 14usize ), Pos ( 15usize )),
    AddClamp(Pos ( 14usize ), Pos ( 15usize )),
    16usize.into(),
    HalfBtf(Neg ( 8usize ), 17usize, Pos ( 56usize ), 30usize, ),
    HalfBtf(Neg ( 56usize ), 18usize, Neg ( 8usize ), 29usize, ),
    19usize.into(),
    20usize.into(),
    HalfBtf(Neg ( 40usize ), 21usize, Pos ( 24usize ), 26usize, ),
    HalfBtf(Neg ( 24usize ), 22usize, Neg ( 40usize ), 25usize, ),
    23usize.into(),
    24usize.into(),
    HalfBtf(Neg ( 40usize ), 22usize, Pos ( 24usize ), 25usize, ),
    HalfBtf(Pos ( 24usize ), 21usize, Pos ( 40usize ), 26usize, ),
    27usize.into(),
    28usize.into(),
    HalfBtf(Neg ( 8usize ), 18usize, Pos ( 56usize ), 29usize, ),
    HalfBtf(Pos ( 56usize ), 17usize, Pos ( 8usize ), 30usize, ),
    31usize.into(),
  ];
  let stg4 = Stage::next(stg3, stg4);
  let stg4 = VarArr::let_(dst, format_args!("{}_idct16_stg4", disc), &stg4);
  let stg5 = vec![
    HalfBtf(Pos ( 32usize ), 0usize, Pos ( 32usize ), 1usize, ),
    HalfBtf(Pos ( 32usize ), 0usize, Neg ( 32usize ), 1usize, ),
    HalfBtf(Pos ( 48usize ), 2usize, Neg ( 16usize ), 3usize, ),
    HalfBtf(Pos ( 16usize ), 2usize, Pos ( 48usize ), 3usize, ),
    AddClamp(Pos ( 4usize ), Pos ( 5usize )),
    AddClamp(Pos ( 4usize ), Neg ( 5usize )),
    AddClamp(Neg ( 6usize ), Pos ( 7usize )),
    AddClamp(Pos ( 6usize ), Pos ( 7usize )),
    8usize.into(),
    HalfBtf(Neg ( 16usize ), 9usize, Pos ( 48usize ), 14usize, ),
    HalfBtf(Neg ( 48usize ), 10usize, Neg ( 16usize ), 13usize, ),
    11usize.into(),
    12usize.into(),
    HalfBtf(Neg ( 16usize ), 10usize, Pos ( 48usize ), 13usize, ),
    HalfBtf(Pos ( 48usize ), 9usize, Pos ( 16usize ), 14usize, ),
    15usize.into(),
    AddClamp(Pos ( 16usize ), Pos ( 19usize )),
    AddClamp(Pos ( 17usize ), Pos ( 18usize )),
    AddClamp(Pos ( 17usize ), Neg ( 18usize )),
    AddClamp(Pos ( 16usize ), Neg ( 19usize )),
    AddClamp(Neg ( 20usize ), Pos ( 23usize )),
    AddClamp(Neg ( 21usize ), Pos ( 22usize )),
    AddClamp(Pos ( 21usize ), Pos ( 22usize )),
    AddClamp(Pos ( 20usize ), Pos ( 23usize )),
    AddClamp(Pos ( 24usize ), Pos ( 27usize )),
    AddClamp(Pos ( 25usize ), Pos ( 26usize )),
    AddClamp(Pos ( 25usize ), Neg ( 26usize )),
    AddClamp(Pos ( 24usize ), Neg ( 27usize )),
    AddClamp(Neg ( 28usize ), Pos ( 31usize )),
    AddClamp(Neg ( 29usize ), Pos ( 30usize )),
    AddClamp(Pos ( 29usize ), Pos ( 30usize )),
    AddClamp(Pos ( 28usize ), Pos ( 31usize )),
  ];
  let stg5 = Stage::next(stg4, stg5);
  let stg5 = VarArr::let_(dst, format_args!("{}_idct16_stg5", disc), &stg5);
  let stg6 = vec![
    AddClamp(Pos ( 0usize ), Pos ( 3usize )),
    AddClamp(Pos ( 1usize ), Pos ( 2usize )),
    AddClamp(Pos ( 1usize ), Neg ( 2usize )),
    AddClamp(Pos ( 0usize ), Neg ( 3usize )),
    4usize.into(),
    HalfBtf(Neg ( 32usize ), 5usize, Pos ( 32usize ), 6usize, ),
    HalfBtf(Pos ( 32usize ), 5usize, Pos ( 32usize ), 6usize, ),
    7usize.into(),
    AddClamp(Pos ( 8usize ), Pos ( 11usize )),
    AddClamp(Pos ( 9usize ), Pos ( 10usize )),
    AddClamp(Pos ( 9usize ), Neg ( 10usize )),
    AddClamp(Pos ( 8usize ), Neg ( 11usize )),
    AddClamp(Neg ( 12usize ), Pos ( 15usize )),
    AddClamp(Neg ( 13usize ), Pos ( 14usize )),
    AddClamp(Pos ( 13usize ), Pos ( 14usize )),
    AddClamp(Pos ( 12usize ), Pos ( 15usize )),
    16usize.into(),
    17usize.into(),
    HalfBtf(Neg ( 16usize ), 18usize, Pos ( 48usize ), 29usize, ),
    HalfBtf(Neg ( 16usize ), 19usize, Pos ( 48usize ), 28usize, ),
    HalfBtf(Neg ( 48usize ), 20usize, Neg ( 16usize ), 27usize, ),
    HalfBtf(Neg ( 48usize ), 21usize, Neg ( 16usize ), 26usize, ),
    22usize.into(),
    23usize.into(),
    24usize.into(),
    25usize.into(),
    HalfBtf(Neg ( 16usize ), 21usize, Pos ( 48usize ), 26usize, ),
    HalfBtf(Neg ( 16usize ), 20usize, Pos ( 48usize ), 27usize, ),
    HalfBtf(Pos ( 48usize ), 19usize, Pos ( 16usize ), 28usize, ),
    HalfBtf(Pos ( 48usize ), 18usize, Pos ( 16usize ), 29usize, ),
    30usize.into(),
    31usize.into(),
  ];
  let stg6 = Stage::next(stg5, stg6);
  let stg6 = VarArr::let_(dst, format_args!("{}_idct16_stg6", disc), &stg6);
  let stg7 = vec![
    AddClamp(Pos ( 0usize ), Pos ( 7usize )),
    AddClamp(Pos ( 1usize ), Pos ( 6usize )),
    AddClamp(Pos ( 2usize ), Pos ( 5usize )),
    AddClamp(Pos ( 3usize ), Pos ( 4usize )),
    AddClamp(Pos ( 3usize ), Neg ( 4usize )),
    AddClamp(Pos ( 2usize ), Neg ( 5usize )),
    AddClamp(Pos ( 1usize ), Neg ( 6usize )),
    AddClamp(Pos ( 0usize ), Neg ( 7usize )),
    8usize.into(),
    9usize.into(),
    HalfBtf(Neg ( 32usize ), 10usize, Pos ( 32usize ), 13usize, ),
    HalfBtf(Neg ( 32usize ), 11usize, Pos ( 32usize ), 12usize, ),
    HalfBtf(Pos ( 32usize ), 11usize, Pos ( 32usize ), 12usize, ),
    HalfBtf(Pos ( 32usize ), 10usize, Pos ( 32usize ), 13usize, ),
    14usize.into(),
    15usize.into(),
    AddClamp(Pos ( 16usize ), Pos ( 23usize )),
    AddClamp(Pos ( 17usize ), Pos ( 22usize )),
    AddClamp(Pos ( 18usize ), Pos ( 21usize )),
    AddClamp(Pos ( 19usize ), Pos ( 20usize )),
    AddClamp(Pos ( 19usize ), Neg ( 20usize )),
    AddClamp(Pos ( 18usize ), Neg ( 21usize )),
    AddClamp(Pos ( 17usize ), Neg ( 22usize )),
    AddClamp(Pos ( 16usize ), Neg ( 23usize )),
    AddClamp(Neg ( 24usize ), Pos ( 31usize )),
    AddClamp(Neg ( 25usize ), Pos ( 30usize )),
    AddClamp(Neg ( 26usize ), Pos ( 29usize )),
    AddClamp(Neg ( 27usize ), Pos ( 28usize )),
    AddClamp(Pos ( 27usize ), Pos ( 28usize )),
    AddClamp(Pos ( 26usize ), Pos ( 29usize )),
    AddClamp(Pos ( 25usize ), Pos ( 30usize )),
    AddClamp(Pos ( 24usize ), Pos ( 31usize )),
  ];
  let stg7 = Stage::next(stg6, stg7);
  let stg7 = VarArr::let_(dst, format_args!("{}_idct16_stg7", disc), &stg7);
  let stg8 = vec![
    AddClamp(Pos ( 0usize ), Pos ( 15usize )),
    AddClamp(Pos ( 1usize ), Pos ( 14usize )),
    AddClamp(Pos ( 2usize ), Pos ( 13usize )),
    AddClamp(Pos ( 3usize ), Pos ( 12usize )),
    AddClamp(Pos ( 4usize ), Pos ( 11usize )),
    AddClamp(Pos ( 5usize ), Pos ( 10usize )),
    AddClamp(Pos ( 6usize ), Pos ( 9usize )),
    AddClamp(Pos ( 7usize ), Pos ( 8usize )),
    AddClamp(Pos ( 7usize ), Neg ( 8usize )),
    AddClamp(Pos ( 6usize ), Neg ( 9usize )),
    AddClamp(Pos ( 5usize ), Neg ( 10usize )),
    AddClamp(Pos ( 4usize ), Neg ( 11usize )),
    AddClamp(Pos ( 3usize ), Neg ( 12usize )),
    AddClamp(Pos ( 2usize ), Neg ( 13usize )),
    AddClamp(Pos ( 1usize ), Neg ( 14usize )),
    AddClamp(Pos ( 0usize ), Neg ( 15usize )),
    16usize.into(),
    17usize.into(),
    18usize.into(),
    19usize.into(),
    HalfBtf(Neg ( 32usize ), 20usize, Pos ( 32usize ), 27usize, ),
    HalfBtf(Neg ( 32usize ), 21usize, Pos ( 32usize ), 26usize, ),
    HalfBtf(Neg ( 32usize ), 22usize, Pos ( 32usize ), 25usize, ),
    HalfBtf(Neg ( 32usize ), 23usize, Pos ( 32usize ), 24usize, ),
    HalfBtf(Pos ( 32usize ), 23usize, Pos ( 32usize ), 24usize, ),
    HalfBtf(Pos ( 32usize ), 22usize, Pos ( 32usize ), 25usize, ),
    HalfBtf(Pos ( 32usize ), 21usize, Pos ( 32usize ), 26usize, ),
    HalfBtf(Pos ( 32usize ), 20usize, Pos ( 32usize ), 27usize, ),
    28usize.into(),
    29usize.into(),
    30usize.into(),
    31usize.into(),
  ];
  let stg8 = Stage::next(stg7, stg8);
  let stg8 = VarArr::let_(dst, format_args!("{}_idct16_stg8", disc), &stg8);

  let mut stg9 = Vec::new();
  for i in 0..16 {
    stg9.push(AddClamp(Pos(i), Pos(31 - i)));
  }
  for i in 0..16 {
    stg9.push(AddClamp(Pos(15 - i), Neg(i + 16)));
  }
  let stg9 = Stage::next(stg8, stg9);
  let stg9 = VarArr::let_(dst, format_args!("{}_idct32", disc),
                          &stg9);

  stg9
}
fn idct64<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
  where T: Array, U: Display,
{
  let stg1 = vec![
    0usize.into(),
    32usize.into(),
    16usize.into(),
    48usize.into(),
    8usize.into(),
    40usize.into(),
    24usize.into(),
    56usize.into(),
    4usize.into(),
    36usize.into(),
    20usize.into(),
    52usize.into(),
    12usize.into(),
    44usize.into(),
    28usize.into(),
    60usize.into(),
    2usize.into(),
    34usize.into(),
    18usize.into(),
    50usize.into(),
    10usize.into(),
    42usize.into(),
    26usize.into(),
    58usize.into(),
    6usize.into(),
    38usize.into(),
    22usize.into(),
    54usize.into(),
    14usize.into(),
    46usize.into(),
    30usize.into(),
    62usize.into(),
    1usize.into(),
    33usize.into(),
    17usize.into(),
    49usize.into(),
    9usize.into(),
    41usize.into(),
    25usize.into(),
    57usize.into(),
    5usize.into(),
    37usize.into(),
    21usize.into(),
    53usize.into(),
    13usize.into(),
    45usize.into(),
    29usize.into(),
    61usize.into(),
    3usize.into(),
    35usize.into(),
    19usize.into(),
    51usize.into(),
    11usize.into(),
    43usize.into(),
    27usize.into(),
    59usize.into(),
    7usize.into(),
    39usize.into(),
    23usize.into(),
    55usize.into(),
    15usize.into(),
    47usize.into(),
    31usize.into(),
    63usize.into(),
  ];
  let stg1 = Stage::next(input, stg1);
  let stg1 = VarArr::let_(dst, format_args!("{}_idct64_stg1", disc), &stg1);
  let stg2 = vec![
    0usize.into(),
    1usize.into(),
    2usize.into(),
    3usize.into(),
    4usize.into(),
    5usize.into(),
    6usize.into(),
    7usize.into(),
    8usize.into(),
    9usize.into(),
    10usize.into(),
    11usize.into(),
    12usize.into(),
    13usize.into(),
    14usize.into(),
    15usize.into(),
    16usize.into(),
    17usize.into(),
    18usize.into(),
    19usize.into(),
    20usize.into(),
    21usize.into(),
    22usize.into(),
    23usize.into(),
    24usize.into(),
    25usize.into(),
    26usize.into(),
    27usize.into(),
    28usize.into(),
    29usize.into(),
    30usize.into(),
    31usize.into(),
    HalfBtf(Pos ( 63usize ), 32usize, Neg ( 1usize ), 63usize, ),
    HalfBtf(Pos ( 31usize ), 33usize, Neg ( 33usize ), 62usize, ),
    HalfBtf(Pos ( 47usize ), 34usize, Neg ( 17usize ), 61usize, ),
    HalfBtf(Pos ( 15usize ), 35usize, Neg ( 49usize ), 60usize, ),
    HalfBtf(Pos ( 55usize ), 36usize, Neg ( 9usize ), 59usize, ),
    HalfBtf(Pos ( 23usize ), 37usize, Neg ( 41usize ), 58usize, ),
    HalfBtf(Pos ( 39usize ), 38usize, Neg ( 25usize ), 57usize, ),
    HalfBtf(Pos ( 7usize ), 39usize, Neg ( 57usize ), 56usize, ),
    HalfBtf(Pos ( 59usize ), 40usize, Neg ( 5usize ), 55usize, ),
    HalfBtf(Pos ( 27usize ), 41usize, Neg ( 37usize ), 54usize, ),
    HalfBtf(Pos ( 43usize ), 42usize, Neg ( 21usize ), 53usize, ),
    HalfBtf(Pos ( 11usize ), 43usize, Neg ( 53usize ), 52usize, ),
    HalfBtf(Pos ( 51usize ), 44usize, Neg ( 13usize ), 51usize, ),
    HalfBtf(Pos ( 19usize ), 45usize, Neg ( 45usize ), 50usize, ),
    HalfBtf(Pos ( 35usize ), 46usize, Neg ( 29usize ), 49usize, ),
    HalfBtf(Pos ( 3usize ), 47usize, Neg ( 61usize ), 48usize, ),
    HalfBtf(Pos ( 61usize ), 47usize, Pos ( 3usize ), 48usize, ),
    HalfBtf(Pos ( 29usize ), 46usize, Pos ( 35usize ), 49usize, ),
    HalfBtf(Pos ( 45usize ), 45usize, Pos ( 19usize ), 50usize, ),
    HalfBtf(Pos ( 13usize ), 44usize, Pos ( 51usize ), 51usize, ),
    HalfBtf(Pos ( 53usize ), 43usize, Pos ( 11usize ), 52usize, ),
    HalfBtf(Pos ( 21usize ), 42usize, Pos ( 43usize ), 53usize, ),
    HalfBtf(Pos ( 37usize ), 41usize, Pos ( 27usize ), 54usize, ),
    HalfBtf(Pos ( 5usize ), 40usize, Pos ( 59usize ), 55usize, ),
    HalfBtf(Pos ( 57usize ), 39usize, Pos ( 7usize ), 56usize, ),
    HalfBtf(Pos ( 25usize ), 38usize, Pos ( 39usize ), 57usize, ),
    HalfBtf(Pos ( 41usize ), 37usize, Pos ( 23usize ), 58usize, ),
    HalfBtf(Pos ( 9usize ), 36usize, Pos ( 55usize ), 59usize, ),
    HalfBtf(Pos ( 49usize ), 35usize, Pos ( 15usize ), 60usize, ),
    HalfBtf(Pos ( 17usize ), 34usize, Pos ( 47usize ), 61usize, ),
    HalfBtf(Pos ( 33usize ), 33usize, Pos ( 31usize ), 62usize, ),
    HalfBtf(Pos ( 1usize ), 32usize, Pos ( 63usize ), 63usize, ),
  ];
  let stg2 = Stage::next(stg1, stg2);
  let stg2 = VarArr::let_(dst, format_args!("{}_idct64_stg2", disc), &stg2);
  let stg3 = vec![
    0usize.into(),
    1usize.into(),
    2usize.into(),
    3usize.into(),
    4usize.into(),
    5usize.into(),
    6usize.into(),
    7usize.into(),
    8usize.into(),
    9usize.into(),
    10usize.into(),
    11usize.into(),
    12usize.into(),
    13usize.into(),
    14usize.into(),
    15usize.into(),
    HalfBtf(Pos ( 62usize ), 16usize, Neg ( 2usize ), 31usize, ),
    HalfBtf(Pos ( 30usize ), 17usize, Neg ( 34usize ), 30usize, ),
    HalfBtf(Pos ( 46usize ), 18usize, Neg ( 18usize ), 29usize, ),
    HalfBtf(Pos ( 14usize ), 19usize, Neg ( 50usize ), 28usize, ),
    HalfBtf(Pos ( 54usize ), 20usize, Neg ( 10usize ), 27usize, ),
    HalfBtf(Pos ( 22usize ), 21usize, Neg ( 42usize ), 26usize, ),
    HalfBtf(Pos ( 38usize ), 22usize, Neg ( 26usize ), 25usize, ),
    HalfBtf(Pos ( 6usize ), 23usize, Neg ( 58usize ), 24usize, ),
    HalfBtf(Pos ( 58usize ), 23usize, Pos ( 6usize ), 24usize, ),
    HalfBtf(Pos ( 26usize ), 22usize, Pos ( 38usize ), 25usize, ),
    HalfBtf(Pos ( 42usize ), 21usize, Pos ( 22usize ), 26usize, ),
    HalfBtf(Pos ( 10usize ), 20usize, Pos ( 54usize ), 27usize, ),
    HalfBtf(Pos ( 50usize ), 19usize, Pos ( 14usize ), 28usize, ),
    HalfBtf(Pos ( 18usize ), 18usize, Pos ( 46usize ), 29usize, ),
    HalfBtf(Pos ( 34usize ), 17usize, Pos ( 30usize ), 30usize, ),
    HalfBtf(Pos ( 2usize ), 16usize, Pos ( 62usize ), 31usize, ),
    AddClamp(Pos ( 32usize ), Pos ( 33usize )),
    AddClamp(Pos ( 32usize ), Neg ( 33usize )),
    AddClamp(Neg ( 34usize ), Pos ( 35usize )),
    AddClamp(Pos ( 34usize ), Pos ( 35usize )),
    AddClamp(Pos ( 36usize ), Pos ( 37usize )),
    AddClamp(Pos ( 36usize ), Neg ( 37usize )),
    AddClamp(Neg ( 38usize ), Pos ( 39usize )),
    AddClamp(Pos ( 38usize ), Pos ( 39usize )),
    AddClamp(Pos ( 40usize ), Pos ( 41usize )),
    AddClamp(Pos ( 40usize ), Neg ( 41usize )),
    AddClamp(Neg ( 42usize ), Pos ( 43usize )),
    AddClamp(Pos ( 42usize ), Pos ( 43usize )),
    AddClamp(Pos ( 44usize ), Pos ( 45usize )),
    AddClamp(Pos ( 44usize ), Neg ( 45usize )),
    AddClamp(Neg ( 46usize ), Pos ( 47usize )),
    AddClamp(Pos ( 46usize ), Pos ( 47usize )),
    AddClamp(Pos ( 48usize ), Pos ( 49usize )),
    AddClamp(Pos ( 48usize ), Neg ( 49usize )),
    AddClamp(Neg ( 50usize ), Pos ( 51usize )),
    AddClamp(Pos ( 50usize ), Pos ( 51usize )),
    AddClamp(Pos ( 52usize ), Pos ( 53usize )),
    AddClamp(Pos ( 52usize ), Neg ( 53usize )),
    AddClamp(Neg ( 54usize ), Pos ( 55usize )),
    AddClamp(Pos ( 54usize ), Pos ( 55usize )),
    AddClamp(Pos ( 56usize ), Pos ( 57usize )),
    AddClamp(Pos ( 56usize ), Neg ( 57usize )),
    AddClamp(Neg ( 58usize ), Pos ( 59usize )),
    AddClamp(Pos ( 58usize ), Pos ( 59usize )),
    AddClamp(Pos ( 60usize ), Pos ( 61usize )),
    AddClamp(Pos ( 60usize ), Neg ( 61usize )),
    AddClamp(Neg ( 62usize ), Pos ( 63usize )),
    AddClamp(Pos ( 62usize ), Pos ( 63usize )),
  ];
  let stg3 = Stage::next(stg2, stg3);
  let stg3 = VarArr::let_(dst, format_args!("{}_idct64_stg3", disc), &stg3);
  let stg4 = vec![
    0usize.into(),
    1usize.into(),
    2usize.into(),
    3usize.into(),
    4usize.into(),
    5usize.into(),
    6usize.into(),
    7usize.into(),
    HalfBtf(Pos ( 60usize ), 8usize, Neg ( 4usize ), 15usize, ),
    HalfBtf(Pos ( 28usize ), 9usize, Neg ( 36usize ), 14usize, ),
    HalfBtf(Pos ( 44usize ), 10usize, Neg ( 20usize ), 13usize, ),
    HalfBtf(Pos ( 12usize ), 11usize, Neg ( 52usize ), 12usize, ),
    HalfBtf(Pos ( 52usize ), 11usize, Pos ( 12usize ), 12usize, ),
    HalfBtf(Pos ( 20usize ), 10usize, Pos ( 44usize ), 13usize, ),
    HalfBtf(Pos ( 36usize ), 9usize, Pos ( 28usize ), 14usize, ),
    HalfBtf(Pos ( 4usize ), 8usize, Pos ( 60usize ), 15usize, ),
    AddClamp(Pos ( 16usize ), Pos ( 17usize )),
    AddClamp(Pos ( 16usize ), Neg ( 17usize )),
    AddClamp(Neg ( 18usize ), Pos ( 19usize )),
    AddClamp(Pos ( 18usize ), Pos ( 19usize )),
    AddClamp(Pos ( 20usize ), Pos ( 21usize )),
    AddClamp(Pos ( 20usize ), Neg ( 21usize )),
    AddClamp(Neg ( 22usize ), Pos ( 23usize )),
    AddClamp(Pos ( 22usize ), Pos ( 23usize )),
    AddClamp(Pos ( 24usize ), Pos ( 25usize )),
    AddClamp(Pos ( 24usize ), Neg ( 25usize )),
    AddClamp(Neg ( 26usize ), Pos ( 27usize )),
    AddClamp(Pos ( 26usize ), Pos ( 27usize )),
    AddClamp(Pos ( 28usize ), Pos ( 29usize )),
    AddClamp(Pos ( 28usize ), Neg ( 29usize )),
    AddClamp(Neg ( 30usize ), Pos ( 31usize )),
    AddClamp(Pos ( 30usize ), Pos ( 31usize )),
    32usize.into(),
    HalfBtf(Neg ( 4usize ), 33usize, Pos ( 60usize ), 62usize, ),
    HalfBtf(Neg ( 60usize ), 34usize, Neg ( 4usize ), 61usize, ),
    35usize.into(),
    36usize.into(),
    HalfBtf(Neg ( 36usize ), 37usize, Pos ( 28usize ), 58usize, ),
    HalfBtf(Neg ( 28usize ), 38usize, Neg ( 36usize ), 57usize, ),
    39usize.into(),
    40usize.into(),
    HalfBtf(Neg ( 20usize ), 41usize, Pos ( 44usize ), 54usize, ),
    HalfBtf(Neg ( 44usize ), 42usize, Neg ( 20usize ), 53usize, ),
    43usize.into(),
    44usize.into(),
    HalfBtf(Neg ( 52usize ), 45usize, Pos ( 12usize ), 50usize, ),
    HalfBtf(Neg ( 12usize ), 46usize, Neg ( 52usize ), 49usize, ),
    47usize.into(),
    48usize.into(),
    HalfBtf(Neg ( 52usize ), 46usize, Pos ( 12usize ), 49usize, ),
    HalfBtf(Pos ( 12usize ), 45usize, Pos ( 52usize ), 50usize, ),
    51usize.into(),
    52usize.into(),
    HalfBtf(Neg ( 20usize ), 42usize, Pos ( 44usize ), 53usize, ),
    HalfBtf(Pos ( 44usize ), 41usize, Pos ( 20usize ), 54usize, ),
    55usize.into(),
    56usize.into(),
    HalfBtf(Neg ( 36usize ), 38usize, Pos ( 28usize ), 57usize, ),
    HalfBtf(Pos ( 28usize ), 37usize, Pos ( 36usize ), 58usize, ),
    59usize.into(),
    60usize.into(),
    HalfBtf(Neg ( 4usize ), 34usize, Pos ( 60usize ), 61usize, ),
    HalfBtf(Pos ( 60usize ), 33usize, Pos ( 4usize ), 62usize, ),
    63usize.into(),
  ];
  let stg4 = Stage::next(stg3, stg4);
  let stg4 = VarArr::let_(dst, format_args!("{}_idct64_stg4", disc), &stg4);
  let stg5 = vec![
    0usize.into(),
    1usize.into(),
    2usize.into(),
    3usize.into(),
    HalfBtf(Pos ( 56usize ), 4usize, Neg ( 8usize ), 7usize, ),
    HalfBtf(Pos ( 24usize ), 5usize, Neg ( 40usize ), 6usize, ),
    HalfBtf(Pos ( 40usize ), 5usize, Pos ( 24usize ), 6usize, ),
    HalfBtf(Pos ( 8usize ), 4usize, Pos ( 56usize ), 7usize, ),
    AddClamp(Pos ( 8usize ), Pos ( 9usize )),
    AddClamp(Pos ( 8usize ), Neg ( 9usize )),
    AddClamp(Neg ( 10usize ), Pos ( 11usize )),
    AddClamp(Pos ( 10usize ), Pos ( 11usize )),
    AddClamp(Pos ( 12usize ), Pos ( 13usize )),
    AddClamp(Pos ( 12usize ), Neg ( 13usize )),
    AddClamp(Neg ( 14usize ), Pos ( 15usize )),
    AddClamp(Pos ( 14usize ), Pos ( 15usize )),
    16usize.into(),
    HalfBtf(Neg ( 8usize ), 17usize, Pos ( 56usize ), 30usize, ),
    HalfBtf(Neg ( 56usize ), 18usize, Neg ( 8usize ), 29usize, ),
    19usize.into(),
    20usize.into(),
    HalfBtf(Neg ( 40usize ), 21usize, Pos ( 24usize ), 26usize, ),
    HalfBtf(Neg ( 24usize ), 22usize, Neg ( 40usize ), 25usize, ),
    23usize.into(),
    24usize.into(),
    HalfBtf(Neg ( 40usize ), 22usize, Pos ( 24usize ), 25usize, ),
    HalfBtf(Pos ( 24usize ), 21usize, Pos ( 40usize ), 26usize, ),
    27usize.into(),
    28usize.into(),
    HalfBtf(Neg ( 8usize ), 18usize, Pos ( 56usize ), 29usize, ),
    HalfBtf(Pos ( 56usize ), 17usize, Pos ( 8usize ), 30usize, ),
    31usize.into(),
    AddClamp(Pos ( 32usize ), Pos ( 35usize )),
    AddClamp(Pos ( 33usize ), Pos ( 34usize )),
    AddClamp(Pos ( 33usize ), Neg ( 34usize )),
    AddClamp(Pos ( 32usize ), Neg ( 35usize )),
    AddClamp(Neg ( 36usize ), Pos ( 39usize )),
    AddClamp(Neg ( 37usize ), Pos ( 38usize )),
    AddClamp(Pos ( 37usize ), Pos ( 38usize )),
    AddClamp(Pos ( 36usize ), Pos ( 39usize )),
    AddClamp(Pos ( 40usize ), Pos ( 43usize )),
    AddClamp(Pos ( 41usize ), Pos ( 42usize )),
    AddClamp(Pos ( 41usize ), Neg ( 42usize )),
    AddClamp(Pos ( 40usize ), Neg ( 43usize )),
    AddClamp(Neg ( 44usize ), Pos ( 47usize )),
    AddClamp(Neg ( 45usize ), Pos ( 46usize )),
    AddClamp(Pos ( 45usize ), Pos ( 46usize )),
    AddClamp(Pos ( 44usize ), Pos ( 47usize )),
    AddClamp(Pos ( 48usize ), Pos ( 51usize )),
    AddClamp(Pos ( 49usize ), Pos ( 50usize )),
    AddClamp(Pos ( 49usize ), Neg ( 50usize )),
    AddClamp(Pos ( 48usize ), Neg ( 51usize )),
    AddClamp(Neg ( 52usize ), Pos ( 55usize )),
    AddClamp(Neg ( 53usize ), Pos ( 54usize )),
    AddClamp(Pos ( 53usize ), Pos ( 54usize )),
    AddClamp(Pos ( 52usize ), Pos ( 55usize )),
    AddClamp(Pos ( 56usize ), Pos ( 59usize )),
    AddClamp(Pos ( 57usize ), Pos ( 58usize )),
    AddClamp(Pos ( 57usize ), Neg ( 58usize )),
    AddClamp(Pos ( 56usize ), Neg ( 59usize )),
    AddClamp(Neg ( 60usize ), Pos ( 63usize )),
    AddClamp(Neg ( 61usize ), Pos ( 62usize )),
    AddClamp(Pos ( 61usize ), Pos ( 62usize )),
    AddClamp(Pos ( 60usize ), Pos ( 63usize )),
  ];
  let stg5 = Stage::next(stg4, stg5);
  let stg5 = VarArr::let_(dst, format_args!("{}_idct64_stg5", disc), &stg5);
  let stg6 = vec![
    HalfBtf(Pos ( 32usize ), 0usize, Pos ( 32usize ), 1usize, ),
    HalfBtf(Pos ( 32usize ), 0usize, Neg ( 32usize ), 1usize, ),
    HalfBtf(Pos ( 48usize ), 2usize, Neg ( 16usize ), 3usize, ),
    HalfBtf(Pos ( 16usize ), 2usize, Pos ( 48usize ), 3usize, ),
    AddClamp(Pos ( 4usize ), Pos ( 5usize )),
    AddClamp(Pos ( 4usize ), Neg ( 5usize )),
    AddClamp(Neg ( 6usize ), Pos ( 7usize )),
    AddClamp(Pos ( 6usize ), Pos ( 7usize )),
    8usize.into(),
    HalfBtf(Neg ( 16usize ), 9usize, Pos ( 48usize ), 14usize, ),
    HalfBtf(Neg ( 48usize ), 10usize, Neg ( 16usize ), 13usize, ),
    11usize.into(),
    12usize.into(),
    HalfBtf(Neg ( 16usize ), 10usize, Pos ( 48usize ), 13usize, ),
    HalfBtf(Pos ( 48usize ), 9usize, Pos ( 16usize ), 14usize, ),
    15usize.into(),
    AddClamp(Pos ( 16usize ), Pos ( 19usize )),
    AddClamp(Pos ( 17usize ), Pos ( 18usize )),
    AddClamp(Pos ( 17usize ), Neg ( 18usize )),
    AddClamp(Pos ( 16usize ), Neg ( 19usize )),
    AddClamp(Neg ( 20usize ), Pos ( 23usize )),
    AddClamp(Neg ( 21usize ), Pos ( 22usize )),
    AddClamp(Pos ( 21usize ), Pos ( 22usize )),
    AddClamp(Pos ( 20usize ), Pos ( 23usize )),
    AddClamp(Pos ( 24usize ), Pos ( 27usize )),
    AddClamp(Pos ( 25usize ), Pos ( 26usize )),
    AddClamp(Pos ( 25usize ), Neg ( 26usize )),
    AddClamp(Pos ( 24usize ), Neg ( 27usize )),
    AddClamp(Neg ( 28usize ), Pos ( 31usize )),
    AddClamp(Neg ( 29usize ), Pos ( 30usize )),
    AddClamp(Pos ( 29usize ), Pos ( 30usize )),
    AddClamp(Pos ( 28usize ), Pos ( 31usize )),
    32usize.into(),
    33usize.into(),
    HalfBtf(Neg ( 8usize ), 34usize, Pos ( 56usize ), 61usize, ),
    HalfBtf(Neg ( 8usize ), 35usize, Pos ( 56usize ), 60usize, ),
    HalfBtf(Neg ( 56usize ), 36usize, Neg ( 8usize ), 59usize, ),
    HalfBtf(Neg ( 56usize ), 37usize, Neg ( 8usize ), 58usize, ),
    38usize.into(),
    39usize.into(),
    40usize.into(),
    41usize.into(),
    HalfBtf(Neg ( 40usize ), 42usize, Pos ( 24usize ), 53usize, ),
    HalfBtf(Neg ( 40usize ), 43usize, Pos ( 24usize ), 52usize, ),
    HalfBtf(Neg ( 24usize ), 44usize, Neg ( 40usize ), 51usize, ),
    HalfBtf(Neg ( 24usize ), 45usize, Neg ( 40usize ), 50usize, ),
    46usize.into(),
    47usize.into(),
    48usize.into(),
    49usize.into(),
    HalfBtf(Neg ( 40usize ), 45usize, Pos ( 24usize ), 50usize, ),
    HalfBtf(Neg ( 40usize ), 44usize, Pos ( 24usize ), 51usize, ),
    HalfBtf(Pos ( 24usize ), 43usize, Pos ( 40usize ), 52usize, ),
    HalfBtf(Pos ( 24usize ), 42usize, Pos ( 40usize ), 53usize, ),
    54usize.into(),
    55usize.into(),
    56usize.into(),
    57usize.into(),
    HalfBtf(Neg ( 8usize ), 37usize, Pos ( 56usize ), 58usize, ),
    HalfBtf(Neg ( 8usize ), 36usize, Pos ( 56usize ), 59usize, ),
    HalfBtf(Pos ( 56usize ), 35usize, Pos ( 8usize ), 60usize, ),
    HalfBtf(Pos ( 56usize ), 34usize, Pos ( 8usize ), 61usize, ),
    62usize.into(),
    63usize.into(),
  ];
  let stg6 = Stage::next(stg5, stg6);
  let stg6 = VarArr::let_(dst, format_args!("{}_idct64_stg6", disc), &stg6);
  let stg7 = vec![
    AddClamp(Pos ( 0usize ), Pos ( 3usize )),
    AddClamp(Pos ( 1usize ), Pos ( 2usize )),
    AddClamp(Pos ( 1usize ), Neg ( 2usize )),
    AddClamp(Pos ( 0usize ), Neg ( 3usize )),
    4usize.into(),
    HalfBtf(Neg ( 32usize ), 5usize, Pos ( 32usize ), 6usize, ),
    HalfBtf(Pos ( 32usize ), 5usize, Pos ( 32usize ), 6usize, ),
    7usize.into(),
    AddClamp(Pos ( 8usize ), Pos ( 11usize )),
    AddClamp(Pos ( 9usize ), Pos ( 10usize )),
    AddClamp(Pos ( 9usize ), Neg ( 10usize )),
    AddClamp(Pos ( 8usize ), Neg ( 11usize )),
    AddClamp(Neg ( 12usize ), Pos ( 15usize )),
    AddClamp(Neg ( 13usize ), Pos ( 14usize )),
    AddClamp(Pos ( 13usize ), Pos ( 14usize )),
    AddClamp(Pos ( 12usize ), Pos ( 15usize )),
    16usize.into(),
    17usize.into(),
    HalfBtf(Neg ( 16usize ), 18usize, Pos ( 48usize ), 29usize, ),
    HalfBtf(Neg ( 16usize ), 19usize, Pos ( 48usize ), 28usize, ),
    HalfBtf(Neg ( 48usize ), 20usize, Neg ( 16usize ), 27usize, ),
    HalfBtf(Neg ( 48usize ), 21usize, Neg ( 16usize ), 26usize, ),
    22usize.into(),
    23usize.into(),
    24usize.into(),
    25usize.into(),
    HalfBtf(Neg ( 16usize ), 21usize, Pos ( 48usize ), 26usize, ),
    HalfBtf(Neg ( 16usize ), 20usize, Pos ( 48usize ), 27usize, ),
    HalfBtf(Pos ( 48usize ), 19usize, Pos ( 16usize ), 28usize, ),
    HalfBtf(Pos ( 48usize ), 18usize, Pos ( 16usize ), 29usize, ),
    30usize.into(),
    31usize.into(),
    AddClamp(Pos ( 32usize ), Pos ( 39usize )),
    AddClamp(Pos ( 33usize ), Pos ( 38usize )),
    AddClamp(Pos ( 34usize ), Pos ( 37usize )),
    AddClamp(Pos ( 35usize ), Pos ( 36usize )),
    AddClamp(Pos ( 35usize ), Neg ( 36usize )),
    AddClamp(Pos ( 34usize ), Neg ( 37usize )),
    AddClamp(Pos ( 33usize ), Neg ( 38usize )),
    AddClamp(Pos ( 32usize ), Neg ( 39usize )),
    AddClamp(Neg ( 40usize ), Pos ( 47usize )),
    AddClamp(Neg ( 41usize ), Pos ( 46usize )),
    AddClamp(Neg ( 42usize ), Pos ( 45usize )),
    AddClamp(Neg ( 43usize ), Pos ( 44usize )),
    AddClamp(Pos ( 43usize ), Pos ( 44usize )),
    AddClamp(Pos ( 42usize ), Pos ( 45usize )),
    AddClamp(Pos ( 41usize ), Pos ( 46usize )),
    AddClamp(Pos ( 40usize ), Pos ( 47usize )),
    AddClamp(Pos ( 48usize ), Pos ( 55usize )),
    AddClamp(Pos ( 49usize ), Pos ( 54usize )),
    AddClamp(Pos ( 50usize ), Pos ( 53usize )),
    AddClamp(Pos ( 51usize ), Pos ( 52usize )),
    AddClamp(Pos ( 51usize ), Neg ( 52usize )),
    AddClamp(Pos ( 50usize ), Neg ( 53usize )),
    AddClamp(Pos ( 49usize ), Neg ( 54usize )),
    AddClamp(Pos ( 48usize ), Neg ( 55usize )),
    AddClamp(Neg ( 56usize ), Pos ( 63usize )),
    AddClamp(Neg ( 57usize ), Pos ( 62usize )),
    AddClamp(Neg ( 58usize ), Pos ( 61usize )),
    AddClamp(Neg ( 59usize ), Pos ( 60usize )),
    AddClamp(Pos ( 59usize ), Pos ( 60usize )),
    AddClamp(Pos ( 58usize ), Pos ( 61usize )),
    AddClamp(Pos ( 57usize ), Pos ( 62usize )),
    AddClamp(Pos ( 56usize ), Pos ( 63usize )),
  ];
  let stg7 = Stage::next(stg6, stg7);
  let stg7 = VarArr::let_(dst, format_args!("{}_idct64_stg7", disc), &stg7);
  let stg8 = vec![
    AddClamp(Pos ( 0usize ), Pos ( 7usize )),
    AddClamp(Pos ( 1usize ), Pos ( 6usize )),
    AddClamp(Pos ( 2usize ), Pos ( 5usize )),
    AddClamp(Pos ( 3usize ), Pos ( 4usize )),
    AddClamp(Pos ( 3usize ), Neg ( 4usize )),
    AddClamp(Pos ( 2usize ), Neg ( 5usize )),
    AddClamp(Pos ( 1usize ), Neg ( 6usize )),
    AddClamp(Pos ( 0usize ), Neg ( 7usize )),
    8usize.into(),
    9usize.into(),
    HalfBtf(Neg ( 32usize ), 10usize, Pos ( 32usize ), 13usize, ),
    HalfBtf(Neg ( 32usize ), 11usize, Pos ( 32usize ), 12usize, ),
    HalfBtf(Pos ( 32usize ), 11usize, Pos ( 32usize ), 12usize, ),
    HalfBtf(Pos ( 32usize ), 10usize, Pos ( 32usize ), 13usize, ),
    14usize.into(),
    15usize.into(),
    AddClamp(Pos ( 16usize ), Pos ( 23usize )),
    AddClamp(Pos ( 17usize ), Pos ( 22usize )),
    AddClamp(Pos ( 18usize ), Pos ( 21usize )),
    AddClamp(Pos ( 19usize ), Pos ( 20usize )),
    AddClamp(Pos ( 19usize ), Neg ( 20usize )),
    AddClamp(Pos ( 18usize ), Neg ( 21usize )),
    AddClamp(Pos ( 17usize ), Neg ( 22usize )),
    AddClamp(Pos ( 16usize ), Neg ( 23usize )),
    AddClamp(Neg ( 24usize ), Pos ( 31usize )),
    AddClamp(Neg ( 25usize ), Pos ( 30usize )),
    AddClamp(Neg ( 26usize ), Pos ( 29usize )),
    AddClamp(Neg ( 27usize ), Pos ( 28usize )),
    AddClamp(Pos ( 27usize ), Pos ( 28usize )),
    AddClamp(Pos ( 26usize ), Pos ( 29usize )),
    AddClamp(Pos ( 25usize ), Pos ( 30usize )),
    AddClamp(Pos ( 24usize ), Pos ( 31usize )),
    32usize.into(),
    33usize.into(),
    34usize.into(),
    35usize.into(),
    HalfBtf(Neg ( 16usize ), 36usize, Pos ( 48usize ), 59usize, ),
    HalfBtf(Neg ( 16usize ), 37usize, Pos ( 48usize ), 58usize, ),
    HalfBtf(Neg ( 16usize ), 38usize, Pos ( 48usize ), 57usize, ),
    HalfBtf(Neg ( 16usize ), 39usize, Pos ( 48usize ), 56usize, ),
    HalfBtf(Neg ( 48usize ), 40usize, Neg ( 16usize ), 55usize, ),
    HalfBtf(Neg ( 48usize ), 41usize, Neg ( 16usize ), 54usize, ),
    HalfBtf(Neg ( 48usize ), 42usize, Neg ( 16usize ), 53usize, ),
    HalfBtf(Neg ( 48usize ), 43usize, Neg ( 16usize ), 52usize, ),
    44usize.into(),
    45usize.into(),
    46usize.into(),
    47usize.into(),
    48usize.into(),
    49usize.into(),
    50usize.into(),
    51usize.into(),
    HalfBtf(Neg ( 16usize ), 43usize, Pos ( 48usize ), 52usize, ),
    HalfBtf(Neg ( 16usize ), 42usize, Pos ( 48usize ), 53usize, ),
    HalfBtf(Neg ( 16usize ), 41usize, Pos ( 48usize ), 54usize, ),
    HalfBtf(Neg ( 16usize ), 40usize, Pos ( 48usize ), 55usize, ),
    HalfBtf(Pos ( 48usize ), 39usize, Pos ( 16usize ), 56usize, ),
    HalfBtf(Pos ( 48usize ), 38usize, Pos ( 16usize ), 57usize, ),
    HalfBtf(Pos ( 48usize ), 37usize, Pos ( 16usize ), 58usize, ),
    HalfBtf(Pos ( 48usize ), 36usize, Pos ( 16usize ), 59usize, ),
    60usize.into(),
    61usize.into(),
    62usize.into(),
    63usize.into(),
  ];
  let stg8 = Stage::next(stg7, stg8);
  let stg8 = VarArr::let_(dst, format_args!("{}_idct64_stg8", disc), &stg8);
  let stg9 = vec![
    AddClamp(Pos ( 0usize ), Pos ( 15usize )),
    AddClamp(Pos ( 1usize ), Pos ( 14usize )),
    AddClamp(Pos ( 2usize ), Pos ( 13usize )),
    AddClamp(Pos ( 3usize ), Pos ( 12usize )),
    AddClamp(Pos ( 4usize ), Pos ( 11usize )),
    AddClamp(Pos ( 5usize ), Pos ( 10usize )),
    AddClamp(Pos ( 6usize ), Pos ( 9usize )),
    AddClamp(Pos ( 7usize ), Pos ( 8usize )),
    AddClamp(Pos ( 7usize ), Neg ( 8usize )),
    AddClamp(Pos ( 6usize ), Neg ( 9usize )),
    AddClamp(Pos ( 5usize ), Neg ( 10usize )),
    AddClamp(Pos ( 4usize ), Neg ( 11usize )),
    AddClamp(Pos ( 3usize ), Neg ( 12usize )),
    AddClamp(Pos ( 2usize ), Neg ( 13usize )),
    AddClamp(Pos ( 1usize ), Neg ( 14usize )),
    AddClamp(Pos ( 0usize ), Neg ( 15usize )),
    16usize.into(),
    17usize.into(),
    18usize.into(),
    19usize.into(),
    HalfBtf(Neg ( 32usize ), 20usize, Pos ( 32usize ), 27usize, ),
    HalfBtf(Neg ( 32usize ), 21usize, Pos ( 32usize ), 26usize, ),
    HalfBtf(Neg ( 32usize ), 22usize, Pos ( 32usize ), 25usize, ),
    HalfBtf(Neg ( 32usize ), 23usize, Pos ( 32usize ), 24usize, ),
    HalfBtf(Pos ( 32usize ), 23usize, Pos ( 32usize ), 24usize, ),
    HalfBtf(Pos ( 32usize ), 22usize, Pos ( 32usize ), 25usize, ),
    HalfBtf(Pos ( 32usize ), 21usize, Pos ( 32usize ), 26usize, ),
    HalfBtf(Pos ( 32usize ), 20usize, Pos ( 32usize ), 27usize, ),
    28usize.into(),
    29usize.into(),
    30usize.into(),
    31usize.into(),
    AddClamp(Pos ( 32usize ), Pos ( 47usize )),
    AddClamp(Pos ( 33usize ), Pos ( 46usize )),
    AddClamp(Pos ( 34usize ), Pos ( 45usize )),
    AddClamp(Pos ( 35usize ), Pos ( 44usize )),
    AddClamp(Pos ( 36usize ), Pos ( 43usize )),
    AddClamp(Pos ( 37usize ), Pos ( 42usize )),
    AddClamp(Pos ( 38usize ), Pos ( 41usize )),
    AddClamp(Pos ( 39usize ), Pos ( 40usize )),
    AddClamp(Pos ( 39usize ), Neg ( 40usize )),
    AddClamp(Pos ( 38usize ), Neg ( 41usize )),
    AddClamp(Pos ( 37usize ), Neg ( 42usize )),
    AddClamp(Pos ( 36usize ), Neg ( 43usize )),
    AddClamp(Pos ( 35usize ), Neg ( 44usize )),
    AddClamp(Pos ( 34usize ), Neg ( 45usize )),
    AddClamp(Pos ( 33usize ), Neg ( 46usize )),
    AddClamp(Pos ( 32usize ), Neg ( 47usize )),
    AddClamp(Neg ( 48usize ), Pos ( 63usize )),
    AddClamp(Neg ( 49usize ), Pos ( 62usize )),
    AddClamp(Neg ( 50usize ), Pos ( 61usize )),
    AddClamp(Neg ( 51usize ), Pos ( 60usize )),
    AddClamp(Neg ( 52usize ), Pos ( 59usize )),
    AddClamp(Neg ( 53usize ), Pos ( 58usize )),
    AddClamp(Neg ( 54usize ), Pos ( 57usize )),
    AddClamp(Neg ( 55usize ), Pos ( 56usize )),
    AddClamp(Pos ( 55usize ), Pos ( 56usize )),
    AddClamp(Pos ( 54usize ), Pos ( 57usize )),
    AddClamp(Pos ( 53usize ), Pos ( 58usize )),
    AddClamp(Pos ( 52usize ), Pos ( 59usize )),
    AddClamp(Pos ( 51usize ), Pos ( 60usize )),
    AddClamp(Pos ( 50usize ), Pos ( 61usize )),
    AddClamp(Pos ( 49usize ), Pos ( 62usize )),
    AddClamp(Pos ( 48usize ), Pos ( 63usize )),
  ];
  let stg9 = Stage::next(stg8, stg9);
  let stg9 = VarArr::let_(dst, format_args!("{}_idct64_stg9", disc), &stg9);
  let stg10 = vec![
    AddClamp(Pos ( 0usize ), Pos ( 31usize )),
    AddClamp(Pos ( 1usize ), Pos ( 30usize )),
    AddClamp(Pos ( 2usize ), Pos ( 29usize )),
    AddClamp(Pos ( 3usize ), Pos ( 28usize )),
    AddClamp(Pos ( 4usize ), Pos ( 27usize )),
    AddClamp(Pos ( 5usize ), Pos ( 26usize )),
    AddClamp(Pos ( 6usize ), Pos ( 25usize )),
    AddClamp(Pos ( 7usize ), Pos ( 24usize )),
    AddClamp(Pos ( 8usize ), Pos ( 23usize )),
    AddClamp(Pos ( 9usize ), Pos ( 22usize )),
    AddClamp(Pos ( 10usize ), Pos ( 21usize )),
    AddClamp(Pos ( 11usize ), Pos ( 20usize )),
    AddClamp(Pos ( 12usize ), Pos ( 19usize )),
    AddClamp(Pos ( 13usize ), Pos ( 18usize )),
    AddClamp(Pos ( 14usize ), Pos ( 17usize )),
    AddClamp(Pos ( 15usize ), Pos ( 16usize )),
    AddClamp(Pos ( 15usize ), Neg ( 16usize )),
    AddClamp(Pos ( 14usize ), Neg ( 17usize )),
    AddClamp(Pos ( 13usize ), Neg ( 18usize )),
    AddClamp(Pos ( 12usize ), Neg ( 19usize )),
    AddClamp(Pos ( 11usize ), Neg ( 20usize )),
    AddClamp(Pos ( 10usize ), Neg ( 21usize )),
    AddClamp(Pos ( 9usize ), Neg ( 22usize )),
    AddClamp(Pos ( 8usize ), Neg ( 23usize )),
    AddClamp(Pos ( 7usize ), Neg ( 24usize )),
    AddClamp(Pos ( 6usize ), Neg ( 25usize )),
    AddClamp(Pos ( 5usize ), Neg ( 26usize )),
    AddClamp(Pos ( 4usize ), Neg ( 27usize )),
    AddClamp(Pos ( 3usize ), Neg ( 28usize )),
    AddClamp(Pos ( 2usize ), Neg ( 29usize )),
    AddClamp(Pos ( 1usize ), Neg ( 30usize )),
    AddClamp(Pos ( 0usize ), Neg ( 31usize )),
    32usize.into(),
    33usize.into(),
    34usize.into(),
    35usize.into(),
    36usize.into(),
    37usize.into(),
    38usize.into(),
    39usize.into(),
    HalfBtf(Neg ( 32usize ), 40usize, Pos ( 32usize ), 55usize, ),
    HalfBtf(Neg ( 32usize ), 41usize, Pos ( 32usize ), 54usize, ),
    HalfBtf(Neg ( 32usize ), 42usize, Pos ( 32usize ), 53usize, ),
    HalfBtf(Neg ( 32usize ), 43usize, Pos ( 32usize ), 52usize, ),
    HalfBtf(Neg ( 32usize ), 44usize, Pos ( 32usize ), 51usize, ),
    HalfBtf(Neg ( 32usize ), 45usize, Pos ( 32usize ), 50usize, ),
    HalfBtf(Neg ( 32usize ), 46usize, Pos ( 32usize ), 49usize, ),
    HalfBtf(Neg ( 32usize ), 47usize, Pos ( 32usize ), 48usize, ),
    HalfBtf(Pos ( 32usize ), 47usize, Pos ( 32usize ), 48usize, ),
    HalfBtf(Pos ( 32usize ), 46usize, Pos ( 32usize ), 49usize, ),
    HalfBtf(Pos ( 32usize ), 45usize, Pos ( 32usize ), 50usize, ),
    HalfBtf(Pos ( 32usize ), 44usize, Pos ( 32usize ), 51usize, ),
    HalfBtf(Pos ( 32usize ), 43usize, Pos ( 32usize ), 52usize, ),
    HalfBtf(Pos ( 32usize ), 42usize, Pos ( 32usize ), 53usize, ),
    HalfBtf(Pos ( 32usize ), 41usize, Pos ( 32usize ), 54usize, ),
    HalfBtf(Pos ( 32usize ), 40usize, Pos ( 32usize ), 55usize, ),
    56usize.into(),
    57usize.into(),
    58usize.into(),
    59usize.into(),
    60usize.into(),
    61usize.into(),
    62usize.into(),
    63usize.into(),
  ];
  let stg10 = Stage::next(stg9, stg10);
  let stg10 = VarArr::let_(dst, format_args!("{}_idct64_stg10", disc), &stg10);
  let stg11 = vec![
    AddClamp(Pos ( 0usize ), Pos ( 63usize )),
    AddClamp(Pos ( 1usize ), Pos ( 62usize )),
    AddClamp(Pos ( 2usize ), Pos ( 61usize )),
    AddClamp(Pos ( 3usize ), Pos ( 60usize )),
    AddClamp(Pos ( 4usize ), Pos ( 59usize )),
    AddClamp(Pos ( 5usize ), Pos ( 58usize )),
    AddClamp(Pos ( 6usize ), Pos ( 57usize )),
    AddClamp(Pos ( 7usize ), Pos ( 56usize )),
    AddClamp(Pos ( 8usize ), Pos ( 55usize )),
    AddClamp(Pos ( 9usize ), Pos ( 54usize )),
    AddClamp(Pos ( 10usize ), Pos ( 53usize )),
    AddClamp(Pos ( 11usize ), Pos ( 52usize )),
    AddClamp(Pos ( 12usize ), Pos ( 51usize )),
    AddClamp(Pos ( 13usize ), Pos ( 50usize )),
    AddClamp(Pos ( 14usize ), Pos ( 49usize )),
    AddClamp(Pos ( 15usize ), Pos ( 48usize )),
    AddClamp(Pos ( 16usize ), Pos ( 47usize )),
    AddClamp(Pos ( 17usize ), Pos ( 46usize )),
    AddClamp(Pos ( 18usize ), Pos ( 45usize )),
    AddClamp(Pos ( 19usize ), Pos ( 44usize )),
    AddClamp(Pos ( 20usize ), Pos ( 43usize )),
    AddClamp(Pos ( 21usize ), Pos ( 42usize )),
    AddClamp(Pos ( 22usize ), Pos ( 41usize )),
    AddClamp(Pos ( 23usize ), Pos ( 40usize )),
    AddClamp(Pos ( 24usize ), Pos ( 39usize )),
    AddClamp(Pos ( 25usize ), Pos ( 38usize )),
    AddClamp(Pos ( 26usize ), Pos ( 37usize )),
    AddClamp(Pos ( 27usize ), Pos ( 36usize )),
    AddClamp(Pos ( 28usize ), Pos ( 35usize )),
    AddClamp(Pos ( 29usize ), Pos ( 34usize )),
    AddClamp(Pos ( 30usize ), Pos ( 33usize )),
    AddClamp(Pos ( 31usize ), Pos ( 32usize )),
    AddClamp(Pos ( 31usize ), Neg ( 32usize )),
    AddClamp(Pos ( 30usize ), Neg ( 33usize )),
    AddClamp(Pos ( 29usize ), Neg ( 34usize )),
    AddClamp(Pos ( 28usize ), Neg ( 35usize )),
    AddClamp(Pos ( 27usize ), Neg ( 36usize )),
    AddClamp(Pos ( 26usize ), Neg ( 37usize )),
    AddClamp(Pos ( 25usize ), Neg ( 38usize )),
    AddClamp(Pos ( 24usize ), Neg ( 39usize )),
    AddClamp(Pos ( 23usize ), Neg ( 40usize )),
    AddClamp(Pos ( 22usize ), Neg ( 41usize )),
    AddClamp(Pos ( 21usize ), Neg ( 42usize )),
    AddClamp(Pos ( 20usize ), Neg ( 43usize )),
    AddClamp(Pos ( 19usize ), Neg ( 44usize )),
    AddClamp(Pos ( 18usize ), Neg ( 45usize )),
    AddClamp(Pos ( 17usize ), Neg ( 46usize )),
    AddClamp(Pos ( 16usize ), Neg ( 47usize )),
    AddClamp(Pos ( 15usize ), Neg ( 48usize )),
    AddClamp(Pos ( 14usize ), Neg ( 49usize )),
    AddClamp(Pos ( 13usize ), Neg ( 50usize )),
    AddClamp(Pos ( 12usize ), Neg ( 51usize )),
    AddClamp(Pos ( 11usize ), Neg ( 52usize )),
    AddClamp(Pos ( 10usize ), Neg ( 53usize )),
    AddClamp(Pos ( 9usize ), Neg ( 54usize )),
    AddClamp(Pos ( 8usize ), Neg ( 55usize )),
    AddClamp(Pos ( 7usize ), Neg ( 56usize )),
    AddClamp(Pos ( 6usize ), Neg ( 57usize )),
    AddClamp(Pos ( 5usize ), Neg ( 58usize )),
    AddClamp(Pos ( 4usize ), Neg ( 59usize )),
    AddClamp(Pos ( 3usize ), Neg ( 60usize )),
    AddClamp(Pos ( 2usize ), Neg ( 61usize )),
    AddClamp(Pos ( 1usize ), Neg ( 62usize )),
    AddClamp(Pos ( 0usize ), Neg ( 63usize )),
  ];
  let stg11 = Stage::next(stg10, stg11);
  let stg11 = VarArr::let_(dst, format_args!("{}_idct64_stg11", disc), &stg11);
  stg11
}

pub fn inv_tx_add_kernels(file: &mut dyn Write) {
  write_prelude(file);

  let args = vec![
    (quote!(input), quote!(*const i32)),
    (quote!(output), quote!(*mut T)),
    (quote!(output_stride), quote!(i32)),
    (quote!(bd), quote!(u8)),
    (quote!(width), quote!(u16)),
    (quote!(height), quote!(u16)),
  ];
  let mut kernels = KernelDispatch::new(
    "InvTxAddF",
    args,
    None,
    "INV_TX_ADD",
    vec![quote!(4), quote!(4)],
  );

  let mut native = BTreeMap::default();

  let from_crate_root = &["transform", "inverse", ];
  for isa in IsaFeature::sets() {
    let mut isa_module = Module::new_root(from_crate_root,
                                          isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child_file("inv_tx_add",
                                                    px.module_name());
      StdImports.to_tokens(&mut px_module);
      px_module.extend(quote! {
        use rcore::transform::*;
        use rcore::transform::inverse::*;
      });

      let mut ctx = Ctx {
        out: &mut px_module,
        px,
        isa,
        key: (),

        native: &mut native,

        funcs: Functions::default(),
      };
      ctx.inv_tx_adds();
      ctx.push_kernels(&mut kernels);
      ctx.finish();

      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  println!("generated {} inv tx add kernels", kernels.len());

  let tables = kernels.tables();
  writeln!(file, "{}", tables).expect("write inv tx add kernel tables");
}
