// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! We only specialize to the block width.
//! It's the block width which is continuous in memory, and thus is ideal for
//! vectorization.

use std::collections::BTreeMap;
use std::rc::Rc;

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum SmoothDir {
  V,
  H,
  Both,
}

struct Functions {
  dc_128_n: BTreeMap<usize, Func>,
  dc_n: BTreeMap<usize, Func>,
  dc_edge_sum_n: BTreeMap<usize, Func>,
  dc: BTreeMap<(Block, (bool, bool)), TableEntry>,

  paeth_n: BTreeMap<usize, Func>,
  paeth: BTreeMap<Block, Func>,

  cfl_inner_n: BTreeMap<usize, Func>,
  cfl: BTreeMap<(Block, (bool, bool)), Func>,

  smooth_n: BTreeMap<(usize, SmoothDir), Func>,
  smooth: BTreeMap<(Block, SmoothDir), Func>,
}

type Ctx<'a> = super::Ctx<'a, Functions>;
impl<'a> Ctx<'a> {
  fn default_args(&self) -> TokenStream {
    let px = self.px;
    quote! {(
      mut dst: *mut #px,
      dst_stride: i32,
      ac: &[i16],
      alpha: i16,
      above: *const #px,
      left: *const #px,
      top_left: *const #px,
      angle: u16,
      bd: u8,
      width: u16,
      height: u16,
    )}
  }

  fn pred_hot_block(&self, b: Block) -> bool {
    self.isa != IsaFeature::Native
      && b.w() == b.h()
      && b.w() >= 8 && b.w() < 128
  }
  fn pred_table_entry<T>(&self, b: Block, mode: T,
                         variant: (bool, bool),
                         func: Result<Func, (Ident, Rc<TokenStream>)>)
    -> TableEntry
    where T: ToTokens,
  {
    let feature_idx = self.isa.index();
    let b_enum = b.table_idx();

    let variant = match variant {
      (false, false) => quote!(NONE as usize),
      (true, false) => quote!(LEFT as usize),
      (false, true) => quote!(TOP as usize),
      (true, true) => quote!(BOTH as usize),
    };
    let i = quote! {
      [#feature_idx][#b_enum][#mode as usize][#variant]
    };
    let (name, path, func) = match func {
      Ok(f) => (f.name().clone(), f.path(), Some(f)),
      Err((name, path)) => (name, path, None),
    };
    TableEntry {
      indices: Rc::new(i),
      name,
      path,
      func,
    }
  }

  fn pred_dc_128_n(&mut self, width: usize) -> (Ident, Rc<TokenStream>) {

    let px = self.px;
    let px: PrimType = px.into();
    let simd_width = width.min(self.simd_width(px));

    if let Some(f) = self.funcs.dc_128_n.get(&simd_width) {
      return (f.name(), f.path());
    }

    let out_ty = SimdType::new(px, simd_width);

    let args = self.default_args();
    let mut func = self
      .new_func(format_args!("pred_dc_128_{}_n", simd_width),
                args, vec![], true);

    let dst_ptr = Ident::new("dst", Span::call_site());
    let dst = Plane::new(&dst_ptr);

    let v = match self.px {
      PixelType::U8 => {
        let v = 128u8;
        quote!(#v)
      },
      PixelType::U16 => {
        let v = 128u16;
        quote!(#v << (bd - 8))
      },
    };
    let v = out_ty.splat(v)
      .let_(&mut *func, "v");
    let mut bl = BlockLoop::std();
    bl.add_loop_var(dst);
    bl.gen(&mut *func, 1, simd_width as _,
           |body, _, _, vars| {
             let dst = &vars[0];
             v.ustore(body, dst);
           });

    func.to_tokens(&mut **self.out);
    let ret = (func.name(), func.path());
    assert!({
      self.funcs.dc_128_n
        .insert(simd_width, func)
        .is_none()
    });
    ret
  }
  fn pred_dc_128_block(&mut self, b: Block) {
    let key = (b, (false, false));
    if let Some(_) = self.funcs.dc.get(&key) {
      return;
    }

    let w = b.w();
    let (wfn_name, wfn_path) = self.pred_dc_128_n(w);
    let mode = quote!(DC_PRED as usize);
    let variant = (false, false);

    let entry = if self.pred_hot_block(b) {
      let h = b.h();

      let args = self.default_args();
      let mut func = self
        .new_func(format_args!("pred_dc_128_{}_{}", w, h),
                  args, vec![], true);

      Self::block_size_check(&mut *func, b);

      func.extend(quote! {
        #wfn_name(dst, dst_stride, ac, alpha, above, left, top_left,
                  angle, bd, #w as _, #h as _);
      });

      func.to_tokens(&mut **self.out);
      self.pred_table_entry(b, mode, variant,
                            Ok(func))
    } else {
      let func = Err((wfn_name, wfn_path));
      self.pred_table_entry(b, mode,
                            variant,
                            func)
    };
    assert!({
      self.funcs.dc
        .insert(key, entry)
        .is_none()
    });
  }
  fn pred_dc_128_blocks(&mut self) {
    for b in Block::tx_sizes_iter() {
      self.pred_dc_128_block(b);
    }
  }

  fn pred_dc_n(&mut self, width: usize) -> (Ident, Rc<TokenStream>) {
    let px = self.px;
    let simd_width = width.min(self.simd_width(px.into()));

    if let Some(f) = self.funcs.dc_n.get(&simd_width) {
      return (f.name(), f.path());
    }

    let out_ty = SimdType::new(px.into(), simd_width);

    let args = quote! {(
      dst: *mut #px,
      dst_stride: i32,
      avg: #px,
      width: u16,
      height: u16,
    )};
    let mut func = self
      .new_func(format_args!("pred_dc_{}_n", simd_width),
                     args, vec![], true);

    let dst_ptr = Ident::new("dst", Span::call_site());
    let dst = Plane::new(&dst_ptr);

    let v = out_ty.splat(quote!(avg))
      .let_(&mut *func, "avg");
    let mut bl = BlockLoop::std();
    bl.add_loop_var(dst);
    bl.gen(&mut *func, 1, simd_width as _,
           |body, _, _, vars| {
             let dst = &vars[0];
             v.ustore(body, dst);
           });

    func.to_tokens(&mut **self.out);
    let ret = (func.name(), func.path());
    assert!({
      self.funcs.dc_n
        .insert(simd_width, func)
        .is_none()
    });
    ret
  }

  fn pred_dc_avg_ty(&self) -> PrimType {
    match self.px {
      PixelType::U8 => PrimType::U16,
      PixelType::U16 => PrimType::U32,
    }
  }
  fn pred_dc_edge_sum_n(&mut self, width: usize) -> Ident {
    let calc = self.pred_dc_avg_ty();
    // max size in packed_simd for calc, not px
    let simd_width = width.min(self.simd_width(calc));

    if let Some(f) = self.funcs.dc_edge_sum_n.get(&simd_width) {
      return f.name();
    }

    let px = self.px;
    let px_ty = SimdType::new(px.into(), simd_width);
    let calc_ty = SimdType::new(calc, simd_width);

    let args = quote! {(
      sum: &mut #calc,
      src: *const #px,
      size: u16,
    )};
    let mut func = self
      .new_func(format_args!("pred_dc_edge_sum_{}_n", width),
                args, vec![], true);
    func.inline_hint();
    func.extend(quote! {
      debug_assert!(size as usize >= #simd_width);
    });

    let src_ptr = Ident::new("src", Span::call_site());
    let src = Plane::new_stride(&src_ptr, 1i32);

    let mut l = Loop::new("col", quote!(size),
                          PrimType::U16);
    l.add_loop_var(src);
    l.gen(&mut *func, simd_width as _,
          |body, _, vars| {
            let src = &vars[0];
            let t = px_ty.uload(src)
              .cast(calc_ty)
              .let_(body, "t");
            body.extend(quote! {
              *sum += #t.wrapping_sum();
            });
          });

    func.to_tokens(&mut **self.out);
    let name = func.name();
    assert!({
      self.funcs.dc_edge_sum_n
        .insert(simd_width, func)
        .is_none()
    });
    name
  }
  fn pred_dc(&mut self, b: Block, (left, top): (bool, bool)) {
    let w = b.w();
    let h = b.h();
    let px = self.px;

    let key = (b, (left, top));
    if let Some(_) = self.funcs.dc.get(&key) {
      return;
    }

    let args = quote! {(
      mut dst: *mut #px,
      dst_stride: i32,
      ac: &[i16],
      alpha: i16,
      above: *const #px,
      left: *const #px,
      top_left: *const #px,
      angle: u16,
      bd: u8,
      width: u16,
      height: u16,
    )};
    let params = vec![];

    let wfn_name = self.pred_dc_n(w).0;
    let above_edge_sum = self.pred_dc_edge_sum_n(w);
    let left_edge_sum = self.pred_dc_edge_sum_n(h);
    let func = match (left, top) {
      (true, true) => format!("pred_dc_{}_{}", w, h),
      (true, false) => format!("pred_dc_left_{}_{}", w, h),
      (false, true) => format!("pred_dc_top_{}_{}", w, h),
      (false, false) => unreachable!(),
    };
    let mut func = self.new_func(func, args, params,
                                 true);

    let sum_ty = self.pred_dc_avg_ty();
    let mut len = 0;
    if left {
      len += h;
    }
    if top {
      len += w;
    }
    let len_add = len >> 1;

    func.extend(quote! {
      let mut sum = 0;
      if #top {
        #above_edge_sum(&mut sum, above, width);
      }
      if #left {
        #left_edge_sum(&mut sum, left, height);
      }
      sum += #len_add as #sum_ty;
      sum /= #len as #sum_ty;
      let avg = sum as #px;
      #wfn_name(dst, dst_stride, avg, width, height);
    });

    func.to_tokens(&mut **self.out);
    let k = (b, (left, top));
    let mode = quote!(DC_PRED);
    let entry = self.pred_table_entry(b, mode,
                                      (left, top),
                                      Ok(func));
    assert!({
      self.funcs.dc
        .insert(k, entry)
        .is_none()
    });
  }
  fn pred_dc_blocks(&mut self) {
    for b in Block::tx_sizes_iter() {
      for &variant in [(true, true), (true, false), (false, true)].iter() {
        self.pred_dc(b, variant);
      }
    }
  }

  fn pred_paeth_n(&mut self, width: usize) -> Ident {
    let calc = self.pred_dc_avg_ty();
    let simd_width = width.min(self.simd_width(calc));
    if let Some(f) = self.funcs.paeth_n.get(&simd_width) {
      return f.name();
    }

    let px = self.px;
    let calc_ty = SimdType::new(calc, simd_width);
    let px_ty = SimdType::new(px.into(), simd_width);

    let args = quote! {(
      dst: *mut #px,
      dst_stride: i32,
      above: *const #px,
      left: *const #px,
      top_left: *const #px,
      width: u16,
      height: u16,
    )};
    let mut func = self.out
      .new_priv_func(format_args!("pred_paeth_{}_n", simd_width),
                     args, px, self.isa);
    func.extend(quote! {
      debug_assert!(width as usize >= #simd_width);
    });

    let top_left = px_ty.splat(quote!(*top_left))
      .let_(&mut *func, "top_left");
    let calc_top_left = top_left
      .cast(calc_ty)
      .let_(&mut *func, "calc_top_left");

    let dst = Plane::new_named("dst", quote!(dst_stride));
    let left = Plane::new_named("left", quote!(1i32));
    let above = Plane::new_named("above", quote!((width as i32)));

    let mut l = BlockLoop::std();
    l.add_loop_var(dst);
    l.add_loop_var(above);

    l.gen(&mut *func, 1, simd_width as _, |func, r, _, vars| {
      let dst = &vars[0];
      let above = &vars[1];

      let left = left.add_r(quote!((height as i32) - 1 - #r));
      let left = px_ty.splat(quote!(*#left))
        .let_(&mut *func, "left");
      let top = px_ty.uload(above)
        .let_(&mut *func, "top");

      let calc_top = top
        .cast(calc_ty)
        .let_(&mut *func, "calc_top");
      let calc_left = left
        .cast(calc_ty)
        .let_(&mut *func, "calc_left");

      let p_base = (&calc_top + &calc_left - &calc_top_left)
        .let_(&mut *func, "p_base");
      let p_left = (&p_base - &calc_left).abs()
        .let_(&mut *func, "p_left");
      let p_top  = (&p_base - &calc_top).abs()
        .let_(&mut *func, "p_top");
      let p_top_left = (&p_base - &calc_top_left).abs()
        .let_(&mut *func, "p_top_left");

      let out2 = quote! {
        #p_top.le(#p_top_left)
          .select(#top, #top_left)
      };
      let out1 = quote! {
        (#p_left.le(#p_top) & #p_left.le(#p_top_left))
          .select(#left, #out2)
      };

      let out = SimdValue::from(px_ty, out1)
        .let_(&mut *func, "out");
      out.ustore(&mut *func, dst);
    });

    func.to_tokens(&mut **self.out);
    let name = func.name();
    assert!(self.funcs.paeth_n.insert(simd_width, func).is_none());
    name
  }
  fn pred_paeth_blocks(&mut self) {
    for b in Block::tx_sizes_iter().filter(|b| b.w() > 4 && b.h() > 4 ) {
      let w = b.w();
      let h = b.h();
      let px = self.px;

      let wfn_name = self.pred_paeth_n(w);
      let args = self.default_args();
      let mut func = self.out
        .new_func(format_args!("pred_paeth_{}_{}", w, h),
                  args, px, self.isa);
      Self::block_size_check(&mut *func, b);

      func.extend(quote! {
        #wfn_name(dst, dst_stride, above, left, top_left, #w as _, #h as _);
      });

      func.to_tokens(&mut **self.out);
      assert!(self.funcs.paeth.insert(b, func).is_none());
    }
  }

  fn pred_cfl_calc_ty(&self) -> PrimType {
    match self.px {
      PixelType::U8 => PrimType::I16,
      PixelType::U16 => PrimType::I32,
    }
  }
  fn pred_cfl_inner_n(&mut self, width: usize) -> Ident {
    let calc = self.pred_cfl_calc_ty();
    let simd_width = width.min(self.simd_width(calc));
    if let Some(f) = self.funcs.cfl_inner_n.get(&simd_width) {
      return f.name();
    }

    let px = self.px;
    let calc_ty = SimdType::new(calc, simd_width);
    let px_ty = SimdType::new(px.into(), simd_width);

    let args = quote! {(
      dst: *mut #px,
      dst_stride: i32,
      ac: *const i16,
      alpha: i16,
      avg: #calc,
      bd: u8,
      width: u16,
      height: u16,
    )};
    let mut func = self.out
      .new_priv_func(format_args!("pred_cfl_inner_{}_n", width),
                     args, px, self.isa);

    let sample_max = match self.px {
      PixelType::U8 => None,
      PixelType::U16 => {
        let sm = calc_ty.splat(quote!((1 << bd) - 1))
          .let_(&mut *func, "sample_max");
        Some(sm)
      },
    };
    let alpha = calc_ty.splat(quote!(alpha))
      .let_(&mut *func, "alpha");
    let avg = calc_ty.splat(quote!(avg))
      .let_(&mut *func, "avg");
    let thirty_two = calc_ty.splat(quote!(32));
    let zero = calc_ty.splat(quote!(0));

    let dst = Ident::new("dst", Span::call_site());
    let dst = Plane::new(&dst);
    let ac = Plane::new_named("ac", quote!((width as i32)));

    let mut l = BlockLoop::std();
    l.add_loop_var(dst);
    l.add_loop_var(ac);

    l.gen(&mut *func, 1, simd_width as _, |func, _, _, vars| {
      let dst = &vars[0];
      let ac = &vars[1];

      let luma = px_ty.aload(ac)
        .cast(calc_ty)
        .let_(&mut *func, "luma");
      let q6 = (&alpha * &luma)
        .let_(&mut *func, "scaled_luma_q6");
      let abs_q6 = (&q6.abs() + &thirty_two).shr(6u32)
        .let_(&mut *func, "abs_scaled_luma_q6");

      let q0 = SimdValue::from(calc_ty, quote! {
        #q6.lt(#zero).select(-#abs_q6, #abs_q6)
      })
        .let_(&mut *func, "q0");
      let q0 = (&avg + &q0).let_(&mut *func, "q0");
      let q0 = q0.max(&zero);
      let q0 = if let Some(ref sample_max) = sample_max {
        q0.min(sample_max)
      } else {
        q0
      };
      let q0 = q0.cast(px_ty)
        .let_(&mut *func, "q0");
      q0.ustore(&mut *func, dst);
    });

    func.to_tokens(&mut **self.out);
    let name = func.name();
    assert!(self.funcs.cfl_inner_n.insert(simd_width, func).is_none());
    name
  }
  fn pred_cfl_blocks(&mut self) {
    let blocks = Block::tx_sizes_iter()
      .filter(|b| b.w() >= 32 );

    let p = [(false, false), (true, false), (false, true), (true, true)];

    for b in blocks {
      for &(left, top) in p.iter() {
        let w = b.w();
        let h = b.h();
        let px = self.px;
        let calc = self.pred_cfl_calc_ty();

        let dc_f = &self.funcs.dc.get(&(b, (left, top)))
          .expect("no pred_dc fn for block width")
          .name();
        let cfl_inner_f = self.pred_cfl_inner_n(w);

        let args = self.default_args();
        let mut func = match (left, top) {
          (true, true) => self.out
            .new_func(format_args!("pred_cfl_{}_{}", w, h),
                      args, px, self.isa),
          (true, false) => self.out
            .new_func(format_args!("pred_cfl_left_{}_{}", w, h),
                      args, px, self.isa),
          (false, true) => self.out
            .new_func(format_args!("pred_cfl_top_{}_{}", w, h),
                      args, px, self.isa),
          (false, false) => self.out
            .new_func(format_args!("pred_cfl_128_{}_{}", w, h),
                      args, px, self.isa),
        };

        func.extend(quote! {
          #dc_f(dst, dst_stride, ac, alpha, above, left, top_left, angle, bd,
                width, height);

          if alpha == 0 { return; }

          let avg = *dst as #calc;
          #cfl_inner_f(dst, dst_stride, ac.as_ptr(), alpha, avg, bd, width, height);
        });

        func.to_tokens(&mut **self.out);
        let k = (b, (left, top));
        assert!(self.funcs.cfl.insert(k, func).is_none());
      }
    }
  }

  fn add_match_body(&self, match_body: &mut TokenStream) {
    let block_tx_size = |b: &Block| {
      let s = format!("TX_{}X{}", b.0, b.1);
      Ident::new(&s, Span::call_site())
    };

    for ((b, (left, top)), entry) in self.funcs.dc.iter() {
      let tx_size = block_tx_size(b);
      let mode = quote!(DC_PRED);
      let variant = match (left, top) {
        (false, false) => quote!(NONE),
        (true, false) => quote!(LEFT),
        (false, true) => quote!(TOP),
        (true, true) => quote!(BOTH),
      };
      let name = &entry.name;
      match_body.extend(quote! {
        (#tx_size, #mode, #variant) => #name,
      });
    }
    for (b, f) in self.funcs.paeth.iter() {
      let tx_size = block_tx_size(b);
      let mode = quote!(PAETH_PRED);
      let variant = quote!(_);
      let name = &f.name;
      match_body.extend(quote! {
        (#tx_size, #mode, #variant) => #name,
      });
    }
    for ((b, (left, top)), f) in self.funcs.cfl.iter() {
      let tx_size = block_tx_size(b);
      let mode = quote!(UV_CFL_PRED);
      let variant = match (left, top) {
        (false, false) => quote!(NONE),
        (true, false) => quote!(LEFT),
        (false, true) => quote!(TOP),
        (true, true) => quote!(BOTH),
      };
      let name = &f.name;
      match_body.extend(quote! {
        (#tx_size, #mode, #variant) => #name,
      });
    }
  }

  fn add_kernels(&self, set: &mut KernelDispatch) {
    let all_variants = &[
      (false, false), (true, false),
      (false, true), (true, true),
    ];

    for (_, f) in self.funcs.dc.iter() {
      set.push_kernel(self.px, f.idx(), f.path());
    }

    let mut add = |b: &Block, path: Rc<TokenStream>, mode, variants: &[(bool, bool)],
                   enable: bool| {
      let feature_idx = self.isa.index();
      let b_enum = b.table_idx();

      for &(left, top) in variants.iter() {
        let variant = match (left, top) {
          (false, false) => quote!(NONE as usize),
          (true, false) => quote!(LEFT as usize),
          (false, true) => quote!(TOP as usize),
          (true, true) => quote!(BOTH as usize),
        };
        let i = quote! {
          [#feature_idx][#b_enum][#mode][#variant]
        };
        if enable {
          set.push_kernel(self.px, i, path.clone());
        }
      }
    };
    for (b, f) in self.funcs.paeth.iter() {
      let mode = quote!(PAETH_PRED as usize);
      let path = f.path();
      // FIXME(diamond): corruption, but it's not obvious why.
      add(b, path, mode, all_variants, false);
    }
    for ((b, (left, top)), f) in self.funcs.cfl.iter() {
      let mode = quote!(UV_CFL_PRED as usize);
      let path = f.path();
      add(b, path, mode, &[(*left, *top)], true);
    }
  }
}
pub fn predict_kernels(file: &mut dyn Write) {
  write_prelude(file);
  writeln!(file, "use rcore::predict::PredictionMode::*;
use rcore::predict::PredictionVariant::*;").unwrap();

  let args = vec![
    (quote!(dst), quote!(*mut T)),
    (quote!(dst_stride), quote!(i32)),
    (quote!(ac), quote!(&[i16])),
    (quote!(alpha), quote!(i16)),
    (quote!(above), quote!(*const T)),
    (quote!(left), quote!(*const T)),
    (quote!(top_left), quote!(*const T)),
    (quote!(angle), quote!(u16)),
    (quote!(bd), quote!(u8)),
    (quote!(width), quote!(u16)),
    (quote!(height), quote!(u16)),
  ];
  let ret = None;
  let table_sizes = vec![quote!(4), quote!(UV_CFL_PRED as usize + 1), ];
  let mut kernels = KernelDispatch::new("PredictF", args,
                                        ret, "PREDICT",
                                        table_sizes);

  let from_crate_root = &["predict", ];
  for isa in IsaFeature::sets() {
    let mut isa_module = Module::new_root(from_crate_root,
                                          isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child(px.module_name());
      px_module.extend(quote! {
        use rcore::predict::*;
        use rcore::transform::TxSize::*;
        use rcore::predict::PredictionMode::*;
        use rcore::predict::PredictionVariant::*;
      });
      StdImports.to_tokens(&mut px_module);

      {
        let mut ctx = Ctx {
          out: &mut px_module,
          isa,
          px,
          key: (),

          native: &mut BTreeMap::new(),

          funcs: Functions {
            dc_128_n: Default::default(),

            dc_edge_sum_n: Default::default(),
            dc_n: Default::default(),
            dc: Default::default(),

            paeth_n: Default::default(),
            paeth: Default::default(),

            cfl_inner_n: Default::default(),
            cfl: Default::default(),

            smooth_n: Default::default(),
            smooth: Default::default(),
          },
        };
        ctx.pred_dc_128_blocks();
        ctx.pred_dc_blocks();
        ctx.pred_paeth_blocks();
        ctx.pred_cfl_blocks();

        ctx.add_kernels(&mut kernels);
      }

      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  println!("generated {} predict kernels", kernels.len());
  let tables = kernels.tables();
  writeln!(file, "{}", tables).expect("write predict kernel tables");
}
