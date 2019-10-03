// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::collections::*;
use std::io::Write;

use super::*;

fn butterfly_let<T, U>(
  dst: &mut TokenStream, l: &SimdValue, r: &SimdValue, ln: T, rn: U,
) -> (SimdValue, SimdValue)
where
  T: Display,
  U: Display,
{
  let (l, r) = l.butterfly(r);
  let l = l.let_(dst, ln);
  let r = r.let_(dst, rn);
  (l, r)
}

#[derive(Clone, Default)]
struct Functions {
  sad_internal: BTreeMap<usize, Func>,
  sad: BTreeMap<Block, Func>,
  satd_internal: BTreeMap<usize, Func>,
  satd: BTreeMap<Block, Func>,
}

impl<'a> Ctx<'a, Functions> {
  fn sad(&mut self, width: usize) -> Rc<TokenStream> {
    let calc_prim = PrimType::I32; // TODO use i16 if possible
    let hsum = PrimType::U32;

    assert!(self.simd_width(calc_prim) >= width);
    let simd_width = width;
    if let Some(f) = self.funcs.sad_internal.get(&simd_width) {
      return f.path();
    }

    let px = self.px;
    let args = quote!{(
      left: *const #px,
      left_stride: i32,
      right: *const #px,
      right_stride: i32,
      width: u16,
      height: u16,
    ) -> u32
    };

    let mut func = self.new_func(format_args!("sad_{}nx{}n",
                                              simd_width, simd_width),
                                 args, vec![], true);
    func.inline_hint();

    let sum = Var::new_mut("sum", 0u32);
    sum.to_tokens(&mut *func);

    let left_ptr = Ident::new("left", Span::call_site());
    let left = Plane::new(&left_ptr);
    let right_ptr = Ident::new("right", Span::call_site());
    let right = Plane::new(&right_ptr);

    let mut looop = BlockLoop::std();
    looop.add_loop_var(left);
    looop.add_loop_var(right);

    let load_simd = SimdType::new(px.into(), simd_width);
    let calc_simd = SimdType::new(calc_prim, simd_width);

    looop.gen(&mut *func, 1, simd_width as _, |body, _, _, vars| {
      let col_left = &vars[0];
      let col_right = &vars[1];
      let l = load_simd.uload(col_left)
        .let_(&mut *body, "l");
      let r = load_simd.uload(col_right)
        .let_(&mut *body, "r");
      let l = l.cast(calc_simd).let_(&mut *body, "l");
      let r = r.cast(calc_simd).let_(&mut *body, "r");
      let abs = (&l - &r).abs().let_(&mut *body, "abs");
      sum.add_assign(&mut *body, quote!(#abs.wrapping_sum() as #hsum));
    });

    func.extend(quote! {
      sum
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.sad_internal.insert(simd_width, func).is_none());
    self.funcs.sad_internal.get(&simd_width)
      .unwrap()
      .path()
  }
  fn sad_kernel(&mut self, b: Block) -> Rc<TokenStream> {
    if let Some(f) = self.funcs.sad.get(&b) {
      return f.path();
    }

    let px = self.px;
    let args = quote!{(
      left: *const #px,
      left_stride: i32,
      right: *const #px,
      right_stride: i32,
      width: u16,
      height: u16,
    ) -> u32
    };

    let w = b.w();
    let h = b.h();

    let mut func = self.new_func(format_args!("sad_{}x{}", w, h),
                                 args, vec![], true);
    Self::block_size_check(&mut func, b);

    let size = 8usize.min(w.min(h));

    let internal = self.sad(size);
    func.extend(quote! {
      #internal(left, left_stride, right, right_stride, #w as _, #h as _)
    });

    func.to_tokens(&mut **self.out);
    let path = func.path();
    assert!(self.funcs.sad.insert(b, func).is_none());
    path
  }
  fn sad_kernels(&mut self, kernels: &mut KernelDispatch) {
    for b in Block::blocks_iter().filter(|b| b.w() >= 4 && b.h() >= 4 ) {
      let path = if self.hot_block(b) {
        // this block is hot; allow LLVM to unroll/etc
        self.sad_kernel(b)
      } else {
        // warm, just use a generic impl
        let size = 8usize.min(b.w().min(b.h()));
        self.sad(size)
      };
      let feature_idx = self.isa.index();
      let b_enum = b.table_idx();
      let idx = quote! {
        [#feature_idx][#b_enum]
      };
      kernels.push_kernel(self.px, idx, path);
    }
  }
  fn satd4<T>(&self, dst: &mut TokenStream, sum: &Var<T>, porg: &Plane, pref: &Plane)
    where T: ToTokens,
  {
    const LANES: usize = 4usize;

    const FIRST_HALF: [u32; 8] = [0, 1, 2, 3, 8, 9, 10, 11];
    const SECOND_HALF: [u32; 8] = [4, 5, 6, 7, 12, 13, 14, 15];

    const TRANSPOSE_IDX1: [u32; 8] = [0, 8, 4, 12, 2, 10, 6, 14];
    const TRANSPOSE_IDX2: [u32; 8] = [1, 9, 5, 13, 3, 11, 7, 15];

    let px = self.px;

    let load_ty = SimdType::new(px.into(), LANES);
    let calc_prim = match px {
      PixelType::U8 => PrimType::I16,
      PixelType::U16 => PrimType::I32,
    };
    let calc_ty = SimdType::new(calc_prim, 8);

    let a =
      (0..4i32).map(|i| load_ty.uload(porg.add_r(i))).collect();
    let a = Vector::new(load_ty, a);
    let a = VarArr::let_mut(dst, "a", &a);
    let b =
      (0..4i32).map(|i| load_ty.uload(pref.add_r(i))).collect();
    let b = Vector::new(load_ty, b);
    let b = VarArr::let_mut(dst, "b", &b);

    let a02 = a.get(0).concat(&a.get(2)).let_(dst, "a02");
    let a02 = a02.cast(calc_ty).let_(dst, "a02");
    let a13 = a.get(1).concat(&a.get(3)).let_(dst, "a13");
    let a13 = a13.cast(calc_ty).let_(dst, "a13");

    let b02 = b.get(0).concat(&b.get(2)).let_(dst, "b02");
    let b02 = b02.cast(calc_ty).let_(dst, "b02");
    let b13 = b.get(1).concat(&b.get(3)).let_(dst, "b13");
    let b13 = b13.cast(calc_ty).let_(dst, "b13");

    let ab02 = (a02 - b02).let_(dst, "ab02");
    let ab13 = (a13 - b13).let_(dst, "ab13");

    let (a0a2, a1a3) = butterfly_let(dst, &ab02, &ab13, "a0a2", "a1a3");
    let a0a1 = SimdValue::shuffle2(&a0a2, &a1a3, &FIRST_HALF).let_(dst, "a0a1");
    let a2a3 = SimdValue::shuffle2(&a0a2, &a1a3, &SECOND_HALF).let_(dst, "a2a3");
    let (b0b2, b1b3) = butterfly_let(dst, &a0a1, &a2a3, "b0b2", "b1b3");

    let t0 = SimdValue::shuffle2(&b0b2, &b1b3, &TRANSPOSE_IDX1).let_(dst, "t0");
    let t1 = SimdValue::shuffle2(&b0b2, &b1b3, &TRANSPOSE_IDX2).let_(dst, "t1");

    let (a0a2, a1a3) = butterfly_let(dst, &t0, &t1, "a0a2", "a1a3");
    let a0a1 =
      SimdValue::shuffle2(&a0a2, &a1a3, { &[0u32, 1, 2, 3, 8, 9, 10, 11] })
        .let_(dst, "a0a1");
    let a2a3 =
      SimdValue::shuffle2(&a0a2, &a1a3, { &[4u32, 5, 6, 7, 12, 13, 14, 15] })
        .let_(dst, "a2a3");

    // Use the fact that
    //   (abs(a+b)+abs(a-b))/2 = max(abs(a),abs(b))
    // to merge the final butterfly with the abs and the first stage of
    // accumulation.
    let c0 = a0a1.abs().max(&a2a3.abs()) * calc_ty.splat(2);
    let c0 = c0.cast(SimdType::new(PrimType::U32, 8)).let_(dst, "c0");
    sum.add_assign(dst, quote!(#c0.wrapping_sum()));
  }
  fn satd8<T>(&self, dst: &mut TokenStream, sum: &Var<T>, porg: &Plane, pref: &Plane)
    where T: ToTokens,
  {
    const LANES: usize = 8usize;

    let px = self.px;

    let load_ty = SimdType::new(px.into(), LANES);
    let calc_prim = match px {
      PixelType::U8 => PrimType::I16,
      PixelType::U16 => PrimType::I32,
    };
    let calc_ty = SimdType::new(calc_prim, 8);

    let a = (0..8i32)
      .map(|i| load_ty.uload(porg.add_r(i)).cast(calc_ty))
      .collect();
    let a = Vector::new(calc_ty, a);
    let a = VarArr::let_(dst, "a", &a);
    let b = (0..8i32)
      .map(|i| load_ty.uload(pref.add_r(i)).cast(calc_ty))
      .collect();
    let b = Vector::new(calc_ty, b);
    let b = VarArr::let_(dst, "b", &b);

    let ab = (0..8).map(|i| a.get(i) - b.get(i)).collect();
    let ab = Vector::new(calc_ty, ab);
    let ab = VarArr::let_(dst, "ab", &ab);

    let (a0, a1) = butterfly_let(dst, &ab.get(0), &ab.get(1), "a0", "a1");
    let (a2, a3) = butterfly_let(dst, &ab.get(2), &ab.get(3), "a2", "a3");
    let (a4, a5) = butterfly_let(dst, &ab.get(4), &ab.get(5), "a4", "a5");
    let (a6, a7) = butterfly_let(dst, &ab.get(6), &ab.get(7), "a6", "a7");

    let (b0, b2) = butterfly_let(dst, &a0, &a2, "b0", "b1");
    let (b1, b3) = butterfly_let(dst, &a1, &a3, "b2", "b3");
    let (b4, b6) = butterfly_let(dst, &a4, &a6, "b4", "b5");
    let (b5, b7) = butterfly_let(dst, &a5, &a7, "b6", "b7");

    let (c0, c4) = b0.butterfly(&b4);
    let (c1, c5) = b1.butterfly(&b5);
    let (c2, c6) = b2.butterfly(&b6);
    let (c3, c7) = b3.butterfly(&b7);

    let c0 = c0.let_mut(dst, "c0");
    let c1 = c1.let_mut(dst, "c1");
    let c2 = c2.let_mut(dst, "c2");
    let c3 = c3.let_mut(dst, "c3");
    let c4 = c4.let_mut(dst, "c4");
    let c5 = c5.let_mut(dst, "c5");
    let c6 = c6.let_mut(dst, "c6");
    let c7 = c7.let_mut(dst, "c7");

    let c = vec![&c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7];

    // Transpose
    let mut transpose = TokenStream::default();
    for i in 0..8 {
      for j in 0..i {
        let l = &c[j];
        let r = &c[i];
        transpose.extend(quote! {
          let l = #l.extract_unchecked(#i);
          let r = #r.extract_unchecked(#j);
          #l = #l.replace_unchecked(#i, r);
          #r = #r.replace_unchecked(#j, l);
        });
      }
    }
    // wrap in a block so it can be collapsed by editors.
    dst.extend(quote!({ #transpose }));

    let (a0, a1) = butterfly_let(dst, &c0, &c1, "a0", "a1");
    let (a2, a3) = butterfly_let(dst, &c2, &c3, "a2", "a3");
    let (a4, a5) = butterfly_let(dst, &c4, &c5, "a4", "a5");
    let (a6, a7) = butterfly_let(dst, &c6, &c7, "a6", "a7");

    let (b0, b2) = butterfly_let(dst, &a0, &a2, "b0", "b2");
    let (b1, b3) = butterfly_let(dst, &a1, &a3, "b1", "b3");
    let (b4, b6) = butterfly_let(dst, &a4, &a6, "b4", "b6");
    let (b5, b7) = butterfly_let(dst, &a5, &a7, "b5", "b7");

    // Use the fact that
    //   (abs(a+b)+abs(a-b))/2 = max(abs(a),abs(b))
    // to merge the final butterfly with the abs and the first stage of
    // accumulation.
    //
    // What on Earth does this mean:
    // Avoid pabsw by using max(a, b) + max(a + b + 0x7FFF, 0x7FFF) instead.
    // Actually calculates (abs(a+b)+abs(a-b))/2-0x7FFF.
    // The final sum must be offset to compensate for subtracting 0x7FFF.

    let two = calc_ty.splat(2);
    let c0 = (b0.abs().max(&b4.abs()) * &two).let_(dst, "c0");
    let c1 = (b1.abs().max(&b5.abs()) * &two).let_(dst, "c1");
    let c2 = (b2.abs().max(&b6.abs()) * &two).let_(dst, "c2");
    let c3 = (b3.abs().max(&b7.abs()) * &two).let_(dst, "c3");

    let sum_ty = SimdType::new(PrimType::U32, LANES);
    let d0 = c0.cast(sum_ty).let_(dst, "d0");
    let d1 = c1.cast(sum_ty).let_(dst, "d1");
    let d2 = c2.cast(sum_ty).let_(dst, "d2");
    let d3 = c3.cast(sum_ty).let_(dst, "d3");

    sum.add_assign(dst, quote!(#d0.wrapping_sum()));
    sum.add_assign(dst, quote!(#d1.wrapping_sum()));
    sum.add_assign(dst, quote!(#d2.wrapping_sum()));
    sum.add_assign(dst, quote!(#d3.wrapping_sum()));
  }
  fn satd(&mut self, b: Block) -> Rc<TokenStream> {
    let size = 8.min(b.w().min(b.h()));
    if let Some(f) = self.funcs.satd_internal.get(&size) {
      return f.path();
    }

    let px = self.px;
    let args = quote!{(
      porg: *const #px,
      porg_stride: i32,
      pref: *const #px,
      pref_stride: i32,
      width: u16,
      height: u16,
    ) -> u32
    };

    let mut func = self.new_func(format_args!("satd_{}nx{}n",
                                              size, size),
                                 args, vec![], true);
    func.inline_hint();

    let sum = Var::new_mut("sum", 0u32);
    sum.to_tokens(&mut *func);

    let porg_ptr = Ident::new("porg", Span::call_site());
    let porg = Plane::new(&porg_ptr);
    let pref_ptr = Ident::new("pref", Span::call_site());
    let pref = Plane::new(&pref_ptr);

    let mut looop = BlockLoop::std();
    looop.add_loop_var(porg);
    looop.add_loop_var(pref);

    looop.gen(&mut *func, size as _, size as _, |body, _, _, vars| {
      let porg = &vars[0];
      let pref = &vars[1];
      match size {
        4 => {
          self.satd4(body, &sum, porg, pref);
        },
        8 => {
          self.satd8(body, &sum, porg, pref);
        },
        _ => unreachable!(),
      }
    });

    func.extend(quote! {
      let size = width.min(height).min(8);
      // Normalize the results
      let ln = rcore::util::msb(size as i32) as u64;
      ((sum + (1 << ln >> 1)) >> ln) as u32
    });

    func.to_tokens(&mut **self.out);
    let path = func.path();
    assert!(self.funcs.satd_internal.insert(size, func).is_none());
    path
  }
  fn satd_kernel(&mut self, b: Block) -> Rc<TokenStream> {
    if let Some(f) = self.funcs.satd.get(&b) {
      return f.path();
    }

    let px = self.px;
    let args = quote!{(
      porg: *const #px,
      porg_stride: i32,
      pref: *const #px,
      pref_stride: i32,
      width: u16,
      height: u16,
    ) -> u32
    };

    let w = b.w();
    let h = b.h();

    let mut func = self.new_func(format_args!("satd_{}x{}", w, h),
                                 args, vec![], true);

    let internal = self.satd(b);
    func.extend(quote! {
      #internal(porg, porg_stride, pref, pref_stride, #w as _, #h as _)
    });

    func.to_tokens(&mut **self.out);
    let path = func.path();
    assert!(self.funcs.satd.insert(b, func).is_none());
    path
  }
  fn satd_kernels(&mut self, kernels: &mut KernelDispatch) {
    for b in Block::blocks_iter().filter(|b| b.w() >= 4 && b.h() >= 4 ) {
      let path = if self.hot_block(b) {
        // this block is hot; allow LLVM to unroll/etc
        self.satd_kernel(b)
      } else {
        // warm, just use a generic impl
        self.satd(b)
      };
      let feature_idx = self.isa.index();
      let b_enum = b.table_idx();
      let idx = quote! {
        [#feature_idx][#b_enum]
      };
      kernels.push_kernel(self.px, idx, path);
    }
  }
}

pub fn sad_kernels(file: &mut dyn Write) {
  write_prelude(file);

  let args = vec![
    (quote!(left), quote!(*const T)),
    (quote!(left_stride), quote!(i32)),
    (quote!(right), quote!(*const T)),
    (quote!(right_stride), quote!(i32)),
    (quote!(width), quote!(u16)),
    (quote!(height), quote!(u16)),
  ];
  let ret = quote!(u32);
  let mut kernels = KernelDispatch::new("SadF", args, Some(ret), "SAD", vec![]);

  let from_crate_root = &["dist", "sad"];

  let mut native = BTreeMap::default();

  for isa in IsaFeature::sets() {
    if IsaFeature::Sse2 < isa && isa < IsaFeature::Sse4_1 ||
      IsaFeature::Sse4_1 < isa && isa < IsaFeature::Avx2 {
      // no new instructions for us here.
      continue;
    }

    let mut isa_module = Module::new_root(from_crate_root,
                                          isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child_file("sad", px.module_name());
      StdImports.to_tokens(&mut px_module);
      let mut ctx = Ctx {
        out: &mut px_module,
        isa,
        px,
        key: (),

        native: &mut native,

        funcs: Functions::default(),
      };

      ctx.sad_kernels(&mut kernels);

      ctx.finish();
      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  println!("generated {} sad kernels", kernels.len());

  let tables = kernels.tables();
  writeln!(file, "{}", tables).expect("write sad kernel tables");
}

pub fn satd_kernels(file: &mut dyn Write) {
  write_prelude(file);

  let args = vec![
    (quote!(porg), quote!(*const T)),
    (quote!(porg_stride), quote!(i32)),
    (quote!(pref), quote!(*const T)),
    (quote!(pref_stride), quote!(i32)),
    (quote!(width), quote!(u16)),
    (quote!(height), quote!(u16)),
  ];
  let ret = quote!(u32);
  let mut kernels = KernelDispatch::new("SatdF", args, Some(ret), "SATD",
                                        vec![]);

  let mut native = BTreeMap::default();

  let from_crate_root = &["dist", "satd"];
  for isa in IsaFeature::sets() {
    let mut isa_module = Module::new_root(from_crate_root,
                                          isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child_file("satd", px.module_name());
      StdImports.to_tokens(&mut px_module);
      let mut ctx = Ctx {
        out: &mut px_module,
        isa,
        px,
        key: (),

        native: &mut native,

        funcs: Functions::default(),
      };

      ctx.satd_kernels(&mut kernels);

      ctx.finish();
      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  println!("generated {} satd kernels", kernels.len());

  let tables = kernels.tables();
  writeln!(file, "{}", tables).expect("write satd kernel tables");
}
