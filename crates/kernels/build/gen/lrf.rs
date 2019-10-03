// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::collections::*;

use super::*;

#[derive(Default)]
struct Functions {
  /// fn sgrproj_sum_finish(
  ///  ssq: u32, sum: u32, n: u32, one_over_n: u32, s: u32, bdm8: usize,
  ///) -> (u32, u32);
  sum_finish: BTreeMap<usize, Func>,
  ///fn get_integral_square(
  ///  iimg: &[u32], stride: usize, x: usize, /*y: usize,*/ size: usize,
  ///) -> u32;
  /// y is zero.
  integral_sq: BTreeMap<usize, Func>,

  box_ab_internal: BTreeMap<usize, Func>,
  box_ab_rn_internal: BTreeMap<(usize, usize), Func>,
  box_ab_rn: BTreeMap<(usize, usize), Func>,

  box_f_rn_internal: BTreeMap<(usize, usize), Func>,
  box_f_rn: BTreeMap<(usize, usize), Func>,

  solve_filter: BTreeMap<usize, Func>,
  stripe_filter: BTreeMap<usize, Func>,
  sgrproj_internal: BTreeMap<usize, Func>,
}

type Ctx<'a> = super::Ctx<'a, Functions>;
impl<'a> Ctx<'a> {
  fn sum_finish(&mut self, width: usize) -> Ident {
    if let Some(f) = self.funcs.sum_finish.get(&width) {
      return f.name();
    }

    let simd_width = width;
    let (calc, scalc) = (PrimType::U32, PrimType::I32);
    let calc_ty = SimdType::new(calc, simd_width);
    let scalc_ty = SimdType::new(scalc, simd_width);

    let args = quote! {(
      ssq: #calc_ty,
      sum: #calc_ty,
      n: #calc,
      one_over_n: #calc,
      s: #calc,
      bdm8: u8,
    )
    -> (#calc_ty, #calc_ty)
    };
    let mut func = self.out
      .new_priv_func(format_args!("sum_finish_{}", width),
                     args, self.px, self.isa);
    func.inline_hint();

    let bdm8 = match self.px {
      PixelType::U8 => quote!(0 as #calc),
      _ => quote!(bdm8 as #calc),
    };
    let ssq = SimdValue::from(calc_ty, quote!(ssq));
    let sum = SimdValue::from(calc_ty, quote!(sum));
    let one_over_n = calc_ty.splat(quote!(one_over_n));
    let s = calc_ty.splat(quote!(s));
    let scaled_ssq = ssq.round_shift(quote!(2 * #bdm8))
      .let_(&mut *func, "scaled_ssq");
    let scaled_sum = sum.round_shift(&bdm8)
      .let_(&mut *func, "scaled_sum");
    let n = calc_ty.splat(quote!(n));
    let p_l = (&scaled_ssq * &n).cast(scalc_ty)
      .let_(&mut *func, "p_l");
    let p_r = (&scaled_sum * &scaled_sum).cast(scalc_ty)
      .let_(&mut *func, "p_r");
    let p = (&p_l - &p_r)
      .max(&scalc_ty.splat(quote!(0)))
      .cast(calc_ty)
      .let_(&mut *func, "p");
    let z = (&p * &s).round_shift(quote!(SGRPROJ_MTABLE_BITS as #calc))
      .let_(&mut *func, "z");

    let v255 = calc_ty.splat(quote!(255));
    let v256 = calc_ty.splat(quote!(256));
    let v0 = calc_ty.splat(quote!(0));
    let v1 = calc_ty.splat(quote!(1));
    let e = SimdValue::from(calc_ty, quote! {
      (((#z << SGRPROJ_SGR_BITS as #calc) + (#z >> 1)) / (#z + 1))
    });
    let e = quote! {
      #z.eq(#v0)
        .select(#v1, #e)
    };
    let a = quote! {
      #z.ge(#v255)
        .select(#v256, #e)
    };
    let a = SimdValue::from(calc_ty, a)
      .let_(&mut *func, "a");
    let b = (calc_ty.splat(quote!((1 << SGRPROJ_SGR_BITS))) - &a) * &sum * &one_over_n;
    let b = b.let_(&mut *func, "b");
    let b = b.round_shift(quote!(SGRPROJ_RECIP_BITS as #calc))
      .let_(&mut *func, "b");
    func.extend(quote! {
      (#a, #b)
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.sum_finish.insert(width, func).is_none());
    self.funcs.sum_finish.get(&width)
      .unwrap()
      .name()
  }
  /// Using an integral image, compute the sum of a square region
  fn integral_square(&mut self, width: usize) -> Ident {
    if let Some(f) = self.funcs.integral_sq.get(&width) {
      return f.name();
    }

    let simd_width = width;
    let calc = PrimType::U32;
    let calc_ty = SimdType::new(calc, simd_width);

    let args = quote! {(
      iimg: *const #calc,
      stride: u32,
      //x: u16, == 0
      // y: u16, == 0
      size: u16,
    )
    -> #calc_ty
    };
    let mut func = self.out
      .new_priv_func(format_args!("get_integral_sq_{}", width),
                     args, self.px, self.isa);
    func.inline_hint();

    let t0 = calc_ty.uload(quote!(iimg));
    let t1 = calc_ty.uload(quote!(iimg.add(size as usize)));
    let t2 = calc_ty.uload(quote!(iimg.add((size * stride) as usize)));
    let t3 = calc_ty.uload(quote!(iimg.add((size * stride + size) as usize)));

    // Cancel out overflow in iimg by using wrapping arithmetic

    // simd ops are always wrapping..
    let out = &t0 + &t3 - &t2 - &t1;

    func.extend(quote! {
      let size = size as u32;
      #out
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.integral_sq.insert(width, func).is_none());
    self.funcs.integral_sq.get(&width)
      .unwrap()
      .name()
  }
  fn box_ab_internal(&mut self, width: usize) -> Ident {
    let calc = PrimType::U32;
    let simd_width = width.min(self.simd_width(calc));
    if let Some(f) = self.funcs.box_ab_internal.get(&simd_width) {
      return f.name();
    }

    let calc_ty = SimdType::new(calc, simd_width);

    let args = quote! {(
      r: u32,
      mut af: *mut #calc,
      mut bf: *mut #calc,
      mut iimg: *const #calc,
      mut iimg_sq: *const #calc,
      iimg_stride: u32,
      s: u32,
      bdm8: u8,
      range: Range<u16>,
    )};
    let mut func = self.out
      .new_priv_func(format_args!("box_ab_internal_{}", simd_width),
                     args, self.px, self.isa);

    func.extend(quote! {
      let d = (r * 2 + 1) as u16;
      let n = (d * d) as u32;
      let one_over_n = if r == 1 { 455 } else { 164 };
    });

    let gis = self.integral_square(simd_width);
    let sum_finish = self.sum_finish(simd_width);

    let mut looop = TokenStream::default();
    looop.extend(quote! {
      let sum = #gis(iimg, iimg_stride, d);
      let ssq = #gis(iimg_sq, iimg_stride, d);
      let (a, b) = #sum_finish(ssq, sum, n, one_over_n, s, bdm8);
    });
    let a = SimdValue::from(calc_ty, quote!(a));
    a.astore(&mut looop, quote!(af));
    let b = SimdValue::from(calc_ty, quote!(b));
    b.astore(&mut looop, quote!(bf));

    let gis_v2 = self.integral_square(2);
    let sum_finish_v2 = self.sum_finish(2);

    func.extend(quote! {
      let mut af = af.add(range.start as usize);
      let mut af_end = af.add(range.end as usize);
      let mut bf = bf.add(range.start as usize);
      let mut iimg = iimg.add(range.start as usize);
      let mut iimg_sq = iimg_sq.add(range.start as usize);

      while af < af_end {
        #looop

        af = af.add(#simd_width);
        bf = bf.add(#simd_width);
        iimg = iimg.add(#simd_width);
        iimg_sq = iimg_sq.add(#simd_width);
      }
    });

    // now finish the last two columns
    func.extend(quote! {
      let sum = #gis_v2(iimg, iimg_stride, d);
      let ssq = #gis_v2(iimg_sq, iimg_stride, d);
      let (a, b) = #sum_finish_v2(ssq, sum, n, one_over_n, s, bdm8);
    });
    let calc_ty = SimdType::new(calc, 2);
    let a = SimdValue::from(calc_ty, quote!(a));
    a.astore(&mut *func, quote!(af));
    let b = SimdValue::from(calc_ty, quote!(b));
    b.astore(&mut *func, quote!(bf));

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.box_ab_internal.insert(simd_width, func).is_none());
    self.funcs.box_ab_internal.get(&simd_width)
      .unwrap()
      .name()
  }
  fn box_ab_rn_internal(&mut self, r: usize, width: usize) -> Ident {
    let calc = PrimType::U32;
    let simd_width = width.min(self.simd_width(calc));
    if let Some(f) = self.funcs.box_ab_rn_internal.get(&(r, simd_width)) {
      return f.name();
    }

    let args = quote! {(
      af: *mut #calc,
      bf: *mut #calc,
      iimg: *const #calc,
      iimg_sq: *const #calc,
      iimg_stride: u32,
      s: u32,
      bdm8: u8,
      range: Range<u16>,
    )};
    let mut func = self.out
      .new_func(format_args!("box_ab_r{}_internal_{}", r, simd_width),
                args, self.px, self.isa);

    let internal = self.box_ab_internal(simd_width);

    let r_u32 = r as u32;
    func.extend(quote! {
      #internal(#r_u32, af, bf, iimg, iimg_sq, iimg_stride, s, bdm8, range);
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.box_ab_rn_internal.insert((r, simd_width), func).is_none());
    self.funcs.box_ab_rn_internal.get(&(r, simd_width))
      .unwrap()
      .name()
  }
  fn box_ab_rn(&mut self, r: usize, width: usize) -> Ident {
    if let Some(f) = self.funcs.box_ab_rn.get(&(r, width)) {
      return f.name();
    }

    let calc = PrimType::U32;
    let simd_width = width.min(self.simd_width(calc));

    let args = quote! {(
      af: *mut #calc,
      bf: *mut #calc,
      iimg: *const #calc,
      iimg_sq: *const #calc,
      iimg_stride: u32,
      s: u32,
      bdm8: u8,
    )};
    let mut func = self.out
      .new_func(format_args!("box_ab_r{}_{}", r, simd_width),
                args, self.px, self.isa);

    let internal = self.box_ab_rn_internal(r, simd_width);

    let width_u16 = width as u16;
    func.extend(quote! {
      #internal(af, bf, iimg, iimg_sq, iimg_stride, s, bdm8,
                0..#width_u16);
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.box_ab_rn.insert((r, width), func).is_none());
    self.funcs.box_ab_rn.get(&(r, width))
      .unwrap()
      .name()
  }
  fn box_f_r0_internal(&mut self, simd_width: usize) -> Ident {
    let key = (0, simd_width);
    if let Some(f) = self.funcs.box_f_rn_internal.get(&key) {
      return f.name();
    }

    let calc = PrimType::U32;
    let px = self.px;
    let calc_ty = SimdType::new(calc, simd_width);
    let px_ty = SimdType::new(px.into(), simd_width);

    let args = quote! {(
      mut f: *mut #calc,
      mut cdeffed: *const #px,
      // don't need cdeffed's stride
      mut range: Range<u16>,
    )};
    let mut func = self.out
      .new_priv_func(format_args!("box_f_r0_internal_{}", simd_width),
                     args, self.px, self.isa);

    func.extend(quote! {
      let shift = (5 + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS) as #calc;
    });

    let mut looop = TokenStream::default();
    px_ty.uload(quote!(cdeffed.add(x as usize)))
      .cast(calc_ty)
      .shl(quote!((SGRPROJ_RST_BITS as u32)))
      .astore(&mut looop, quote!(f.add(x as usize)));

    func.extend(quote! {
      while range.start < range.end {
        let x = range.start;
        #looop

        range.start += #simd_width as u16;
      }
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.box_f_rn_internal.insert(key, func).is_none());
    self.funcs.box_f_rn_internal.get(&key)
      .unwrap()
      .name()
  }
  fn box_f_r1_internal(&mut self, simd_width: usize) -> Ident {
    let key = (1, simd_width);
    if let Some(f) = self.funcs.box_f_rn_internal.get(&key) {
      return f.name();
    }

    let calc = PrimType::U32;
    let px = self.px;
    let calc_ty = SimdType::new(calc, simd_width);
    let px_ty = SimdType::new(px.into(), simd_width);

    let args = quote! {(
      af: [&[#calc; IMAGE_WIDTH_MAX + 2]; 3],
      bf: [&[#calc; IMAGE_WIDTH_MAX + 2]; 3],
      mut f: *mut #calc,
      mut cdeffed: *const #px,
      // don't need cdeffed's stride
      mut range: Range<u16>,
    )};
    let mut func = self.out
      .new_priv_func(format_args!("box_f_r1_internal_{}", simd_width),
                     args, self.px, self.isa);

    func.extend(quote! {
      let shift = (5 + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS) as #calc;
      let af = [af[0].as_ptr(), af[1].as_ptr(), af[2].as_ptr()];
      let bf = [bf[0].as_ptr(), bf[1].as_ptr(), bf[2].as_ptr()];
      let mut f = f as *mut #calc_ty;
    });
    let three = calc_ty.splat(quote!(3))
      .let_(&mut *func, "three");
    let four  = calc_ty.splat(quote!(4))
      .let_(&mut *func, "four");

    let mut looop = TokenStream::default();
    let (af, bf) = {
      let mut xf = |prefix: &str, xf| {
        (0..3usize).map(|i| {
          (0..3u16).map(|j| {
            let ptr = quote!(#xf[#i].add((x + #j) as _));
            // x will always leave us aligned
            let v = if j == 0 {
              calc_ty.uload(ptr)
            } else {
              calc_ty.uload(ptr)
            };
            v.let_(&mut looop, format_args!("{}_{}_{}", prefix, i, j))
          })
            .collect::<Vec<_>>()
        })
          .collect::<Vec<_>>()
      };
      (xf("af", quote!(af)),
       xf("bf", quote!(bf)))
    };

    let x = |xf: &[Vec<SimdValue>]| {
      &three * &(&xf[0][0] + &xf[2][0] + &xf[0][2] + &xf[2][2])
        + &four
        * &(&xf[1][0]
          + &xf[0][1]
          + &xf[1][1]
          + &xf[2][1]
          + &xf[1][2])
    };
    let a = x(&af)
      .let_(&mut looop, "a");
    let b = x(&bf)
      .let_(&mut looop, "b");

    let p = px_ty.uload(quote!(cdeffed))
      .let_(&mut looop, "p")
      .cast(calc_ty)
      .let_(&mut looop, "p");

    let v = a * p + b;
    let v = v.round_shift(quote!(shift));

    let simd_width = simd_width as u16;
    func.extend(quote! {
      while range.start < range.end {
        let x = range.start;
        #looop

        *f = #v;

        range.start += #simd_width;
        f = f.add(1usize);
        cdeffed = cdeffed.add(#simd_width as _);
      }
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.box_f_rn_internal.insert(key, func).is_none());
    self.funcs.box_f_rn_internal.get(&key)
      .unwrap()
      .name()
  }
  fn box_f_r2_internal(&mut self, simd_width: usize) -> Ident {
    let key = (2, simd_width);
    if let Some(f) = self.funcs.box_f_rn_internal.get(&key) {
      return f.name();
    }

    let calc = PrimType::U32;
    let px = self.px;
    let calc_ty = SimdType::new(calc, simd_width);
    let px_ty = SimdType::new(px.into(), simd_width);

    let args = quote! {(
      af: [&[#calc; IMAGE_WIDTH_MAX + 2]; 2],
      bf: [&[#calc; IMAGE_WIDTH_MAX + 2]; 2],
      mut f0: *mut #calc, mut f1: *mut #calc,
      mut cdeffed: *const #px,
      cdeffed_stride: i32,
      mut range: Range<u16>,
    )};
    let mut func = self.out
      .new_priv_func(format_args!("box_f_r2_internal_{}", simd_width),
                     args, self.px, self.isa);

    func.extend(quote! {
      let shift = (5 + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS) as #calc;
      let shifto = (4 + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS) as #calc;
      let af = [af[0].as_ptr(), af[1].as_ptr()];
      let bf = [bf[0].as_ptr(), bf[1].as_ptr()];
      let mut f0 = f0 as *mut #calc_ty;
      let mut f1 = f1 as *mut #calc_ty;
    });
    let five = calc_ty.splat(quote!(5))
      .let_(&mut *func, "five");
    let six  = calc_ty.splat(quote!(6))
      .let_(&mut *func, "six");

    let mut looop = TokenStream::default();
    let (af, bf) = {
      let mut xf = |prefix: &str, xf| {
        (0..2usize).map(|i| {
          (0..3u16).map(|j| {
            let ptr = quote!(#xf[#i].add((x + #j) as usize));
            // x will always leave us aligned
            let v = if j == 0 {
              calc_ty.uload(ptr)
            } else {
              calc_ty.uload(ptr)
            };
            v.let_(&mut looop, format_args!("{}_{}_{}", prefix, i, j))
          })
            .collect::<Vec<_>>()
        })
          .collect::<Vec<_>>()
      };
      (xf("af", quote!(af)), xf("bf", quote!(bf)))
    };

    let x = |i: usize, xf: &[Vec<SimdValue>]| {
      &five * &(&xf[i][0] + &xf[i][2] + &six * &xf[i][1])
    };

    let a = x(0, &af)
      .let_(&mut looop, "a");
    let b = x(0, &bf)
      .let_(&mut looop, "b");
    let ao = x(1, &af)
      .let_(&mut looop, "ao");
    let bo = x(1, &bf)
      .let_(&mut looop, "bo");

    let p = px_ty.uload(quote!(cdeffed))
      .let_(&mut looop, "p")
      .cast(calc_ty)
      .let_(&mut looop, "p");
    let po = px_ty.uload(quote!(cdeffed.add(cdeffed_stride as usize)))
      .let_(&mut looop, "po")
      .cast(calc_ty)
      .let_(&mut looop, "po");

    let v = (&a + &ao) * &p + &b + &bo;
    let vo = &ao * &po + &bo;

    let v = v.round_shift(quote!(shift));
    let vo = vo.round_shift(quote!(shifto));

    let simd_width = simd_width as u16;
    func.extend(quote! {
      while range.start < range.end {
        let x = range.start;
        #looop

        *f0 = #v;
        *f1 = #vo;

        range.start += #simd_width;
        f0 = f0.add(1usize);
        f1 = f1.add(1usize);
        cdeffed = cdeffed.add(#simd_width as _);
      }
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.box_f_rn_internal.insert(key, func).is_none());
    self.funcs.box_f_rn_internal.get(&key)
      .unwrap()
      .name()
  }
  fn box_f_r0(&mut self, width: usize) -> Ident {
    let key = (0, width);
    if let Some(f) = self.funcs.box_f_rn.get(&key) {
      return f.name();
    }

    let px = self.px;
    let calc = PrimType::U32;
    let simd_width = width
      .min(self.simd_width(calc))
      .min(self.simd_width(px.into()));

    let args = quote! {(
      f: *mut #calc,
      cdeffed: *const #px,
      // don't need cdeffed's stride
    )};
    let mut func = self.out
      .new_func(format_args!("box_f_r0_{}", width),
                args, self.px, self.isa);

    let internal = self.box_f_r0_internal(simd_width);
    func.extend(quote! {
      #internal(f, cdeffed, 0..#width as _);
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.box_f_rn.insert(key, func).is_none());
    self.funcs.box_f_rn.get(&key)
      .unwrap()
      .name()
  }
  fn box_f_r1(&mut self, width: usize) -> Ident {
    let key = (1, width);
    if let Some(f) = self.funcs.box_f_rn.get(&key) {
      return f.name();
    }

    let px = self.px;
    let calc = PrimType::U32;
    let simd_width = width
      .min(self.simd_width(calc))
      .min(self.simd_width(px.into()));

    let args = quote! {(
      af: [&[#calc; IMAGE_WIDTH_MAX + 2]; 3],
      bf: [&[#calc; IMAGE_WIDTH_MAX + 2]; 3],
      f: *mut #calc,
      cdeffed: *const #px,
      // don't need cdeffed's stride
    )};
    let mut func = self.out
      .new_func(format_args!("box_f_r1_{}", width),
                     args, self.px, self.isa);

    let internal = self.box_f_r1_internal(simd_width);
    func.extend(quote! {
      #internal(af, bf, f, cdeffed, 0..#width as _);
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.box_f_rn.insert(key, func).is_none());
    self.funcs.box_f_rn.get(&key)
      .unwrap()
      .name()
  }
  fn box_f_r2(&mut self, width: usize) -> Ident {
    let key = (2, width);
    if let Some(f) = self.funcs.box_f_rn.get(&key) {
      return f.name();
    }

    let px = self.px;
    let calc = PrimType::U32;
    let simd_width = width
      .min(self.simd_width(calc))
      .min(self.simd_width(px.into()));

    let args = quote! {(
      af: [&[#calc; IMAGE_WIDTH_MAX + 2]; 2],
      bf: [&[#calc; IMAGE_WIDTH_MAX + 2]; 2],
      f0: *mut #calc, f1: *mut #calc,
      cdeffed: *const #px,
      cdeffed_stride: i32,
    )};
    let mut func = self.out
      .new_func(format_args!("box_f_r2_{}", width),
                args, self.px, self.isa);

    let internal = self.box_f_r2_internal(simd_width);
    func.extend(quote! {
      #internal(af, bf, f0, f1, cdeffed, cdeffed_stride, 0..#width as _);
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.box_f_rn.insert(key, func).is_none());
    self.funcs.box_f_rn.get(&key)
      .unwrap()
      .name()
  }

  fn sgrproj_internal(&mut self, w: usize) -> Ident {
    let (calc, scalc) = (PrimType::U32, PrimType::I32);
    let simd_width = w.min(self.simd_width(calc));

    if let Some(f) = self.funcs.sgrproj_internal.get(&simd_width) {
      return f.name();
    }

    let px = self.px;
    let calc_ty = SimdType::new(calc, simd_width);
    let scalc_ty = SimdType::new(scalc, simd_width);

    let args = quote! {(
      set: u8,
      mut iimg: *const #calc,
      mut iimg_sq: *const #calc,
      iimg_stride: u32,
      mut cdeffed: *const #px,
      cdeffed_stride: i32,
      bdm8: u8,
      width: u16,
      height: u16,
      mut apply_filter: AF,
    )};
    let params = vec![
      (quote!(AF), quote! {
        FnMut(
          /*y:*/ u16, /*x:*/ u16,
          /*f_r2_ab:*/ #scalc_ty,
          /*f_r1:*/ #scalc_ty,
        )
      }),
    ];
    let mut func = self
      .new_func(format_args!("sgrproj_internal_{}", simd_width),
                args, params, true);

    let box_ab_r1 = self.box_ab_rn_internal(1, simd_width);
    let box_ab_r2 = self.box_ab_rn_internal(2, simd_width);
    let box_f_r2 = self.box_f_r2_internal(simd_width);
    let box_f_r1 = self.box_f_r1_internal(simd_width);
    let box_f_r0 = self.box_f_r0_internal(simd_width);

    let simd_width = simd_width as u16;
    let width = w as u16;

    let range = quote!(0..#width);

    func.extend(quote! {
      let mut a_r2: [AlignedArray<[#calc; IMAGE_WIDTH_MAX + 2]>; 2] =
        [AlignedArray::uninitialized(); 2];
      let mut b_r2: [AlignedArray<[#calc; IMAGE_WIDTH_MAX + 2]>; 2] =
        [AlignedArray::uninitialized(); 2];
      let mut f_r2_0: AlignedArray<[#calc; IMAGE_WIDTH_MAX]> =
        AlignedArray::uninitialized();
      let mut f_r2_1: AlignedArray<[#calc; IMAGE_WIDTH_MAX]> =
        AlignedArray::uninitialized();
      let mut a_r1: [AlignedArray<[#calc; IMAGE_WIDTH_MAX + 2]>; 3] =
        [AlignedArray::uninitialized(); 3];
      let mut b_r1: [AlignedArray<[#calc; IMAGE_WIDTH_MAX + 2]>; 3] =
        [AlignedArray::uninitialized(); 3];
      let mut f_r1: AlignedArray<[#calc; IMAGE_WIDTH_MAX]> =
        AlignedArray::uninitialized();

      let s_r2: u32 = SGRPROJ_PARAMS_S[set as usize][0];
      let s_r1: u32 = SGRPROJ_PARAMS_S[set as usize][1];

      /* prime the intermediate arrays */
      // One oddness about the radius=2 intermediate array computations that
      // the spec doesn't make clear: Although the spec defines computation
      // of every row (of a, b and f), only half of the rows (every-other
      // row) are actually used.
      if s_r2 > 0 {
        #box_ab_r2(
          a_r2[0].as_mut_ptr(),
          b_r2[0].as_mut_ptr(),
          iimg,
          iimg_sq,
          iimg_stride,
          s_r2,
          bdm8,
          #range,
        );
      }
      if s_r1 > 0 {
        let iimg_offset = iimg_stride + 1;
        #box_ab_r1(
          a_r1[0].as_mut_ptr(),
          b_r1[0].as_mut_ptr(),
          iimg.add(iimg_offset as _),
          iimg_sq.add(iimg_offset as _),
          iimg_stride,
          s_r1,
          bdm8,
          #range,
        );
        #box_ab_r1(
          a_r1[1].as_mut_ptr(),
          b_r1[1].as_mut_ptr(),
          iimg.add((iimg_stride + iimg_offset) as _),
          iimg_sq.add((iimg_stride + iimg_offset) as _),
          iimg_stride,
          s_r1,
          bdm8,
          #range,
        );
      }

      /* iterate by row */
      // Increment by two to handle the use of even rows by r=2 and run a nested
      //  loop to handle increments of one.
      let mut y = 0u16;
      while y < height {
      //for y in (0..stripe_h).step_by(2) {
        // get results to use y and y+1
        let f_r2_ab: [_; 2] = if s_r2 > 0 {
          #box_ab_r2(
            a_r2[((y / 2 + 1) % 2) as usize].as_mut_ptr(),
            b_r2[((y / 2 + 1) % 2) as usize].as_mut_ptr(),
            iimg.add((2 * iimg_stride) as _),
            iimg_sq.add((2 * iimg_stride) as _),
            iimg_stride,
            s_r2,
            bdm8,
            #range,
          );
          let ap0: [_; 2] = [&*a_r2[((y / 2) % 2) as usize], &*a_r2[((y / 2 + 1) % 2) as usize]];
          let bp0: [_; 2] = [&*b_r2[((y / 2) % 2) as usize], &*b_r2[((y / 2 + 1) % 2) as usize]];
          #box_f_r2(
            ap0,
            bp0,
            f_r2_0.as_mut_ptr(),
            f_r2_1.as_mut_ptr(),
            cdeffed,
            cdeffed_stride,
            #range,
          );
          [&f_r2_0, &f_r2_1]
        } else {
          #box_f_r0(
            f_r2_0.as_mut_ptr(),
            cdeffed,
            #range,
          );
          // share results for both rows
          [&f_r2_0, &f_r2_0]
        };

        let mut dy = 0u16;
        while dy < 2 && y + dy < height {
        //for dy in 0..(2.min(height - y)) {
          let y = y + dy;
          if s_r1 > 0 {
            let iimg_offset = 2 * iimg_stride + iimg_stride + 1;
            #box_ab_r1(
              a_r1[((y + 2) % 3) as usize].as_mut_ptr(),
              b_r1[((y + 2) % 3) as usize].as_mut_ptr(),
              iimg.add(iimg_offset as _),
              iimg_sq.add(iimg_offset as _),
              iimg_stride,
              s_r1,
              bdm8,
              #range,
            );
            let ap1: [_; 3] = [
              &*a_r1[((y + 0) % 3) as usize],
              &*a_r1[((y + 1) % 3) as usize],
              &*a_r1[((y + 2) % 3) as usize],
            ];
            let bp1: [_; 3] = [
              &*b_r1[((y + 0) % 3) as usize],
              &*b_r1[((y + 1) % 3) as usize],
              &*b_r1[((y + 2) % 3) as usize],
            ];
            #box_f_r1(
              ap1,
              bp1,
              f_r1.as_mut_ptr(),
              cdeffed,
              #range,
            );
          } else {
            #box_f_r0(
              f_r1.as_mut_ptr(),
              cdeffed,
              #range,
            );
          }

          let mut x = 0u16;
          while x < #width {
            let f_r2_ab = f_r2_ab[dy as usize].as_ptr().add(x as _);
            let f_r2_ab = *(f_r2_ab as *const #calc_ty);
            let f_r1 = *(f_r1.as_ptr().add(x as _) as *const #calc_ty);
            apply_filter(y, x, f_r2_ab.cast(), f_r1.cast());
            x += #simd_width;
          }
          dy += 1;
        }

        y += 2;
        iimg = iimg.add(iimg_stride as _);
        iimg_sq = iimg_sq.add(iimg_stride as _);
        cdeffed = cdeffed.add(cdeffed_stride as _);
      }
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.sgrproj_internal.insert(simd_width as _, func).is_none());
    self.funcs.sgrproj_internal.get(&(simd_width as _))
      .unwrap()
      .name()
  }

  fn solve(&mut self, w: usize) -> Ident {
    let (calc, scalc) = (PrimType::U32, PrimType::I32);
    let simd_width = w.min(self.simd_width(calc));

    if let Some(f) = self.funcs.solve_filter.get(&w) {
      return f.name();
    }

    let px = self.px;
    let px_ty = SimdType::new(px.into(), simd_width);
    let scalc_ty = SimdType::new(scalc, simd_width);

    let args = quote! {(
      set: u8,
      iimg: *const #calc,
      iimg_sq: *const #calc,
      mut cdeffed: *const #px,
      cdeffed_stride: i32,
      mut input: *const #px,
      input_stride: i32,
      bdm8: u8,
      height: u16,
    ) -> (i8, i8) };
    let params = vec![];
    let mut func = self
      .new_func(format_args!("sgrproj_solve_{}", w),
                args, params, true);

    let sgrproj = self.sgrproj_internal(simd_width);

    let mut apply_filter = TokenStream::default();
    let cdeffed = quote!(cdeffed.add(((y as i32) * cdeffed_stride + x as i32) as _));
    let input = quote!(input.add(((y as i32) * input_stride + x as i32) as _));
    let u = px_ty.uload(cdeffed)
      .cast(scalc_ty)
      .shl(quote!(SGRPROJ_RST_BITS))
      .let_(&mut apply_filter, "u");
    let s = px_ty.uload(input)
      .cast(scalc_ty)
      .shl(quote!(SGRPROJ_RST_BITS))
      .let_(&mut apply_filter, "s");
    let s = s - &u;
    let s = s.let_(&mut apply_filter, "s");

    // packed_simd is again limiting us on the vector sizes for some types.
    let f64_simd_width = PrimType::F64.max_simd_width();
    let f64_ty = SimdType::new(PrimType::F64, f64_simd_width);

    let f_r2_ab = SimdValue::from(scalc_ty, quote!(f_r2_ab));
    let f_r1 = SimdValue::from(scalc_ty, quote!(f_r1));
    let f2 = (f_r2_ab - &u).let_(&mut apply_filter, "f2");
    let f1 = (f_r1 - &u).let_(&mut apply_filter, "f1");

    let h00_acc = Var::let_mut(&mut apply_filter, "h00_acc", 0f64);
    let h11_acc = Var::let_mut(&mut apply_filter, "h11_acc", 0f64);
    let h01_acc = Var::let_mut(&mut apply_filter, "h01_acc", 0f64);
    let c0_acc = Var::let_mut(&mut apply_filter, "c0_acc", 0f64);
    let c1_acc = Var::let_mut(&mut apply_filter, "c1_acc", 0f64);

    for chunk in (0..simd_width).step_by(f64_simd_width) {
      let range = chunk..chunk+f64_simd_width;
      let f2 = f2.select_range(range.clone())
        .let_(&mut apply_filter, "f2c")
        .cast(f64_ty)
        .let_(&mut apply_filter, "f2c");
      let f1 = f1.select_range(range.clone())
        .let_(&mut apply_filter, "f1c")
        .cast(f64_ty)
        .let_(&mut apply_filter, "f1c");
      let s = s.select_range(range.clone())
        .let_(&mut apply_filter, "sc")
        .cast(f64_ty)
        .let_(&mut apply_filter, "sc");

      apply_filter.extend(quote! {
        #h00_acc += (#f2 * #f2).sum();
      });
      apply_filter.extend(quote! {
        #h11_acc += (#f1 * #f1).sum();
      });
      apply_filter.extend(quote! {
        #h01_acc += (#f2 * #f1).sum();
      });
      apply_filter.extend(quote! {
        #c0_acc += (#f2 * #s).sum();
      });
      apply_filter.extend(quote! {
        #c1_acc += (#f1 * #s).sum();
      });
    }

    // now emit the atomic add:
    // we have to use intrinsics because AtomicF64 doesn't exist.
    apply_filter.extend(quote! {
      h[0][0] += #h00_acc;
      h[1][1] += #h11_acc;
      h[0][1] += #h01_acc;
      c[0] += #c0_acc;
      c[1] += #c1_acc;
    });

    func.extend(quote! {
      let mut h: [[f64; 2]; 2] = [[0., 0.], [0., 0.]];
      let mut c: [f64; 2] = [0., 0.];

      let iimg_stride = SOLVE_IMAGE_STRIDE as u32;
      let width = #w as u16;

      let mut apply_filter = |y, x, f_r2_ab, f_r1| {
        #apply_filter
      };

      #sgrproj(set, iimg, iimg_sq, iimg_stride, cdeffed, cdeffed_stride,
               bdm8, width, height, apply_filter);

      // now finish:
      let s_r2: u32 = SGRPROJ_PARAMS_S[set as usize][0];
      let s_r1: u32 = SGRPROJ_PARAMS_S[set as usize][1];
      // this is lifted almost in-tact from libaom
      let n = width as f64 * height as f64;
      h[0][0] /= n;
      h[0][1] /= n;
      h[1][1] /= n;
      h[1][0] = h[0][1];
      c[0] *= (1 << SGRPROJ_PRJ_BITS) as f64 / n;
      c[1] *= (1 << SGRPROJ_PRJ_BITS) as f64 / n;
      let (xq0, xq1) = if s_r2 == 0 {
        // H matrix is now only the scalar h[1][1]
        // C vector is now only the scalar c[1]
        if h[1][1] == 0. {
          (0, 0)
        } else {
          (0, (c[1] / h[1][1]).round() as i32)
        }
      } else if s_r1 == 0 {
        // H matrix is now only the scalar h[0][0]
        // C vector is now only the scalar c[0]
        if h[0][0] == 0. {
          (0, 0)
        } else {
          ((c[0] / h[0][0]).round() as i32, 0)
        }
      } else {
        let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
        if det == 0. {
          (0, 0)
        } else {
          // If scaling up dividend would overflow, instead scale down the divisor
          let div1 = h[1][1] * c[0] - h[0][1] * c[1];
          let div2 = h[0][0] * c[1] - h[1][0] * c[0];
          ((div1 / det).round() as i32, (div2 / det).round() as i32)
        }
      };
      {
        let xqd0 =
          clamp(xq0, SGRPROJ_XQD_MIN[0] as i32, SGRPROJ_XQD_MAX[0] as i32);
        let xqd1 = clamp(
          (1 << SGRPROJ_PRJ_BITS) - xqd0 - xq1,
          SGRPROJ_XQD_MIN[1] as i32,
          SGRPROJ_XQD_MAX[1] as i32,
        );
        (xqd0 as i8, xqd1 as i8)
      }
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.solve_filter.insert(w, func).is_none());
    self.funcs.solve_filter.get(&w)
      .unwrap()
      .name()
  }
  fn stripe(&mut self, w: usize) -> Ident {
    let (calc, scalc) = (PrimType::U32, PrimType::I32);
    let simd_width = w.min(self.simd_width(calc));

    if let Some(f) = self.funcs.stripe_filter.get(&w) {
      return f.name();
    }

    let px = self.px;
    let px_ty = SimdType::new(px.into(), simd_width);
    let scalc_ty = SimdType::new(scalc, simd_width);

    let args = quote! {(
      set: u8,
      xqd: [i8; 2],
      iimg: *const #calc,
      iimg_sq: *const #calc,
      iimg_stride: u32,
      mut cdeffed: *const #px,
      cdeffed_stride: i32,
      mut out: *mut #px,
      out_stride: i32,
      bdm8: u8,
      height: u16,
    )};
    let params = vec![];
    let mut func = self
      .new_func(format_args!("sgrproj_stripe_{}", w),
                args, params, true);

    let w0 = scalc_ty
      .splat(quote!(xqd[0] as #scalc))
      .let_(&mut *func, "w0");
    let w1 = scalc_ty
      .splat(quote!(xqd[1] as #scalc))
      .let_(&mut *func, "w1");
    let w2 = scalc_ty
      .splat(quote!((1 << SGRPROJ_PRJ_BITS) as #scalc));
    let w2 = (w2 - &w0 - &w1).let_(&mut *func, "w2");

    let sgrproj = self.sgrproj_internal(w);

    let mut apply_filter = TokenStream::default();

    let f_r2_ab = SimdValue::from(scalc_ty, quote!(f_r2_ab));
    let f_r1 = SimdValue::from(scalc_ty, quote!(f_r1));
    let cdeffed = quote!(cdeffed.add(((y as i32) * cdeffed_stride + x as i32) as _));
    let out = quote!(out.add(((y as i32) * out_stride + x as i32) as _));
    let u = px_ty.uload(cdeffed)
      .cast(scalc_ty)
      .shl(quote!(SGRPROJ_RST_BITS))
      .let_(&mut apply_filter, "u");
    let v = (&w0 * &f_r2_ab + &w1 * &u + &w2 * &f_r1)
      .let_(&mut apply_filter, "v");
    let s = v.round_shift(quote!((SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) as u32))
      .let_(&mut apply_filter, "s");

    // now clamp:
    let s = s.max(&s.ty().splat(quote!(0)));
    let s = if self.px == PixelType::U16 {
      s.min(&s.ty().splat(quote!((1 << (bdm8 + 8)) - 1)))
    } else {
      s
    };
    let s = s.cast(px_ty)
      .let_(&mut apply_filter, "s");
    s.ustore(&mut apply_filter, out);

    func.extend(quote! {
      let width = #w as u16;
      let mut apply_filter = |y, x, f_r2_ab, f_r1| {
        #apply_filter
      };

      #sgrproj(set, iimg, iimg_sq, iimg_stride, cdeffed, cdeffed_stride,
               bdm8, width, height, apply_filter);
    });

    func.to_tokens(&mut **self.out);
    assert!(self.funcs.stripe_filter.insert(w, func).is_none());
    self.funcs.stripe_filter.get(&w)
      .unwrap()
      .name()
  }

  fn gen_kernels(&mut self) {
    for &w in [32, 64, 128].iter() {
      self.solve(w);
      self.stripe(w);
    }
  }

  fn add_solve_kernels(&self, set: &mut KernelDispatch) {
    let mut add = |w, path: Rc<TokenStream>| {
      let feature_idx = self.isa.index();
      let b_enum = Block(w, w).table_idx();

      let i = quote! {
        [#feature_idx][#b_enum]
      };
      set.push_kernel(self.px, i, path.clone());
    };

    for &w in [32, 64, 128].iter() {
      let path = self.funcs.solve_filter.get(&w)
        .unwrap()
        .path();
      add(w, path);
    }
  }
  fn add_stripe_kernels(&self, set: &mut KernelDispatch) {
    let mut add = |w, path: Rc<TokenStream>| {
      let feature_idx = self.isa.index();
      let b_enum = Block(w, w).table_idx();

      let i = quote! {
        [#feature_idx][#b_enum]
      };
      set.push_kernel(self.px, i, path.clone());
    };

    for &w in [32, 64, 128].iter() {
      let path = self.funcs.stripe_filter.get(&w)
        .unwrap()
        .path();
      add(w, path);
    }
  }
}

pub fn lrf_kernels(file: &mut dyn Write) {
  write_prelude(file);

  let args = vec![
    (quote!(set), quote!(u8)),
    (quote!(xqd), quote!([i8; 2])),
    (quote!(iimg), quote!(*const u32)),
    (quote!(iimg_sq), quote!(*const u32)),
    (quote!(iimg_stride), quote!(u32)),
    (quote!(cdeffed), quote!(*const T)),
    (quote!(cdeffed_stride), quote!(i32)),
    (quote!(out), quote!(*mut T)),
    (quote!(out_stride), quote!(i32)),
    (quote!(bdm8), quote!(u8)),
    (quote!(height), quote!(u16)),
  ];
  let ret = None;
  let table_sizes = vec![];
  let mut stripe_kernels = KernelDispatch::new("StripeF", args,
                                               ret, "STRIPE",
                                               table_sizes);
  let args = vec![
    (quote!(set), quote!(u8)),
    (quote!(iimg), quote!(*const u32)),
    (quote!(iimg_sq), quote!(*const u32)),
    (quote!(cdeffed), quote!(*const T)),
    (quote!(cdeffed_stride), quote!(i32)),
    (quote!(input), quote!(*const T)),
    (quote!(input_stride), quote!(i32)),
    (quote!(bdm8), quote!(u8)),
    (quote!(height), quote!(u16)),
  ];
  let ret = Some(quote!((i8, i8)));
  let table_sizes = vec![];
  let mut solve_kernels = KernelDispatch::new("SolveF", args,
                                              ret, "SOLVE",
                                              table_sizes);

  let mut native = BTreeMap::default();

  let from_crate_root = &["lrf", ];
  for isa in IsaFeature::sets() {
    let mut isa_module = Module::new_root(from_crate_root,
                                          isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child(px.module_name());
      px_module.extend(quote! {
        use rcore::lrf::*;
        use rcore::util::clamp;
      });
      StdImports.to_tokens(&mut px_module);

      let mut ctx = Ctx {
        out: &mut px_module,
        isa,
        px,
        key: (),

        native: &mut native,

        funcs: Default::default(),
      };
      ctx.gen_kernels();
      ctx.add_stripe_kernels(&mut stripe_kernels);
      ctx.add_solve_kernels(&mut solve_kernels);

      ctx.finish();

      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  let kernels = stripe_kernels.len() + solve_kernels.len();
  println!("generated {} lrf kernels", kernels);

  let tables = stripe_kernels.tables();
  writeln!(file, r#"pub mod stripe {{
  use super::*;
  {}
}}"#, tables)
    .expect("write lrf stripe kernel tables");
  let tables = solve_kernels.tables();
  writeln!(file, r#"pub mod solve {{
  use super::*;
  {}
}}"#, tables)
    .expect("write lrf solve kernel tables");
}
