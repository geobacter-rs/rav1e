// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! Helpers for zero overhead Plane iteration in Rust. Tailored to AV1 encoding,
//! thus we should never iterate over ~32k rows or cols, let alone ~65k, we
//! use u16 iteration variables, as u16 address calculations are cheap on X86/GPUs.
//! `Range` is specifically avoided: internally, `<Range<_> as Iterator>::next`
//! calls `mem::swap`, which calls `mem::copy_nonoverlapping`, which contains a
//! `debug_assert!` checking that the two arguments are indeed non-overlapping.
//! Note: libstd's debug_assertions profile setting is independent of our setting!
//! Thus even if we disable debug_assertions, libstd will still have them.
//! Now, most of the time, LLVM is able to determine that this `debug_assert!` is
//! always dead (because, you know, it is), but not *always*. Additionally, using
//! `Range` is harder on LLVM's unroller, even for static trip counts.
//! Thus, we have here C/C++-esk for-loops using while-loops. LLVM unrolls these
//! consistently when loop counts are known.

use super::*;
use crate::plane::*;

#[derive(Clone)]
pub struct BlockLoop {
  loop_vars: Vec<Plane>,

  iter_ty: PrimType,

  width: TokenStream,
  height: TokenStream,
}
impl BlockLoop {
  pub fn std() -> Self {
    Self::new(quote!(width), quote!(height),
              PrimType::U16)
  }
  pub fn std_u8() -> Self {
    Self::new(quote!(width), quote!(height),
              PrimType::U8)
  }
  pub fn new<T, U>(width: T, height: U, iter_ty: PrimType) -> Self
    where T: ToTokens, U: ToTokens,
  {
    BlockLoop {
      loop_vars: vec![],

      iter_ty,

      width: quote! { (#width) as #iter_ty },
      height: quote! { (#height) as #iter_ty },
    }
  }
  pub fn add_loop_var(&mut self, var: Plane) {
    self.loop_vars.push(var);
  }

  pub fn gen<F>(&self, into: &mut TokenStream,
                row_incr: u16, column_incr: u16,
                f: F)
    where F: FnOnce(&mut TokenStream, &Ident, &Ident, &[Plane]),
  {
    let width = &self.width;
    let height = &self.height;
    let iter_ty = self.iter_ty;
    let mut body = TokenStream::default();

    let r = Ident::new("r", Span::call_site());
    let c = Ident::new("c", Span::call_site());

    let row_vars = self.loop_vars
      .iter()
      .map(|p| {
        let s = format!("row_{}", p.name());
        let s = Ident::new(&s, Span::call_site());
        Plane::new_stride(&s, p.stride())
      })
      .collect::<Vec<_>>();
    let col_vars = self.loop_vars
      .iter()
      .map(|p| {
        let s = format!("col_{}", p.name());
        let s = Ident::new(&s, Span::call_site());
        Plane::new_stride(&s, p.stride())
      })
      .collect::<Vec<_>>();

    let mut let_row_vars = TokenStream::default();
    let mut let_col_vars = TokenStream::default();
    let mut next_row = TokenStream::default();
    let mut next_col = TokenStream::default();
    for ((row, col), var) in row_vars.iter().zip(col_vars.iter()).zip(self.loop_vars.iter()) {
      let var_name = var.name();
      let_row_vars.extend(quote! {
        let mut #row = #var_name;
      });
      let_col_vars.extend(quote! {
        let mut #col = #row;
      });
      let stride = var.stride();
      next_row.extend(quote! {
        #row = #row.offset(((#row_incr as i32) * #stride) as isize);
      });
      next_col.extend(quote! {
        #col = #col.offset((#column_incr) as isize);
      });
    }

    f(&mut body, &r, &c, &col_vars);

    into.extend(quote! {
      let mut riter = 0 as #iter_ty;

      #let_row_vars

      while riter < #height {
        let r = riter as i32;
        let mut citer = 0 as #iter_ty;

        #let_col_vars

        while citer < #width {
          let c = citer as i32;

          #body

          #next_col
          citer += (#column_incr as #iter_ty);
        }

        riter += (#row_incr as #iter_ty);
        #next_row
      }
    });
  }
}

#[derive(Clone)]
pub struct Loop<'a> {
  prefix: &'a str,
  loop_vars: Vec<Plane>,

  iter_ty: PrimType,

  width: TokenStream,
}
impl<'a> Loop<'a> {
  pub fn new<T>(name: &'a str, width: T, iter_ty: PrimType) -> Self
    where T: ToTokens,
  {
    Loop {
      prefix: name,
      loop_vars: vec![],

      iter_ty,

      width: quote! { (#width) as #iter_ty },
    }
  }
  pub fn add_loop_var(&mut self, var: Plane) {
    self.loop_vars.push(var);
  }

  pub fn gen<F>(&self, into: &mut TokenStream,
                incr: u16, f: F)
    where F: FnOnce(&mut TokenStream, &Ident, &[Plane]),
  {
    let width = &self.width;
    let iter_ty = self.iter_ty;
    let mut body = TokenStream::default();

    let r_iter = Ident::new(&format!("{}_iter", self.prefix),
                            Span::call_site());
    let r = Ident::new(&self.prefix, Span::call_site());

    let vars = self.loop_vars
      .iter()
      .map(|p| {
        let s = format!("{}_{}", self.prefix, p.name());
        let s = Ident::new(&s, Span::call_site());
        Plane::new_stride(&s, p.stride())
      })
      .collect::<Vec<_>>();

    let mut let_row_vars = TokenStream::default();
    let mut next_row = TokenStream::default();
    for (iter_var, var) in vars.iter().zip(self.loop_vars.iter()) {
      let var_name = var.name();
      let_row_vars.extend(quote! {
        let mut #iter_var = #var_name;
      });
      let stride = var.stride();
      next_row.extend(quote! {
        #iter_var = #iter_var.add(((#incr as i32) * #stride) as usize);
      });
    }

    f(&mut body, &r, &vars);

    into.extend(quote! {
      let mut #r_iter = 0 as #iter_ty;

      #let_row_vars

      while #r_iter < #width {
        let #r = #r_iter as i32;

        #body

        #r_iter += (#incr as #iter_ty);
        #next_row
      }
    });
  }
}
