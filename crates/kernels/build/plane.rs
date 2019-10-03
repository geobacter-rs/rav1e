// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::fmt::Display;

use proc_macro2::{Ident, Span, TokenStream};
use quote::*;

/// A (ptr, row stride) pair. Typically passed as separate arguments,
/// both mutable as values.
#[derive(Clone, Debug)]
pub struct Plane {
  ptr: Ident,
  stride: TokenStream,
}
impl Plane {
  pub fn new(ptr_name: &Ident) -> Self {
    let stride_name = format!("{}_stride", ptr_name);
    let stride_name = Ident::new(&stride_name, Span::call_site());
    Plane { ptr: ptr_name.clone(), stride: quote!(#stride_name) }
  }
  pub fn new_stride<T>(ptr: &Ident, stride: T) -> Self
  where
    T: ToTokens,
  {
    Plane { ptr: ptr.clone(), stride: quote!(#stride) }
  }
  pub fn new_const_stride(ptr: Ident, stride: usize) -> Self {
    Plane { ptr, stride: quote!(#stride) }
  }
  pub fn new_named<T, U>(name: T, stride: U) -> Self
    where T: Display,
          U: ToTokens,
  {
    let name = name.to_string();
    let name = Ident::new(&name, Span::call_site());
    Plane {
      ptr: name,
      stride: quote!(#stride)
    }
  }

  pub fn name(&self) -> &Ident {
    &self.ptr
  }
  pub fn stride(&self) -> &TokenStream {
    &self.stride
  }

  pub fn add_rc<T, U>(&self, row: T, col: U) -> TokenStream
  where
    T: ToTokens,
    U: ToTokens,
  {
    let ptr = &self.ptr;
    let stride = &self.stride;
    quote! {
      #ptr.add(((#row) * (#stride) + (#col)) as usize)
    }
  }
  pub fn add_r<T>(&self, row: T) -> TokenStream
    where
      T: ToTokens,
  {
    let ptr = &self.ptr;
    let stride = &self.stride;
    quote! {
      #ptr.add(((#row) * (#stride)) as usize)
    }
  }
  pub fn add<T>(&self, col: T) -> TokenStream
  where
    T: ToTokens,
  {
    let ptr = &self.ptr;
    quote! { #ptr.add(#col) }
  }
}
impl ToTokens for Plane {
  fn to_tokens(&self, to: &mut TokenStream) {
    self.ptr.to_tokens(to);
  }
}
