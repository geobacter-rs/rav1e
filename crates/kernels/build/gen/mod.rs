// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::collections::BTreeMap;
use std::env;
use std::fmt::Display;
use std::iter::*;
use std::io::Write;
use std::ops::{self, *};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use super::*;

pub mod dist;
pub mod lrf;
pub mod mc;
pub mod predict;
pub mod tx;

use self::plane::Plane;
use self::loops::*;
use self::NegIdx::*;

pub const MAX_TX_SIZE: usize = 64;

pub struct StdImports;
impl ToTokens for StdImports {
  fn to_tokens(&self, to: &mut TokenStream) {
    to.extend(quote! {
      use rcore::cpu_features::CpuFeatureLevel;
      use rcore::partition::BlockSize;
      use rcore::util::{AlignedArray, ISimd, round_shift, };

      use packed_simd::{Simd, i32x8, m8, FromCast, shuffle, Cast, };

      use std::ops::Range;
    });
  }
}

pub fn macro_<T>(mac: T) -> TokenStream
where
  T: Display,
{
  let mac = Ident::new(&mac.to_string(), Span::call_site());
  let mut ts = quote!(#mac);
  ts.extend(
    Some(TokenTree::Punct(Punct::new('!', Spacing::Alone).into())).into_iter(),
  );
  ts
}
pub fn call_macro<T, U>(mac: T, args: U) -> TokenStream
where
  T: Display,
  U: ToTokens,
{
  let mac = Ident::new(&mac.to_string(), Span::call_site());
  let mut ts = quote!(#mac);
  ts.extend(
    Some(TokenTree::Punct(Punct::new('!', Spacing::Alone).into())).into_iter(),
  );
  args.to_tokens(&mut ts);
  ts
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Block(usize, usize);
impl Block {
  fn blocks() -> &'static [Block] {
    const C: &'static [Block] = &[
      Block(2, 2),
      Block(2, 4),
      Block(2, 8),
      Block(4, 2),
      Block(4, 4),
      Block(4, 8),
      Block(4, 16),
      Block(8, 2),
      Block(8, 4),
      Block(8, 8),
      Block(8, 16),
      Block(8, 32),
      Block(16, 4),
      Block(16, 8),
      Block(16, 16),
      Block(16, 32),
      Block(16, 64),
      Block(32, 8),
      Block(32, 16),
      Block(32, 32),
      Block(32, 64),
      Block(64, 16),
      Block(64, 32),
      Block(64, 64),
      Block(64, 128),
      Block(128, 64),
      Block(128, 128),
    ];
    C
  }
  fn blocks_iter() -> impl Iterator<Item = Block> {
    Self::blocks().iter().cloned()
  }
  fn tx_sizes_iter() -> impl Iterator<Item = Block> {
    Self::blocks_iter()
      .filter(|b| b.w() > 2 && b.h() > 2 )
      .filter(|b| b.w() <= MAX_TX_SIZE && b.h() <= MAX_TX_SIZE )
  }
  fn fn_suffix(&self) -> String {
    format!("{}_{}", self.0, self.1)
  }

  pub fn w(&self) -> usize {
    self.0
  }
  pub fn h(&self) -> usize {
    self.1
  }
  pub fn area(&self) -> usize {
    self.w() * self.h()
  }
  fn ilog(this: usize) -> isize {
    use std::mem::size_of;
    (size_of::<usize>() * 8 - this.leading_zeros() as usize) as isize
  }
  pub fn rect_log_ratio(&self) -> u32 {
    (Self::ilog(self.w()) - Self::ilog(self.h())).abs() as _
  }
  pub fn transpose(&self) -> Self {
    Block(self.1, self.0)
  }

  fn table_idx(&self) -> usize {
    let w = Self::ilog(self.w()) as usize;
    let h = Self::ilog(self.h()) as usize;

    w << 4 | h
  }
  fn table_len() -> usize {
    Block(128, 128).table_idx() + 1
  }

  fn as_type(&self) -> BlockType {
    BlockType(*self)
  }
  fn as_enum(&self) -> BlockEnum {
    BlockEnum(*self)
  }
  fn as_tx(&self) -> TokenStream {
    let s = format!("TX_{}X{}", self.0, self.1);
    let s = Ident::new(&s, Span::call_site());
    quote!(crate::transform::TxSize::#s)
  }
}

struct BlockType(Block);
struct BlockEnum(Block);
impl ToTokens for BlockType {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let s = format!("Block{}x{}", (self.0).0, (self.0).1);
    let s = Ident::new(&s, Span::call_site());
    tokens.extend(quote!(crate::util::#s));
  }
}
impl ToTokens for BlockEnum {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let s = format!("BLOCK_{}X{}", (self.0).0, (self.0).1);
    let s = Ident::new(&s, Span::call_site());
    tokens.extend(quote!(crate::partition::BlockSize::#s));
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum PixelType {
  U8,
  U16,
}
impl PixelType {
  fn types() -> &'static [PixelType] {
    const C: &'static [PixelType] = &[PixelType::U8, PixelType::U16];
    C
  }
  fn types_iter() -> impl Iterator<Item = PixelType> {
    Self::types().iter().cloned()
  }
  fn type_str(&self) -> &'static str {
    match self {
      PixelType::U8 => "u8",
      PixelType::U16 => "u16",
    }
  }
  fn module_name(&self) -> Ident {
    let s = match self {
      PixelType::U8 => "p_u8",
      PixelType::U16 => "p_u16",
    };
    Ident::new(s, Span::call_site())
  }
  fn avx2_width(&self) -> usize {
    match self {
      PixelType::U8 => 256 / 8,
      PixelType::U16 => 256 / 16,
    }
  }
}
impl ToTokens for PixelType {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    tokens.extend(match self {
      PixelType::U8 => quote!(u8),
      PixelType::U16 => quote!(u16),
    });
  }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum PrimType {
  U8,
  I8,
  U16,
  I16,
  U32,
  I32,
  F32,
  F64,
}
impl PrimType {
  fn type_str(&self) -> &'static str {
    match self {
      PrimType::U8 => "u8",
      PrimType::I8 => "i8",
      PrimType::U16 => "u16",
      PrimType::I16 => "i16",
      PrimType::U32 => "u32",
      PrimType::I32 => "i32",
      PrimType::F32 => "f32",
      PrimType::F64 => "f64",
    }
  }
  fn bytes(&self) -> usize {
    match self {
      PrimType::U8 | PrimType::I8 => 1,
      PrimType::U16 | PrimType::I16 => 2,
      PrimType::U32 | PrimType::I32 | PrimType::F32 => 4,
      PrimType::F64 => 8,
    }
  }
  fn bits(&self) -> usize {
    self.bytes() * 8
  }
  fn avx2_width(&self) -> usize {
    256 / self.bits()
  }
  // max width available in `packed_simd`
  fn max_simd_width(&self) -> usize {
    512 / self.bits()
  }
  fn is_signed(&self) -> bool {
    match self {
      PrimType::I8 | PrimType::I16 | PrimType::I32 |
      PrimType::F32 | PrimType::F64 => true,
      _ => false,
    }
  }
}
impl ToTokens for PrimType {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    tokens.extend(match self {
      PrimType::U8 => quote!(u8),
      PrimType::I8 => quote!(i8),
      PrimType::U16 => quote!(u16),
      PrimType::I16 => quote!(i16),
      PrimType::U32 => quote!(u32),
      PrimType::I32 => quote!(i32),
      PrimType::F32 => quote!(f32),
      PrimType::F64 => quote!(f64),
    });
  }
}
impl From<PixelType> for PrimType {
  fn from(px: PixelType) -> Self {
    match px {
      PixelType::U8 => PrimType::U8,
      PixelType::U16 => PrimType::U16,
    }
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SimdType(PrimType, usize);
impl SimdType {
  fn new(prim: PrimType, width: usize) -> Self {
    SimdType(prim, width)
  }
  fn assume_type(&self, value: TokenStream) -> SimdValue {
    SimdValue(self.clone(), value)
  }
  fn ty(&self) -> PrimType {
    self.0
  }
  fn w(&self) -> usize {
    self.1
  }
  fn is_signed(&self) -> bool {
    self.ty().is_signed()
  }

  fn bit_width(&self) -> usize {
    self.ty().bits() * self.w()
  }

  fn uload<T>(&self, ptr: T) -> SimdValue
    where
      T: ToTokens,
  {
    SimdValue::uload(*self, ptr)
  }
  fn aload<T>(&self, ptr: T) -> SimdValue
    where
      T: ToTokens,
  {
    SimdValue::aload(*self, ptr)
  }

  fn splat<T>(&self, v: T) -> SimdValue
  where
    T: ToTokens,
  {
    let ety = self.ty();
    SimdValue::from(*self, quote!(<#self>::splat(#v as #ety)))
  }

  fn indices_ty(&self) -> SimdType {
    SimdType(PrimType::U32, self.w())
  }
  fn indices<F>(&self, mut f: F) -> SimdValue
  where
    F: FnMut(u32) -> TokenStream,
  {
    let mut idxs = TokenStream::default();
    for i in 0..self.w() {
      let v = f(i as _);
      idxs.extend(quote! { #v, });
    }
    let ty = self.indices_ty();
    SimdValue::from(ty, quote! { <#ty>::new(#idxs) })
  }
}
impl ToTokens for SimdType {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let ty = &self.0;
    let width = &self.1;
    tokens.extend(quote!(Simd<[#ty; #width]>));
  }
}
#[derive(Clone, Debug)]
struct SimdValue(SimdType, TokenStream);
impl SimdValue {
  fn from<T>(ty: SimdType, v: T) -> Self
  where
    T: ToTokens,
  {
    SimdValue(ty, v.into_token_stream())
  }
  fn default(ty: SimdType) -> Self {
    Self::from(ty, quote! { <#ty>::default() })
  }
  fn from_cast<T>(ty: SimdType, v: T) -> Self
  where
    T: ToTokens,
  {
    let v = quote!(<#ty>::from_cast(#v));
    SimdValue(ty, v)
  }
  fn uload<T>(ty: SimdType, ptr: T) -> Self
  where
    T: ToTokens,
  {
    let w = ty.w();
    let ety = ty.ty();
    let v = quote! {
      <#ty>::from(*(#ptr as *const [#ety; #w]))
    };
    SimdValue(ty, v)
  }
  fn aload<T>(ty: SimdType, ptr: T) -> Self
  where
    T: ToTokens,
  {
    let v = quote! {
      *(#ptr as *const #ty)
    };
    SimdValue(ty, v)
  }
  fn ustore<T>(&self, dst: &mut TokenStream, ptr: T)
  where
    T: ToTokens,
  {
    let w = self.ty().w();
    let ety = self.ty().ty();
    let v = quote! {
      *(#ptr as *mut [#ety; #w]) = ::std::mem::transmute::<_, [#ety; #w]>(#self);
    };
    dst.extend(v);
  }
  fn astore<T>(&self, dst: &mut TokenStream, ptr: T)
  where
    T: ToTokens,
  {
    let ty = self.ty();
    let v = quote! {
      *(#ptr as *mut #ty) = #self;
    };
    dst.extend(v);
  }
  fn ty(&self) -> SimdType {
    self.0
  }
  fn w(&self) -> usize {
    self.ty().w()
  }
  fn value(&self) -> &TokenStream {
    &self.1
  }
  fn unwrap_value(self) -> TokenStream {
    self.1
  }

  fn bitcast(&self, to: SimdType) -> Self {
    assert_eq!(self.ty().bit_width(), to.bit_width());
    let v = quote!(<#to>::from_bits(#self));
    SimdValue(to, v)
  }
  fn cast(&self, to: SimdType) -> Self {
    assert_eq!(self.ty().w(), to.w());
    let v = quote!(<#to>::from_cast(#self));
    SimdValue(to, v)
  }

  fn abs(&self) -> Self {
    SimdValue(self.ty(), quote!(#self.abs()))
  }
  fn clamp<T, U>(&self, min: T, max: U) -> Self
  where
    T: ToTokens,
    U: ToTokens,
  {
    let elem = self.ty().ty();
    let v = quote!(#self.clamp(#min as #elem, #max as #elem));
    SimdValue::from(self.ty(), v)
  }
  fn round_shift<T>(&self, bit: T) -> Self
    where
      T: ToTokens,
  {
    let v = quote!((#self).round_shift(#bit));
    SimdValue::from(self.ty(), v)
  }
  fn shl<T>(&self, bits: T) -> Self
    where
      T: ToTokens,
  {
    let v = quote!(#self << (#bits as u32));
    SimdValue::from(self.ty(), v)
  }
  fn shr<T>(&self, bits: T) -> Self
  where
    T: ToTokens,
  {
    let v = quote!(#self >> (#bits as u32));
    SimdValue::from(self.ty(), v)
  }
  fn min(&self, rhs: &Self) -> Self {
    let v = quote!(#self.min(#rhs));
    SimdValue::from(self.ty(), v)
  }
  fn max(&self, rhs: &Self) -> Self {
    let v = quote!(#self.max(#rhs));
    SimdValue::from(self.ty(), v)
  }
  fn butterfly(&self, rhs: &Self) -> (Self, Self) {
    (self + rhs, self - rhs)
  }

  /// Take `self` and extend its width to `to_width` elements, filling
  /// with `v` (a value of the vector primitive).
  fn extend<T>(&self, to_width: usize, v: T) -> SimdValue
  where
    T: ToTokens,
  {
    assert_ne!(to_width, 0);

    let len = self.w();
    if to_width == len {
      return self.clone();
    }

    let rhs = self.ty().splat(v);
    let mut idxs = Vec::with_capacity(to_width);
    for i in 0..to_width {
      if i <= len {
        idxs.push(quote!(#i));
      } else {
        idxs.push(quote!(#len));
      }
    }

    Self::shuffle2(self, &rhs, &idxs)
  }

  /// We can't use shuffle!() in the build script. Instead we must craft
  /// a token stream which includes `shuffle!(..)`
  fn shuffle2<T>(l: &SimdValue, r: &SimdValue, idx: &[T]) -> Self
  where
    T: ToTokens,
  {
    assert_eq!(l.ty(), r.ty());

    let mut idxs = TokenStream::default();
    let idx_len = idx.len();
    for (ii, i) in idx.iter().enumerate() {
      if ii + 1 == idx_len {
        // trailing comma not allowed *facepalm*
        idxs.extend(quote! { #i });
      } else {
        idxs.extend(quote! { #i, });
      }
    }

    let mut ts = macro_("shuffle");
    ts.extend(quote! {
     { #l, #r, [#idxs] }
    });

    let ty = SimdType::new(l.ty().ty(), idx.len());
    SimdValue::from(ty, ts)
  }
  fn shuffle<T>(&self, idx: &[T]) -> Self
  where
    T: ToTokens,
  {
    let mut idxs = TokenStream::default();
    let idx_len = idx.len();
    for (ii, i) in idx.iter().enumerate() {
      if ii + 1 == idx_len {
        // trailing comma not allowed *facepalm*
        idxs.extend(quote! { #i });
      } else {
        idxs.extend(quote! { #i, });
      }
    }

    let mut ts = macro_("shuffle");
    ts.extend(quote! {
     { #self, [#idxs] }
    });

    let ty = SimdType::new(self.ty().ty(), idx.len());
    SimdValue::from(ty, ts)
  }
  fn select_range(&self, range: Range<usize>) -> Self {
    let idxs = range.map(|i| i as u32).collect::<Vec<_>>();
    self.shuffle(&idxs)
  }
  fn concat(&self, rhs: &SimdValue) -> Self {
    let new_len = (self.w() + rhs.w()) as u32;
    let idxs = (0u32..new_len).collect::<Vec<_>>();

    Self::shuffle2(self, rhs, &idxs)
  }
  fn scatter<T, U>(&self, dst: &mut TokenStream, ptr: T, stride: U)
    where T: ToTokens,
          U: ToTokens,
  {
    assert!(self.w() <= 8, "simd too large to scatter :(");
    let ty = self.ty();
    let e = ty.ty();
    let w = self.w();

    let offsets = ty.indices(|idx| quote!(#idx) );
    dst.extend(quote! {{
      let all_mask = <Simd<[m8; #w]>>::splat(true);
      let ptr = <Simd<[*mut #e; #w]>>::splat(#ptr);
      let offsets = #offsets * (#stride as u32);
      let ptr = ptr.add(<Simd<[usize; #w]>>::from_cast(offsets));
      ptr.write(all_mask, #self);
    }});
  }
  fn let_<T>(&self, dest: &mut TokenStream, name: T) -> Self
  where
    T: Display,
  {
    let name = Ident::new(&name.to_string(), Span::call_site());
    let ty = self.ty();
    dest.extend(quote! {
      let #name: #ty = #self;
    });
    SimdValue::from(ty, quote!(#name))
  }
  fn let_mut<T>(&self, dest: &mut TokenStream, name: T) -> Self
  where
    T: Display,
  {
    let name = Ident::new(&name.to_string(), Span::call_site());
    let ty = self.ty();
    dest.extend(quote! {
      let mut #name: #ty = #self;
    });
    SimdValue::from(ty, quote!(#name))
  }

  #[allow(dead_code)]
  fn debug<T>(&self, dst: &mut TokenStream, name: T)
  where
    T: Display,
  {
    let fmt = format!("{}: {{:?}}", name);
    dst.extend(call_macro(
      "println",
      quote! {{
        #fmt, #self,
      }},
    ));
  }
}
impl<'a> Add<&'a SimdValue> for &'a SimdValue {
  type Output = SimdValue;
  fn add(self, rhs: &'a SimdValue) -> SimdValue {
    assert_eq!(self.ty(), rhs.ty());
    let ty = self.ty();
    let v = quote! { (#self + #rhs) };
    SimdValue::from(ty, v)
  }
}
impl<'a> Sub<&'a SimdValue> for &'a SimdValue {
  type Output = SimdValue;
  fn sub(self, rhs: &'a SimdValue) -> SimdValue {
    assert_eq!(self.ty(), rhs.ty());
    let ty = self.ty();
    let v = quote! { (#self - #rhs) };
    SimdValue::from(ty, v)
  }
}
impl<'a> Mul<&'a SimdValue> for &'a SimdValue {
  type Output = SimdValue;
  fn mul(self, rhs: &'a SimdValue) -> SimdValue {
    assert_eq!(self.ty(), rhs.ty());
    let ty = self.ty();
    let v = quote! { (#self * #rhs) };
    SimdValue::from(ty, v)
  }
}
impl<'a> ops::Neg for &'a SimdValue {
  type Output = SimdValue;
  fn neg(self) -> SimdValue {
    assert!(self.ty().is_signed());
    let v = quote!((-#self));
    SimdValue::from(self.ty(), v)
  }
}
impl<'a> Add<&'a SimdValue> for SimdValue {
  type Output = SimdValue;
  fn add(self, rhs: &'a SimdValue) -> SimdValue {
    &self + rhs
  }
}
impl<'a> Sub<&'a SimdValue> for SimdValue {
  type Output = SimdValue;
  fn sub(self, rhs: &'a SimdValue) -> SimdValue {
    &self - rhs
  }
}
impl<'a> Mul<&'a SimdValue> for SimdValue {
  type Output = SimdValue;
  fn mul(self, rhs: &'a SimdValue) -> SimdValue {
    &self * rhs
  }
}
impl Add<SimdValue> for SimdValue {
  type Output = SimdValue;
  fn add(self, rhs: SimdValue) -> SimdValue {
    &self + &rhs
  }
}
impl Sub<SimdValue> for SimdValue {
  type Output = SimdValue;
  fn sub(self, rhs: SimdValue) -> SimdValue {
    &self - &rhs
  }
}
impl Mul<SimdValue> for SimdValue {
  type Output = SimdValue;
  fn mul(self, rhs: SimdValue) -> SimdValue {
    &self * &rhs
  }
}
impl ops::Neg for SimdValue {
  type Output = SimdValue;
  fn neg(self) -> SimdValue {
    -(&self)
  }
}
impl ToTokens for SimdValue {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    self.1.to_tokens(tokens);
  }
}

struct Var<T> {
  name: Ident,
  mutable: bool,
  value: T,
}
impl<T> Var<T>
where
  T: ToTokens,
{
  fn new<U>(name: U, value: T) -> Self
  where
    U: Display,
  {
    let name = Ident::new(&name.to_string(), Span::call_site());
    Self::new_ident(name, false, value)
  }
  fn new_mut<U>(name: U, value: T) -> Self
  where
    U: Display,
  {
    let name = Ident::new(&name.to_string(), Span::call_site());
    Self::new_ident(name, true, value)
  }
  fn new_ident(name: Ident, mutable: bool, value: T) -> Self {
    Var { name, mutable, value }
  }

  fn let_<U>(dest: &mut TokenStream, name: U, value: T) -> Ident
  where
    U: Display,
    T: ToTokens,
  {
    let name = Ident::new(&name.to_string(), Span::call_site());
    dest.extend(quote! {
      let #name = #value;
    });
    name
  }
  fn let_mut<U>(dest: &mut TokenStream, name: U, value: T) -> Ident
  where
    U: Display,
    T: ToTokens,
  {
    let name = Ident::new(&name.to_string(), Span::call_site());
    dest.extend(quote! {
      let mut #name = #value;
    });
    name
  }

  fn add_assign(&self, dst: &mut TokenStream, v: TokenStream) {
    assert!(self.mutable);
    let name = &self.name;
    dst.extend(quote! {
      #name += #v;
    });
  }
  fn assign<U>(&self, dst: &mut TokenStream, v: U)
  where
    U: ToTokens,
  {
    assert!(self.mutable);
    let name = &self.name;
    dst.extend(quote! {
      #name = #v;
    });
  }

  fn name(&self) -> &Ident {
    &self.name
  }
  fn value(&self) -> &T {
    &self.value
  }
}
impl<T> ToTokens for Var<T>
where
  T: ToTokens,
{
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let name = &self.name;
    let value = &self.value;
    if !self.mutable {
      tokens.extend(quote! {
        let #name = #value;
      });
    } else {
      tokens.extend(quote! {
        let mut #name = #value;
      });
    }
  }
}
impl<T> Deref for Var<T>
where
  T: ToTokens,
{
  type Target = Ident;
  fn deref(&self) -> &Ident {
    &self.name
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum IsaFeature {
  Native,

  // x86
  Sse2,
  Sse3,
  Ssse3,
  Sse4_1,
  Sse4_2,
  Avx,
  Avx2,
  Avx512,

  // ARM
  Neon,
}
impl IsaFeature {
  fn fn_suffix(&self) -> &'static str {
    match self {
      IsaFeature::Native => "native",
      IsaFeature::Sse2 => "sse2",
      IsaFeature::Sse3 => "sse3",
      IsaFeature::Ssse3 => "ssse3",
      IsaFeature::Sse4_1 => "sse4_1",
      IsaFeature::Sse4_2 => "sse4_2",
      IsaFeature::Avx => "avx",
      IsaFeature::Avx2 => "avx2",
      IsaFeature::Avx512 => "avx512",
      IsaFeature::Neon => "neon",
    }
  }
  fn target_feature(&self) -> &'static str {
    match self {
      IsaFeature::Native => {
        panic!("IsaFeature::Native isn't a target feature")
      }
      IsaFeature::Sse2 => "sse2",
      IsaFeature::Sse3 => "sse3",
      IsaFeature::Ssse3 => "ssse3",
      IsaFeature::Sse4_1 => "sse4.1",
      IsaFeature::Sse4_2 => "sse4.2",
      IsaFeature::Avx => "avx",
      IsaFeature::Avx2 => "avx2",
      IsaFeature::Avx512 => "avx512",
      IsaFeature::Neon => "neon",
    }
  }

  fn module_name(&self) -> Ident {
    let name = self.fn_suffix();
    let name = format!("i_{}", name);
    Ident::new(&name, Span::call_site())
  }

  fn index(&self) -> usize {
    match self {
      IsaFeature::Native => 0,

      IsaFeature::Sse2 => 1,
      IsaFeature::Ssse3 => 2,
      IsaFeature::Avx2 => 3,

      IsaFeature::Neon => 1,
      _ => panic!("{:?} has no index", self),
    }
  }
  fn simd_bits(&self) -> usize {
    match self {
      IsaFeature::Avx2 => 256,
      IsaFeature::Avx512 => 512,
      _ => 128,
    }
  }
  fn simd_width(&self, ty: PrimType) -> usize {
    self.simd_bits() / ty.bits()
  }

  fn ptr_simd_lanes(&self) -> usize {
    match self {
      IsaFeature::Avx512 => 8,
      _ => 4,
    }
  }
  fn is_native(&self) -> bool {
    if let &IsaFeature::Native = self {
      true
    } else {
      false
    }
  }

  fn sets() -> Vec<IsaFeature> {
    let mut out = vec![IsaFeature::Native];
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    if arch == "x86_64" || arch == "x86" {
      out.push(IsaFeature::Avx2);
    } else if arch.contains("arm") {
      // TODO need to add requisite code to rav1e for ARM
      out.push(IsaFeature::Neon);
    }

    out
  }
}

impl ToTokens for IsaFeature {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    if let IsaFeature::Native = self {
      return;
    }

    let s = self.target_feature();
    tokens.extend(quote! {
      #[target_feature(enable = #s)]
    });
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NegIdx {
  Neg(usize),
  Pos(usize),
}
impl NegIdx {
  fn i(&self) -> usize {
    match self {
      Neg(i) | Pos(i) => *i,
    }
  }
  fn is_neg(&self) -> bool {
    match self {
      Neg(_) => true,
      _ => false,
    }
  }
}
impl From<usize> for NegIdx {
  fn from(v: usize) -> NegIdx {
    Pos(v)
  }
}
impl ops::Neg for NegIdx {
  type Output = NegIdx;
  fn neg(self) -> NegIdx {
    match self {
      Pos(i) => Neg(i),
      Neg(i) => Pos(i),
    }
  }
}

trait Array {
  fn ty(&self) -> SimdType;
  fn len(&self) -> usize;
  fn get(&self, idx: usize) -> SimdValue;
  fn get_neg(&self, idx: NegIdx) -> SimdValue {
    let v = self.get(idx.i());
    if idx.is_neg() {
      v.neg()
    } else {
      v
    }
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue);

  fn iter(&self) -> ArrayIter<&Self>
    where Self: Sized,
  {
    let len = self.len();
    ArrayIter {
      r: 0..len,
      a: self,
    }
  }

  fn map<F>(self, f: F) -> ArrayMap<Self, F>
  where
    Self: Sized,
    F: Fn(usize, SimdValue) -> SimdValue,
  {
    let ty = if self.len() > 0 {
      // meh.
      f(0, self.get(0)).ty()
    } else {
      // in this case the type won't matter
      self.ty()
    };
    ArrayMap { ty, array: self, map: f }
  }
  fn trunc(self, len: usize) -> ArrayTrunc<Self>
  where
    Self: Sized,
  {
    assert!(self.len() >= len);
    ArrayTrunc { len, array: self }
  }

  fn assign(&self, dst: &mut TokenStream, from: &dyn Array) {
    assert_eq!(self.ty(), from.ty());
    assert_eq!(self.len(), from.len());

    for i in 0..self.len() {
      let v = from.get(i);
      self.set(dst, i, v);
    }
  }

  #[allow(dead_code)]
  fn debug(&self, dst: &mut TokenStream, name: &dyn Display) {
    for i in 0..self.len() {
      let item = self.get(i);
      let fmt = format!("{}[{}]: {{:?}}", name, i);
      dst.extend(call_macro(
        "println",
        quote! {{
          #fmt, #item,
        }},
      ));
    }
  }
  #[allow(dead_code)]
  fn debug_transpose(&self, dst: &mut TokenStream, name: &dyn Display) {
    for r in 0..self.ty().w() {
      let mut args = TokenStream::default();
      let mut fmt = format!("{}[{}]: [", name, r);

      for i in 0..self.len() {
        let item = self.get(i);

        args.extend(quote! {
          #item.extract_unchecked(#r),
        });
        fmt.push_str("{:?}");
        if i != self.len() - 1 {
          fmt.push_str(", ");
        }
      }
      fmt.push_str("]");
      dst.extend(call_macro(
        "println",
        quote! {{
          #fmt, #args
        }},
      ));
    }
  }
}
impl<'a, T> Array for &'a T
where
  T: Array + ?Sized,
{
  fn ty(&self) -> SimdType {
    (&**self).ty()
  }
  fn len(&self) -> usize {
    (&**self).len()
  }
  fn get(&self, idx: usize) -> SimdValue {
    (&**self).get(idx)
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    (&**self).set(dst, idx, v)
  }
}
impl Array for Box<dyn Array> {
  fn ty(&self) -> SimdType {
    (&**self).ty()
  }
  fn len(&self) -> usize {
    (&**self).len()
  }
  fn get(&self, idx: usize) -> SimdValue {
    (&**self).get(idx)
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    (&**self).set(dst, idx, v)
  }
}
struct ArrayIter<T>
  where T: Array,
{
  r: Range<usize>,
  a: T,
}
impl<T> Iterator for ArrayIter<T>
  where T: Array,
{
  type Item = SimdValue;
  fn next(&mut self) -> Option<SimdValue> {
    if let Some(i) = self.r.next() {
      Some(self.a.get(i))
    } else {
      None
    }
  }
  fn size_hint(&self) -> (usize, Option<usize>) {
    self.r.size_hint()
  }
}
impl<T> DoubleEndedIterator for ArrayIter<T>
  where T: Array,
{
  fn next_back(&mut self) -> Option<SimdValue> {
    if let Some(i) = self.r.next_back() {
      Some(self.a.get(i))
    } else {
      None
    }
  }
}

struct Slice<T = Ident>(SimdType, T, usize)
where
  T: ToTokens;
impl<T> Slice<T>
where
  T: ToTokens,
{
  fn from_ptr(ty: SimdType, name: T, len: usize) -> Self {
    Slice(ty, name, len)
  }
}
impl<T> Array for Slice<T>
where
  T: ToTokens,
{
  fn ty(&self) -> SimdType {
    self.0
  }
  fn len(&self) -> usize {
    self.2
  }
  fn get(&self, idx: usize) -> SimdValue {
    assert!(self.len() > idx);
    let ty = self.ty();
    SimdValue::from(self.0, quote! { (*(#self as *const #ty).add(#idx)) })
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    assert!(self.len() > idx);
    assert_eq!(self.0, v.ty());
    let ty = self.ty();
    dst.extend(quote! {
      *(#self as *mut #ty).add(#idx) = #v;
    });
  }
}
impl<T> ToTokens for Slice<T>
where
  T: ToTokens,
{
  fn to_tokens(&self, to: &mut TokenStream) {
    self.1.to_tokens(to);
  }
}

struct Vector(SimdType, Vec<SimdValue>);
impl Vector {
  fn new(ty: SimdType, v: Vec<SimdValue>) -> Self {
    Vector(ty, v)
  }
}
impl Array for Vector {
  fn ty(&self) -> SimdType {
    self.0
  }
  fn len(&self) -> usize {
    self.1.len()
  }
  fn get(&self, idx: usize) -> SimdValue {
    self.1[idx].clone()
  }
  fn set(&self, _dst: &mut TokenStream, _idx: usize, _v: SimdValue) {
    panic!("Vector is read only");
  }
}

/// An "array" of variables. Really just a indexed set of variables.
#[derive(Clone)]
struct VarArr {
  ty: SimdType,
  prefix: String,
  names: Vec<TokenStream>,
}
impl VarArr {
  fn let_<T>(dst: &mut TokenStream, prefix: T, slice: &dyn Array) -> Self
  where
    T: Display,
  {
    let names = (0..slice.len())
      .map(|i| {
        slice.get(i).let_(dst, format_args!("{}_{}", prefix, i)).unwrap_value()
      })
      .collect();
    VarArr { ty: slice.ty(), prefix: prefix.to_string(), names }
  }
  fn let_mut<T>(dst: &mut TokenStream, prefix: T, slice: &dyn Array) -> Self
  where
    T: Display,
  {
    let names = (0..slice.len())
      .map(|i| {
        slice
          .get(i)
          .let_mut(dst, format_args!("{}_{}", prefix, i))
          .unwrap_value()
      })
      .collect();
    VarArr { ty: slice.ty(), prefix: prefix.to_string(), names }
  }
}
impl Array for VarArr {
  fn ty(&self) -> SimdType {
    self.ty
  }
  fn len(&self) -> usize {
    self.names.len()
  }
  fn get(&self, idx: usize) -> SimdValue {
    assert!(self.len() > idx);
    SimdValue::from(self.ty(), &self.names[idx])
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    assert!(self.len() > idx);
    assert_eq!(self.ty(), v.ty());
    let name = &self.names[idx];
    dst.extend(quote! {
      #name = #v;
    });
  }
}
struct ArrayTrunc<T>
where
  T: Array,
{
  len: usize,
  array: T,
}
impl<T> Array for ArrayTrunc<T>
where
  T: Array,
{
  fn ty(&self) -> SimdType {
    self.array.ty()
  }
  fn len(&self) -> usize {
    self.len
  }
  fn get(&self, idx: usize) -> SimdValue {
    assert!(self.len() > idx);
    self.array.get(idx)
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    assert!(self.len() > idx);
    self.array.set(dst, idx, v);
  }
}

struct ArrayMap<T, F>
where
  T: Array,
  F: Fn(usize, SimdValue) -> SimdValue,
{
  ty: SimdType,
  array: T,
  map: F,
}
impl<T, F> ArrayMap<T, F>
where
  T: Array,
  F: Fn(usize, SimdValue) -> SimdValue,
{
}
impl<T, F> Array for ArrayMap<T, F>
where
  T: Array,
  F: Fn(usize, SimdValue) -> SimdValue,
{
  fn ty(&self) -> SimdType {
    self.ty
  }
  fn len(&self) -> usize {
    self.array.len()
  }
  fn get(&self, idx: usize) -> SimdValue {
    assert!(self.len() > idx);
    (&self.map)(idx, self.array.get(idx))
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    // Yes, this is a bit weird, but it is useful for modifying
    // the value to set.
    assert!(self.len() > idx);
    let v = (&self.map)(idx, v);
    self.array.set(dst, idx, v);
  }
}

struct Module {
  parents: Vec<(Ident, bool)>,
  filename: Option<PathBuf>,
  name: Ident,
  tt: TokenStream,
}
impl Module {
  fn new_root(from_crate_root: &[&str], name: Ident) -> Self {
    let mut parents = Vec::with_capacity(from_crate_root.len() + 1);
    for &segment in from_crate_root {
      parents.push((Ident::new(segment, Span::call_site()), true));
    }
    Module {
      parents,
      filename: None,
      name,
      tt: Default::default(),
    }
  }
  fn write_no_edit_comment(out: &mut dyn Write) {
    writeln!(out, "// Programmatically generated; all changes will be overwritten.")
      .unwrap();
  }
  fn finish_root(self, out: &mut dyn Write) {
    assert!(self.parents.iter().all(|&(_, from_crate)| from_crate),
            "don't use this for children");
    Self::write_no_edit_comment(out);
    let this = quote!(#self);
    writeln!(out, "{}", this).expect("failed to write module partition");
  }

  fn new_child(&self, name: Ident) -> Module {
    let mut parents = self.parents.clone();
    parents.push((self.name.clone(), false));
    Module { parents, filename: None, name, tt: Default::default() }
  }
  fn new_child_file(&self, file_prefix: &str, name: Ident) -> Module {
    let mut path = file_prefix.to_owned();
    for &(ref parent, from_crate) in self.parents.iter() {
      if from_crate { continue; }
      path.push_str(&format!("_{}", parent));
    }
    path.push_str(&format!("_{}_{}_kernels.rs", self.name, name));

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);

    let mut parents = self.parents.clone();
    parents.push((self.name.clone(), false));
    Module {
      parents,
      filename: Some(out_dir.join(path)),
      name,
      tt: Default::default(),
    }
  }

  fn new_func<T>(&self, name: T, args: TokenStream,
                 px: PixelType, isa: IsaFeature)
    -> Func
    where T: Display,
  {
    let name = format!("{}_{}_{}", name, px.type_str(), isa.fn_suffix());
    let name = Ident::new(&name, Span::call_site());

    let mut path = vec![Ident::new("crate", Span::call_site())];
    for &(ref parent, _) in self.parents.iter() {
      path.push(parent.clone());
    }
    path.push(self.name.clone());
    path.push(name.clone());

    Func {
      public: true,
      name,
      args,
      ts_path: Func::build_path(&path),
      path,
      px,
      isa,
      inline: None,
      attributes: Default::default(),
      body: TokenStream::default(),
      params: Default::default(),
    }
  }
  fn new_priv_func<T>(&self, name: T, args: TokenStream,
                      px: PixelType, isa: IsaFeature)
    -> Func
    where T: Display,
  {
    let name = format!("{}_{}_{}", name, px.type_str(), isa.fn_suffix());
    let name = Ident::new(&name, Span::call_site());

    let mut path = vec![Ident::new("crate", Span::call_site())];
    for &(ref parent, _) in self.parents.iter() {
      path.push(parent.clone());
    }
    path.push(self.name.clone());
    path.push(name.clone());

    Func {
      public: false,
      name,
      args,
      ts_path: Func::build_path(&path),
      path,
      px,
      isa,
      inline: None,
      attributes: Default::default(),
      body: TokenStream::default(),
      params: Default::default(),
    }
  }

  fn item_path(&self, item: &Ident) -> TokenStream {
    let mut t = quote!(crate);
    for &(ref parent, _) in self.parents.iter() {
      t = quote!(#t::#parent);
    }

    let this = &self.name;
    quote!(#t::#this::#item)
  }
  fn finish_child(mut self, parent: &mut Module) {
    assert!(self.parents.iter().any(|(_, from_crate)| !from_crate),
            "don't use this for the root");
    assert_eq!(&self.parents.last().unwrap().0, &parent.name, "parent mismatch");

    parent.tt.extend(quote!(#self));

    // write to the target path:
    if let Some(path) = self.filename.take() {
      gen::write_kernel(path, |f| {
        Self::write_no_edit_comment(f);
        writeln!(f, "{}", self.tt).expect("write kernel submodule file");
      });
    }
  }
}

impl Deref for Module {
  type Target = TokenStream;
  fn deref(&self) -> &TokenStream {
    &self.tt
  }
}
impl DerefMut for Module {
  fn deref_mut(&mut self) -> &mut TokenStream {
    &mut self.tt
  }
}
impl ToTokens for Module {
  fn to_tokens(&self, to: &mut TokenStream) {
    let name = &self.name;
    let inner = &self.tt;
    if let Some(ref fp) = self.filename {
      let path = format!("{}", fp.display());
      to.extend(quote! {
        #[path = #path]
        pub mod #name;
      });
    } else {
      to.extend(quote! {
        pub mod #name {
          #inner
        }
      });
    }
  }
}

#[derive(Clone, Debug)]
struct KernelDispatch {
  fn_args: Vec<(TokenStream, TokenStream)>,
  fn_ret: Option<TokenStream>,
  fn_ty: (Ident, TokenStream),
  kname: String,
  u8_ks: Vec<(Rc<TokenStream>, Rc<TokenStream>)>,
  u16_ks: Vec<(Rc<TokenStream>, Rc<TokenStream>)>,
  /// `BLOCK_SIZES_ALL` and `CpuFeatureLevel::len()` are implicit.
  table_array_sizes: Vec<TokenStream>,
}
impl KernelDispatch {
  fn new(
    fn_ty_name: &str, fn_args: Vec<(TokenStream, TokenStream)>,
    fn_ret: Option<TokenStream>, kname: &str,
    table_array_sizes: Vec<TokenStream>,
  ) -> Self {
    let fn_ty_name = Ident::new(fn_ty_name, Span::call_site());

    let mut fn_ty_args = TokenStream::default();
    for (arg, ty) in fn_args.iter() {
      fn_ty_args.extend(quote! {
        #arg: #ty,
      });
    }
    let fn_ret2 = fn_ret.as_ref()
      .map(|ret| quote!(-> #ret ) );
    let fn_ty = quote!(unsafe fn(#fn_ty_args) #fn_ret2 );

    KernelDispatch {
      fn_args,
      fn_ret,
      fn_ty: (fn_ty_name, fn_ty),
      kname: kname.into(),
      u8_ks: Vec::new(),
      u16_ks: Vec::new(),
      table_array_sizes,
    }
  }
  fn push_kernel<T, U>(
    &mut self, px: PixelType, idx: T, path: U,
  )
    where T: Into<Rc<TokenStream>>,
          U: Into<Rc<TokenStream>>,
  {
    let ks = match px {
      PixelType::U8 => &mut self.u8_ks,
      PixelType::U16 => &mut self.u16_ks,
    };
    ks.push((idx.into(), path.into()));
  }

  fn table_ty(&self, px: PixelType) -> TokenStream {
    let fn_ty_name = &self.fn_ty.0;
    let mut table_ty = quote!(Option<#fn_ty_name<#px>>);
    for size in self.table_array_sizes.iter() {
      table_ty = quote!([#table_ty; #size]);
    }

    let blen = Block::table_len();

    quote!([[#table_ty; #blen]; CpuFeatureLevel::len()])
  }
  fn table_default(&self) -> TokenStream {
    let mut table = quote!(None);
    for size in self.table_array_sizes.iter() {
      table = quote!([#table; #size]);
    }
    let blen = Block::table_len();

    quote!([[#table; #blen]; CpuFeatureLevel::len()])
  }
  fn table_init(ks: &[(Rc<TokenStream>, Rc<TokenStream>)]) -> TokenStream {
    let mut table_init = TokenStream::default();
    for &(ref idx, ref path) in ks.iter() {
      table_init.extend(quote! {
        out #idx = Some(#path as _);
      });
    }

    table_init
  }

  fn tables(self) -> TokenStream {
    let fn_ty_name = &self.fn_ty.0;
    let fn_ty = &self.fn_ty.1;

    let u8_table_name = format!("U8_{}_KERNELS", self.kname);
    let u8_table_name = Ident::new(&u8_table_name, Span::call_site());
    let u16_table_name = format!("U16_{}_KERNELS", self.kname);
    let u16_table_name = Ident::new(&u16_table_name, Span::call_site());

    let u8_table_ty = self.table_ty(PixelType::U8);
    let u16_table_ty = self.table_ty(PixelType::U16);

    let u8_table_init = Self::table_init(&self.u8_ks);
    let u16_table_init = Self::table_init(&self.u16_ks);

    let table_default = self.table_default();

    let mut dispatch_args = TokenStream::default();
    let mut dispatch_call = TokenStream::default();
    for (arg, ty) in self.fn_args.iter() {
      dispatch_args.extend(quote!(#arg: #ty, ));
      dispatch_call.extend(quote!(#arg as _, ));
    }
    let mut dispatch_idx = quote!([cpu.as_index()].get(block_idx)?);
    for (i, _) in self.table_array_sizes.iter().enumerate() {
      let name = format!("__idx{}", i);
      let name = Ident::new(&name, Span::call_site());
      dispatch_args.extend(quote!(#name: usize, ));
      dispatch_idx.extend(quote!(.get(#name)?));
    }

    let dispatch_ret = self.fn_ret
      .as_ref()
      .map(|fn_ret| quote!(Option<#fn_ret>) )
      .unwrap_or_else(|| quote!(Option<()>) );

    quote! {
      type #fn_ty_name<T> = #fn_ty;

      static #u8_table_name: #u8_table_ty = {
        let mut out: #u8_table_ty = #table_default;
        #u8_table_init
        out
      };
      static #u16_table_name: #u16_table_ty = {
        let mut out: #u16_table_ty = #table_default;
        #u16_table_init
        out
      };

      #[inline]
      pub unsafe fn dispatch<T, B>(#dispatch_args
                                   block: B,
                                   cpu: CpuFeatureLevel)
        -> #dispatch_ret
        where T: Pixel,
              B: Into<(u16, u16)>,
      {
        use std::mem::size_of;

        let block = block.into();

        assert!(block.0 <= 128);
        assert!(block.1 <= 128);
        debug_assert_eq!(block.0.count_ones(), 1);
        debug_assert_eq!(block.1.count_ones(), 1);

        let block_wi = size_of::<u16>() * 8 - (block.0.leading_zeros() as usize);
        let block_hi = size_of::<u16>() * 8 - (block.1.leading_zeros() as usize);
        let block_idx = block_wi << 4 | block_hi;

        match T::type_enum() {
          PixelType::U8 => {
            let &kernel = #u8_table_name #dispatch_idx;
            let kernel = kernel?;
            return Some(kernel(#dispatch_call));
          },
          PixelType::U16 => {
            let &kernel = #u16_table_name #dispatch_idx;
            let kernel = kernel?;
            return Some(kernel(#dispatch_call));
          },
        }
      }
    }
  }

  fn len(&self) -> usize {
    self.u8_ks.len() + self.u16_ks.len()
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Inline {
  Never,
  Hint,
  Always,
}
impl ToTokens for Inline {
  fn to_tokens(&self, to: &mut TokenStream) {
    let attr = match self {
      Inline::Never => quote! {
        #![inline(never)]
      },
      Inline::Hint => quote! {
        #![inline]
      },
      Inline::Always => quote! {
        #![inline(always)]
      },
    };
    to.extend(attr);
  }
}
#[derive(Clone)]
pub struct Func {
  public: bool,
  name: Ident,
  args: TokenStream,
  path: Vec<Ident>,
  ts_path: Rc<TokenStream>,
  inline: Option<Inline>,
  px: PixelType,
  isa: IsaFeature,
  attributes: TokenStream,
  params: Vec<(TokenStream, TokenStream)>,
  body: TokenStream,
}

impl Func {
  pub fn inline_always(&mut self) {
    self.inline = Some(Inline::Always);
  }
  pub fn inline_never(&mut self) {
    self.inline = Some(Inline::Never);
  }
  pub fn inline_hint(&mut self) {
    self.inline = Some(Inline::Hint);
  }
  pub fn clear_inline(&mut self) {
    self.inline = None;
  }

  pub fn name(&self) -> Ident { self.name.clone() }

  pub fn path(&self) -> Rc<TokenStream> { self.ts_path.clone() }

  fn build_path(path: &[Ident]) -> Rc<TokenStream> {
    let mut t = TokenStream::default();
    for (i, parent) in path.iter().enumerate() {
      if i == 0 {
        t = quote!(#parent);
      } else {
        t = quote!(#t::#parent);
      }
    }
    Rc::new(t)
  }

  pub fn add_attr<T>(&mut self, attr: T)
    where T: ToTokens,
  {
    attr.to_tokens(&mut self.attributes);
  }
}
impl Deref for Func {
  type Target = TokenStream;
  fn deref(&self) -> &TokenStream {
    &self.body
  }
}
impl DerefMut for Func {
  fn deref_mut(&mut self) -> &mut TokenStream {
    &mut self.body
  }
}
impl ToTokens for Func {
  fn to_tokens(&self, to: &mut TokenStream) {
    let name = &self.name;
    let args = &self.args;
    let attrs = &self.attributes;
    let inner = &self.body;

    let inline = if let Some(inline) = self.inline {
      inline.into_token_stream()
    } else {
      TokenStream::default()
    };

    let public = if self.public {
      quote!(pub(crate))
    } else {
      quote!()
    };

    let mut params: Option<TokenStream> = None;
    let mut where_clauses: Option<TokenStream> = None;
    for (param, clause) in self.params.iter() {
      if let Some(ref mut prev_params) = params {
        assert!(where_clauses.is_some());
        let prev_where_clauses = where_clauses.as_mut().unwrap();

        prev_params.extend(quote!(,));
        prev_params.extend(param.clone().into_iter());
        prev_where_clauses.extend(clause.clone().into_iter());
        prev_where_clauses.extend(quote!(,));
      } else {
        assert!(where_clauses.is_none());
        params = Some(param.clone());
        where_clauses = Some(quote! {
          where #param: #clause,
        });
      }
    }

    let params = params
      .map(|p| quote!(<#p>) )
      .unwrap_or_default();
    let where_clauses = where_clauses.unwrap_or_default();

    let isa = match self.inline {
      None | Some(Inline::Never) => self.isa,
      _ => {
        // inlining not allowed with #[target_feature]
        IsaFeature::Native
      },
    };

    to.extend(quote! {
      #attrs #isa
      #public unsafe fn #name #params #args #where_clauses {
        #![allow(unused_variables)]
        #![allow(unused_mut)]
        #![allow(unused_assignments)]
        #![allow(unused_parens)]
        #inline

        #inner
      }
    })
  }
}

struct TableEntry {
  indices: Rc<TokenStream>,
  name: Ident,
  path: Rc<TokenStream>,
  /// If None, then this specific kernel wasn't generated.
  func: Option<Func>,
}
impl TableEntry {
  fn idx(&self) -> Rc<TokenStream> { self.indices.clone() }
  fn name(&self) -> Ident { self.name.clone() }
  fn path(&self) -> Rc<TokenStream> { self.path.clone() }
}
impl Clone for TableEntry {
  fn clone(&self) -> Self {
    let TableEntry {
      indices,
      path,
      name,
      ..
    } = self;
    TableEntry {
      indices: indices.clone(),
      name: name.clone(),
      path: path.clone(),
      func: None,
    }
  }
}

struct Ctx<'a, F, N = &'a mut BTreeMap<PixelType, F>, K = ()> {
  out: &'a mut Module,
  isa: IsaFeature,
  px: PixelType,
  key: K,

  native: N,

  funcs: F,
}
impl<'a, F, N, K> Ctx<'a, F, N, K> {
  fn new_func<T>(&self, name: T, args: TokenStream,
                 params: Vec<(TokenStream, TokenStream)>,
                 public: bool)
    -> Func
    where T: Display,
  {
    let name = format!("{}_{}_{}", name, self.px.type_str(), self.isa.fn_suffix());
    let name = Ident::new(&name, Span::call_site());

    let mut path = vec![Ident::new("crate", Span::call_site())];
    for &(ref parent, _) in self.out.parents.iter() {
      path.push(parent.clone());
    }
    path.push(self.out.name.clone());
    path.push(name.clone());

    let mut f = Func {
      public,
      name,
      args,
      ts_path: Func::build_path(&path),
      path,
      px: self.px,
      isa: self.isa,
      inline: None,
      attributes: Default::default(),
      body: TokenStream::default(),
      params,
    };

    if self.isa < IsaFeature::Avx2 {
      f.inline_never();
    }

    f
  }

  fn simd_bits(&self) -> usize {
    512 // always use max width available in packed_simd
  }
  fn simd_width(&self, ty: PrimType) -> usize {
    self.simd_bits() / ty.bits()
  }
  fn px_simd_width(&self) -> usize {
    self.simd_width(self.px.into())
  }
  fn ptr_simd_width(&self) -> usize {
    self.isa.ptr_simd_lanes()
  }
  fn hot_block(&self, b: Block) -> bool {
    self.isa != IsaFeature::Native &&
      b.rect_log_ratio() <= 1 &&
      b.w() >= 8 && b.w() < 128 &&
      b.h() >= 8 && b.h() < 128
  }

  fn block_size_check(func: &mut TokenStream, b: Block) {
    let w = b.w();
    let h = b.h();
    func.extend(quote! {
      debug_assert_eq!(width as usize, #w);
      debug_assert_eq!(height as usize, #h);
    });
  }
}
impl<'a, F, K> Ctx<'a, F, &'a mut BTreeMap<PixelType, F>, K> {
  fn finish(self) {
    let Ctx {
      isa,
      px,
      native,
      funcs,
      ..
    } = self;

    if isa == IsaFeature::Native {
      assert!(native.insert(px, funcs).is_none());
    }
  }
}
impl<'a, F, N, K> Deref for Ctx<'a, F, N, K> {
  type Target = K;
  fn deref(&self) -> &Self::Target { &self.key }
}
impl<'a, F, N, K> DerefMut for Ctx<'a, F, N, K> {
  fn deref_mut(&mut self) -> &mut Self::Target { &mut self.key }
}
