// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_camel_case_types)]

use std::convert::*;

use crate::context::*;
use crate::partition::*;

pub use self::TxSize::*;

pub mod inverse;

pub static SQRT2_BITS: usize = 12;
pub static SQRT2: i32 = 5793; // 2^12 * sqrt(2)
pub static INV_SQRT2: i32 = 2896; // 2^12 / sqrt(2)

#[derive(Debug, Clone, Copy)]
pub enum TxType1D {
  DCT = 1,
  ADST = 2,
  FLIPADST = 3,
  IDTX = 0,
}
impl TxType1D {
  pub fn is_flip(&self) -> bool {
    match self {
      TxType1D::FLIPADST => true,
      _ => false,
    }
  }
  pub fn tbl_idx(self) -> usize {
    self as usize
  }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord)]
#[repr(C)]
pub enum TxType {
  DCT_DCT = 0,   // DCT  in both horizontal and vertical
  ADST_DCT = 1,  // ADST in vertical, DCT in horizontal
  DCT_ADST = 2,  // DCT  in vertical, ADST in horizontal
  ADST_ADST = 3, // ADST in both directions
  FLIPADST_DCT = 4,
  DCT_FLIPADST = 5,
  FLIPADST_FLIPADST = 6,
  ADST_FLIPADST = 7,
  FLIPADST_ADST = 8,
  IDTX = 9,
  V_DCT = 10,
  H_DCT = 11,
  V_ADST = 12,
  H_ADST = 13,
  V_FLIPADST = 14,
  H_FLIPADST = 15,
}
impl TxType {
  pub fn row_tx(&self) -> TxType1D {
    use TxType::*;
    match self {
      DCT_DCT | ADST_DCT | FLIPADST_DCT | H_DCT => TxType1D::DCT,
      DCT_ADST | ADST_ADST | FLIPADST_ADST | H_ADST => TxType1D::ADST,
      DCT_FLIPADST | FLIPADST_FLIPADST | ADST_FLIPADST | H_FLIPADST => {
        TxType1D::FLIPADST
      },
      IDTX | V_DCT | V_ADST | V_FLIPADST => TxType1D::IDTX,
    }
  }
  pub fn col_tx(&self) -> TxType1D {
    use TxType::*;
    match self {
      DCT_DCT | DCT_ADST | DCT_FLIPADST | V_DCT => TxType1D::DCT,
      ADST_DCT | ADST_ADST | ADST_FLIPADST | V_ADST => TxType1D::ADST,
      FLIPADST_DCT | FLIPADST_ADST | FLIPADST_FLIPADST | V_FLIPADST => {
        TxType1D::FLIPADST
      },
      IDTX | H_DCT | H_ADST | H_FLIPADST => TxType1D::IDTX,
    }
  }
}

pub mod tx_1d_types {
  pub trait Detail {
    const FLIPPED: bool;
    const TBL_IDX: usize;
  }

  macro_rules! d {
    ($($name:ident,)*) => {
      $(
        #[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
        pub struct $name;
      )*
    };
  }

  d!(Dct, Adst, FlipAdst, Id,);

  impl Detail for Dct {
    const FLIPPED: bool = false;
    const TBL_IDX: usize = 1;
  }
  impl Detail for Adst {
    const FLIPPED: bool = false;
    const TBL_IDX: usize = 2;
  }
  impl Detail for FlipAdst {
    const FLIPPED: bool = true;
    const TBL_IDX: usize = 3;
  }
  impl Detail for Id {
    const FLIPPED: bool = false;
    const TBL_IDX: usize = 0;
  }
}

/// Transform Size
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
#[repr(C)]
pub enum TxSize {
  TX_4X4,
  TX_8X8,
  TX_16X16,
  TX_32X32,
  TX_64X64,

  TX_4X8,
  TX_8X4,
  TX_8X16,
  TX_16X8,
  TX_16X32,
  TX_32X16,
  TX_32X64,
  TX_64X32,

  TX_4X16,
  TX_16X4,
  TX_8X32,
  TX_32X8,
  TX_16X64,
  TX_64X16,
}

impl TxSize {
  /// Number of square transform sizes
  pub const TX_SIZES: usize = 5;

  /// Number of transform sizes (including non-square sizes)
  pub const TX_SIZES_ALL: usize = 14 + 5;

  pub fn width(self) -> usize {
    1 << self.width_log2()
  }

  pub fn width_log2(self) -> usize {
    match self {
      TX_4X4 | TX_4X8 | TX_4X16 => 2,
      TX_8X8 | TX_8X4 | TX_8X16 | TX_8X32 => 3,
      TX_16X16 | TX_16X8 | TX_16X32 | TX_16X4 | TX_16X64 => 4,
      TX_32X32 | TX_32X16 | TX_32X64 | TX_32X8 => 5,
      TX_64X64 | TX_64X32 | TX_64X16 => 6,
    }
  }

  pub fn width_index(self) -> usize {
    self.width_log2() - TX_4X4.width_log2()
  }

  pub fn height(self) -> usize {
    1 << self.height_log2()
  }

  pub fn height_log2(self) -> usize {
    match self {
      TX_4X4 | TX_8X4 | TX_16X4 => 2,
      TX_8X8 | TX_4X8 | TX_16X8 | TX_32X8 => 3,
      TX_16X16 | TX_8X16 | TX_32X16 | TX_4X16 | TX_64X16 => 4,
      TX_32X32 | TX_16X32 | TX_64X32 | TX_8X32 => 5,
      TX_64X64 | TX_32X64 | TX_16X64 => 6,
    }
  }

  pub fn height_index(self) -> usize {
    self.height_log2() - TX_4X4.height_log2()
  }

  pub fn width_mi(self) -> usize {
    self.width() >> MI_SIZE_LOG2
  }

  pub fn area(self) -> usize {
    1 << self.area_log2()
  }

  pub fn area_log2(self) -> usize {
    self.width_log2() + self.height_log2()
  }

  pub fn height_mi(self) -> usize {
    self.height() >> MI_SIZE_LOG2
  }

  pub fn block_size(self) -> BlockSize {
    match self {
      TX_4X4 => BLOCK_4X4,
      TX_8X8 => BLOCK_8X8,
      TX_16X16 => BLOCK_16X16,
      TX_32X32 => BLOCK_32X32,
      TX_64X64 => BLOCK_64X64,
      TX_4X8 => BLOCK_4X8,
      TX_8X4 => BLOCK_8X4,
      TX_8X16 => BLOCK_8X16,
      TX_16X8 => BLOCK_16X8,
      TX_16X32 => BLOCK_16X32,
      TX_32X16 => BLOCK_32X16,
      TX_32X64 => BLOCK_32X64,
      TX_64X32 => BLOCK_64X32,
      TX_4X16 => BLOCK_4X16,
      TX_16X4 => BLOCK_16X4,
      TX_8X32 => BLOCK_8X32,
      TX_32X8 => BLOCK_32X8,
      TX_16X64 => BLOCK_16X64,
      TX_64X16 => BLOCK_64X16,
    }
  }

  pub fn sqr(self) -> TxSize {
    match self {
      TX_4X4 | TX_4X8 | TX_8X4 | TX_4X16 | TX_16X4 => TX_4X4,
      TX_8X8 | TX_8X16 | TX_16X8 | TX_8X32 | TX_32X8 => TX_8X8,
      TX_16X16 | TX_16X32 | TX_32X16 | TX_16X64 | TX_64X16 => TX_16X16,
      TX_32X32 | TX_32X64 | TX_64X32 => TX_32X32,
      TX_64X64 => TX_64X64,
    }
  }

  pub fn sqr_up(self) -> TxSize {
    match self {
      TX_4X4 => TX_4X4,
      TX_8X8 | TX_4X8 | TX_8X4 => TX_8X8,
      TX_16X16 | TX_8X16 | TX_16X8 | TX_4X16 | TX_16X4 => TX_16X16,
      TX_32X32 | TX_16X32 | TX_32X16 | TX_8X32 | TX_32X8 => TX_32X32,
      TX_64X64 | TX_32X64 | TX_64X32 | TX_16X64 | TX_64X16 => TX_64X64,
    }
  }

  pub fn by_dims(w: usize, h: usize) -> TxSize {
    match (w, h) {
      (4, 4) => TX_4X4,
      (8, 8) => TX_8X8,
      (16, 16) => TX_16X16,
      (32, 32) => TX_32X32,
      (64, 64) => TX_64X64,
      (4, 8) => TX_4X8,
      (8, 4) => TX_8X4,
      (8, 16) => TX_8X16,
      (16, 8) => TX_16X8,
      (16, 32) => TX_16X32,
      (32, 16) => TX_32X16,
      (32, 64) => TX_32X64,
      (64, 32) => TX_64X32,
      (4, 16) => TX_4X16,
      (16, 4) => TX_16X4,
      (8, 32) => TX_8X32,
      (32, 8) => TX_32X8,
      (16, 64) => TX_16X64,
      (64, 16) => TX_64X16,
      _ => unreachable!(),
    }
  }

  pub fn is_rect(self) -> bool {
    self.width_log2() != self.height_log2()
  }
}
impl Into<(u16, u16)> for TxSize {
  fn into(self) -> (u16, u16) {
    (self.width().try_into().unwrap(),
     self.height().try_into().unwrap())
  }
}

pub mod tx_sizes {
  use crate::context::MI_SIZE_LOG2;
  use crate::predict::Dim;
  use crate::transform::TxSize;
  use crate::util::*;

  pub trait Detail: Dim + Block {
    const TX_SIZE: TxSize;

    const WIDTH_MI: usize = Self::WIDTH >> MI_SIZE_LOG2;
    const WIDTH_INDEX: usize =
      Self::WIDTH_LOG2 - <Block4x4 as Block>::WIDTH_LOG2;
    const HEIGHT_MI: usize = Self::HEIGHT >> MI_SIZE_LOG2;
    const HEIGHT_INDEX: usize =
      Self::HEIGHT_LOG2 - <Block4x4 as Block>::HEIGHT_LOG2;

    fn width_index(&self) -> usize {
      Self::WIDTH_INDEX
    }
    fn width_mi(&self) -> usize {
      Self::WIDTH_MI
    }
    fn height_index(&self) -> usize {
      Self::HEIGHT_INDEX
    }
    fn height_mi(&self) -> usize {
      Self::HEIGHT_MI
    }
  }

  // manually impl because by default `*_INDEX` depends on this.
  impl Detail for Block4x4 {
    const TX_SIZE: TxSize = TxSize::TX_4X4;

    const WIDTH_INDEX: usize = 0;
    const HEIGHT_INDEX: usize = 0;
  }

  macro_rules! block_detail {
    ($W:expr, $H:expr) => {
      paste::item! {
        impl Detail for [<Block $W x $H>] {
          const TX_SIZE: TxSize = TxSize::[<TX_ $W X $H>];
        }
      }
    };
  }

  macro_rules! blocks_detail {
    ($(($W:expr, $H:expr)),+) => {
      $(
        block_detail! { $W, $H }
      )*
    };
  }

  blocks_detail! { (8, 8), (16, 16), (32, 32), (64, 64) }
  blocks_detail! { (4, 8), (8, 16), (16, 32), (32, 64) }
  blocks_detail! { (8, 4), (16, 8), (32, 16), (64, 32) }
  blocks_detail! { (4, 16), (8, 32), (16, 64) }
  blocks_detail! { (16, 4), (32, 8), (64, 16) }
}
