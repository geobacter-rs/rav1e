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
use std::fmt;

use crate::context::*;
use crate::transform::*;

pub use self::BlockSize::*;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq)]
pub enum BlockSize {
  BLOCK_4X4,
  BLOCK_4X8,
  BLOCK_8X4,
  BLOCK_8X8,
  BLOCK_8X16,
  BLOCK_16X8,
  BLOCK_16X16,
  BLOCK_16X32,
  BLOCK_32X16,
  BLOCK_32X32,
  BLOCK_32X64,
  BLOCK_64X32,
  BLOCK_64X64,
  BLOCK_64X128,
  BLOCK_128X64,
  BLOCK_128X128,
  BLOCK_4X16,
  BLOCK_16X4,
  BLOCK_8X32,
  BLOCK_32X8,
  BLOCK_16X64,
  BLOCK_64X16,
  BLOCK_INVALID,
}
impl TryFrom<(usize, usize)> for BlockSize {
  type Error = ();
  fn try_from((w, h): (usize, usize)) -> Result<Self, ()> {
    Ok(match (w, h) {
      (4, 4) => BLOCK_4X4,
      (4, 8) => BLOCK_4X8,
      (8, 4) => BLOCK_8X4,
      (8, 8) => BLOCK_8X8,
      (8, 16) => BLOCK_8X16,
      (16, 8) => BLOCK_16X8,
      (16, 16) => BLOCK_16X16,
      (16, 32) => BLOCK_16X32,
      (32, 16) => BLOCK_32X16,
      (32, 32) => BLOCK_32X32,
      (32, 64) => BLOCK_32X64,
      (64, 32) => BLOCK_64X32,
      (64, 64) => BLOCK_64X64,
      (64, 128) => BLOCK_64X128,
      (128, 64) => BLOCK_128X64,
      (128, 128) => BLOCK_128X128,
      (4, 16) => BLOCK_4X16,
      (16, 4) => BLOCK_16X4,
      (8, 32) => BLOCK_8X32,
      (32, 8) => BLOCK_32X8,
      (16, 64) => BLOCK_16X64,
      (64, 16) => BLOCK_64X16,
      _ => { return Err(()); },
    })
  }
}

impl BlockSize {
  pub const BLOCK_SIZES_ALL: usize = 22;
  pub const BLOCK_SIZES: usize = BlockSize::BLOCK_SIZES_ALL - 6; // BLOCK_SIZES_ALL minus 4:1 non-squares, six of them

  pub fn from_width_and_height(w: usize, h: usize) -> BlockSize {
    Self::try_from((w, h))
      .unwrap_or_else(|()| {
        panic!("invalid block size: ({}, {})", w, h)
      })
  }

  pub fn cfl_allowed(self) -> bool {
    // TODO: fix me when enabling EXT_PARTITION_TYPES
    self <= BlockSize::BLOCK_32X32
  }

  pub fn width(self) -> usize {
    1 << self.width_log2()
  }

  pub fn width_log2(self) -> usize {
    match self {
      BLOCK_4X4 | BLOCK_4X8 | BLOCK_4X16 => 2,
      BLOCK_8X4 | BLOCK_8X8 | BLOCK_8X16 | BLOCK_8X32 => 3,
      BLOCK_16X4 | BLOCK_16X8 | BLOCK_16X16 | BLOCK_16X32 | BLOCK_16X64 => 4,
      BLOCK_32X8 | BLOCK_32X16 | BLOCK_32X32 | BLOCK_32X64 => 5,
      BLOCK_64X16 | BLOCK_64X32 | BLOCK_64X64 | BLOCK_64X128 => 6,
      BLOCK_128X64 | BLOCK_128X128 => 7,
      BLOCK_INVALID => unreachable!(),
    }
  }

  pub fn width_mi(self) -> usize {
    self.width() >> MI_SIZE_LOG2
  }

  pub fn height(self) -> usize {
    1 << self.height_log2()
  }

  pub fn height_log2(self) -> usize {
    match self {
      BLOCK_4X4 | BLOCK_8X4 | BLOCK_16X4 => 2,
      BLOCK_4X8 | BLOCK_8X8 | BLOCK_16X8 | BLOCK_32X8 => 3,
      BLOCK_4X16 | BLOCK_8X16 | BLOCK_16X16 | BLOCK_32X16 | BLOCK_64X16 => 4,
      BLOCK_8X32 | BLOCK_16X32 | BLOCK_32X32 | BLOCK_64X32 => 5,
      BLOCK_16X64 | BLOCK_32X64 | BLOCK_64X64 | BLOCK_128X64 => 6,
      BLOCK_64X128 | BLOCK_128X128 => 7,
      BLOCK_INVALID => unreachable!(),
    }
  }

  pub fn height_mi(self) -> usize {
    self.height() >> MI_SIZE_LOG2
  }

  pub fn tx_size(self) -> TxSize {
    match self {
      BLOCK_4X4 => TX_4X4,
      BLOCK_4X8 => TX_4X8,
      BLOCK_8X4 => TX_8X4,
      BLOCK_8X8 => TX_8X8,
      BLOCK_8X16 => TX_8X16,
      BLOCK_16X8 => TX_16X8,
      BLOCK_16X16 => TX_16X16,
      BLOCK_16X32 => TX_16X32,
      BLOCK_32X16 => TX_32X16,
      BLOCK_32X32 => TX_32X32,
      BLOCK_32X64 => TX_32X64,
      BLOCK_64X32 => TX_64X32,
      BLOCK_4X16 => TX_4X16,
      BLOCK_16X4 => TX_16X4,
      BLOCK_8X32 => TX_8X32,
      BLOCK_32X8 => TX_32X8,
      BLOCK_16X64 => TX_16X64,
      BLOCK_64X16 => TX_64X16,
      BLOCK_INVALID => unreachable!(),
      _ => TX_64X64,
    }
  }

  /// Source: Subsampled_Size (AV1 specification section 5.11.38)
  pub fn subsampled_size(self, xdec: usize, ydec: usize) -> BlockSize {
    match (xdec, ydec) {
      (0, 0) /* 4:4:4 */ => self,
      (1, 0) /* 4:2:2 */ => match self {
        BLOCK_4X4 | BLOCK_8X4 => BLOCK_4X4,
        BLOCK_8X8 => BLOCK_4X8,
        BLOCK_16X4 => BLOCK_8X4,
        BLOCK_16X8 => BLOCK_8X8,
        BLOCK_16X16 => BLOCK_8X16,
        BLOCK_32X8 => BLOCK_16X8,
        BLOCK_32X16 => BLOCK_16X16,
        BLOCK_32X32 => BLOCK_16X32,
        BLOCK_64X16 => BLOCK_32X16,
        BLOCK_64X32 => BLOCK_32X32,
        BLOCK_64X64 => BLOCK_32X64,
        BLOCK_128X64 => BLOCK_64X64,
        BLOCK_128X128 => BLOCK_64X128,
        _ => BLOCK_INVALID
      },
      (1, 1) /* 4:2:0 */ => match self {
        BLOCK_4X4 | BLOCK_4X8 | BLOCK_8X4 | BLOCK_8X8 => BLOCK_4X4,
        BLOCK_4X16 | BLOCK_8X16 => BLOCK_4X8,
        BLOCK_8X32 => BLOCK_4X16,
        BLOCK_16X4 | BLOCK_16X8 => BLOCK_8X4,
        BLOCK_16X16 => BLOCK_8X8,
        BLOCK_16X32 => BLOCK_8X16,
        BLOCK_16X64 => BLOCK_8X32,
        BLOCK_32X8 => BLOCK_16X4,
        BLOCK_32X16 => BLOCK_16X8,
        BLOCK_32X32 => BLOCK_16X16,
        BLOCK_32X64 => BLOCK_16X32,
        BLOCK_64X16 => BLOCK_32X8,
        BLOCK_64X32 => BLOCK_32X16,
        BLOCK_64X64 => BLOCK_32X32,
        BLOCK_64X128 => BLOCK_32X64,
        BLOCK_128X64 => BLOCK_64X32,
        BLOCK_128X128 => BLOCK_64X64,
        _ => BLOCK_INVALID
      },
      _ => unreachable!()
    }
  }

  pub fn largest_chroma_tx_size(self, xdec: usize, ydec: usize) -> TxSize {
    let plane_bsize = self.subsampled_size(xdec, ydec);
    if plane_bsize == BLOCK_INVALID {
      panic!("invalid block size for this subsampling mode");
    }

    let uv_tx = max_txsize_rect_lookup[plane_bsize as usize];

    av1_get_coded_tx_size(uv_tx)
  }

  pub fn is_sqr(self) -> bool {
    self.width_log2() == self.height_log2()
  }

  pub fn is_sub8x8(self, xdec: usize, ydec: usize) -> bool {
    xdec != 0 && self.width_log2() == 2 || ydec != 0 && self.height_log2() == 2
  }

  pub fn sub8x8_offset(self, xdec: usize, ydec: usize) -> (isize, isize) {
    let offset_x = if xdec != 0 && self.width_log2() == 2 { -1 } else { 0 };
    let offset_y = if ydec != 0 && self.height_log2() == 2 { -1 } else { 0 };

    (offset_x, offset_y)
  }

  pub fn greater_than(self, other: BlockSize) -> bool {
    (self.width() > other.width() && self.height() >= other.height())
      || (self.width() >= other.width() && self.height() > other.height())
  }

  pub fn gte(self, other: BlockSize) -> bool {
    self.greater_than(other)
      || (self.width() == other.width() && self.height() == other.height())
  }

  pub fn is_rect_tx_allowed(self) -> bool {
    match self {
      BLOCK_4X4 | BLOCK_8X8 | BLOCK_16X16 | BLOCK_32X32 | BLOCK_64X64
      | BLOCK_64X128 | BLOCK_128X64 | BLOCK_128X128 => false,
      BLOCK_INVALID => unreachable!(),
      _ => true,
    }
  }
}
impl Into<(u16, u16)> for BlockSize {
  fn into(self) -> (u16, u16) {
    (self.width().try_into().unwrap(),
     self.height().try_into().unwrap())
  }
}

impl fmt::Display for BlockSize {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    write!(
      f,
      "{}",
      match self {
        BlockSize::BLOCK_4X4 => "4x4",
        BlockSize::BLOCK_4X8 => "4x8",
        BlockSize::BLOCK_8X4 => "8x4",
        BlockSize::BLOCK_8X8 => "8x8",
        BlockSize::BLOCK_8X16 => "8x16",
        BlockSize::BLOCK_16X8 => "16x8",
        BlockSize::BLOCK_16X16 => "16x16",
        BlockSize::BLOCK_16X32 => "16x32",
        BlockSize::BLOCK_32X16 => "32x16",
        BlockSize::BLOCK_32X32 => "32x32",
        BlockSize::BLOCK_32X64 => "32x64",
        BlockSize::BLOCK_64X32 => "64x32",
        BlockSize::BLOCK_64X64 => "64x64",
        BlockSize::BLOCK_64X128 => "64x128",
        BlockSize::BLOCK_128X64 => "128x64",
        BlockSize::BLOCK_128X128 => "128x128",
        BlockSize::BLOCK_4X16 => "4x16",
        BlockSize::BLOCK_16X4 => "16x4",
        BlockSize::BLOCK_8X32 => "8x32",
        BlockSize::BLOCK_32X8 => "32x8",
        BlockSize::BLOCK_16X64 => "16x64",
        BlockSize::BLOCK_64X16 => "64x16",
        BlockSize::BLOCK_INVALID => "Invalid",
      }
    )
  }
}
