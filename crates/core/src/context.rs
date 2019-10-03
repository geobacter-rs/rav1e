// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_upper_case_globals)]
#![allow(dead_code)]
#![allow(non_camel_case_types)]

use crate::partition::*;
use crate::transform::{*, TxSize::*, };

pub const MI_SIZE_LOG2: usize = 2;

pub static max_txsize_rect_lookup: [TxSize; BlockSize::BLOCK_SIZES_ALL] = [
  TX_4X4,   // 4x4
  TX_4X8,   // 4x8
  TX_8X4,   // 8x4
  TX_8X8,   // 8x8
  TX_8X16,  // 8x16
  TX_16X8,  // 16x8
  TX_16X16, // 16x16
  TX_16X32, // 16x32
  TX_32X16, // 32x16
  TX_32X32, // 32x32
  TX_32X64, // 32x64
  TX_64X32, // 64x32
  TX_64X64, // 64x64
  TX_64X64, // 64x128
  TX_64X64, // 128x64
  TX_64X64, // 128x128
  TX_4X16,  // 4x16
  TX_16X4,  // 16x4
  TX_8X32,  // 8x32
  TX_32X8,  // 32x8
  TX_16X64, // 16x64
  TX_64X16, // 64x16
];

pub fn av1_get_coded_tx_size(tx_size: TxSize) -> TxSize {
  match tx_size {
    TX_64X64 | TX_64X32 | TX_32X64 => TX_32X32,
    TX_16X64 => TX_16X32,
    TX_64X16 => TX_32X16,
    _ => tx_size,
  }
}
