// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

pub const RESTORATION_TILESIZE_MAX_LOG2: usize = 8;

pub const RESTORE_NONE: u8 = 0;
pub const RESTORE_SWITCHABLE: u8 = 1;
pub const RESTORE_WIENER: u8 = 2;
pub const RESTORE_SGRPROJ: u8 = 3;

pub const WIENER_TAPS_MIN: [i8; 3] = [-5, -23, -17];
pub const WIENER_TAPS_MID: [i8; 3] = [3, -7, 15];
pub const WIENER_TAPS_MAX: [i8; 3] = [10, 8, 46];
#[allow(unused)]
pub const WIENER_TAPS_K: [i8; 3] = [1, 2, 3];
pub const WIENER_BITS: usize = 7;

pub const SGRPROJ_XQD_MIN: [i8; 2] = [-96, -32];
pub const SGRPROJ_XQD_MID: [i8; 2] = [-32, 31];
pub const SGRPROJ_XQD_MAX: [i8; 2] = [31, 95];
pub const SGRPROJ_PRJ_SUBEXP_K: u8 = 4;
pub const SGRPROJ_PRJ_BITS: u8 = 7;
pub const SGRPROJ_PARAMS_BITS: u8 = 4;
pub const SGRPROJ_MTABLE_BITS: u8 = 20;
pub const SGRPROJ_SGR_BITS: u8 = 8;
pub const SGRPROJ_RECIP_BITS: u8 = 12;
pub const SGRPROJ_RST_BITS: u8 = 4;
pub const SGRPROJ_PARAMS_S: [[u32; 2]; 1 << SGRPROJ_PARAMS_BITS] = [
  [140, 3236],
  [112, 2158],
  [93, 1618],
  [80, 1438],
  [70, 1295],
  [58, 1177],
  [47, 1079],
  [37, 996],
  [30, 925],
  [25, 863],
  [0, 2589],
  [0, 1618],
  [0, 1177],
  [0, 925],
  [56, 0],
  [22, 0],
];

pub const SOLVE_IMAGE_MAX: usize = (1 << RESTORATION_TILESIZE_MAX_LOG2);
pub const SOLVE_IMAGE_STRIDE: usize = SOLVE_IMAGE_MAX + 6 + 2;
pub const SOLVE_IMAGE_HEIGHT: usize = SOLVE_IMAGE_STRIDE;
pub const SOLVE_IMAGE_SIZE: usize = SOLVE_IMAGE_STRIDE * SOLVE_IMAGE_HEIGHT;

pub const STRIPE_IMAGE_MAX: usize = (1 << RESTORATION_TILESIZE_MAX_LOG2)
  + (1 << (RESTORATION_TILESIZE_MAX_LOG2 - 1));
pub const STRIPE_IMAGE_STRIDE: usize = STRIPE_IMAGE_MAX + 6 + 2;
pub const STRIPE_IMAGE_HEIGHT: usize = 64 + 6 + 2;
pub const STRIPE_IMAGE_SIZE: usize = STRIPE_IMAGE_STRIDE * STRIPE_IMAGE_HEIGHT;

pub const IMAGE_WIDTH_MAX: usize = [STRIPE_IMAGE_MAX, SOLVE_IMAGE_MAX]
  [(STRIPE_IMAGE_MAX < SOLVE_IMAGE_MAX) as usize];
