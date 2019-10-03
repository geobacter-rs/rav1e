// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_camel_case_types)]
#![allow(dead_code)]

pub use self::forward::*;
pub use self::inverse::*;

pub use rcore::transform::*;

use std::convert::*;

use crate::tiling::*;
use crate::util::*;

use crate::cpu_features::CpuFeatureLevel;

mod forward;
mod inverse;

pub static RAV1E_TX_TYPES: &[TxType] = &[
  TxType::DCT_DCT,
  TxType::ADST_DCT,
  TxType::DCT_ADST,
  TxType::ADST_ADST,
  // TODO: Add a speed setting for FLIPADST
  // TxType::FLIPADST_DCT,
  // TxType::DCT_FLIPADST,
  // TxType::FLIPADST_FLIPADST,
  // TxType::ADST_FLIPADST,
  // TxType::FLIPADST_ADST,
  TxType::IDTX,
  TxType::V_DCT,
  TxType::H_DCT,
  //TxType::V_FLIPADST,
  //TxType::H_FLIPADST,
];

pub const TX_TYPES: usize = 16;

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub enum TxSet {
  // DCT only
  TX_SET_DCTONLY,
  // DCT + Identity only
  TX_SET_DCT_IDTX,
  // Discrete Trig transforms w/o flip (4) + Identity (1)
  TX_SET_DTT4_IDTX,
  // Discrete Trig transforms w/o flip (4) + Identity (1) + 1D Hor/vert DCT (2)
  // for 16x16 only
  TX_SET_DTT4_IDTX_1DDCT_16X16,
  // Discrete Trig transforms w/o flip (4) + Identity (1) + 1D Hor/vert DCT (2)
  TX_SET_DTT4_IDTX_1DDCT,
  // Discrete Trig transforms w/ flip (9) + Identity (1)
  TX_SET_DTT9_IDTX,
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver DCT (2)
  TX_SET_DTT9_IDTX_1DDCT,
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver (6)
  // for 16x16 only
  TX_SET_ALL16_16X16,
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver (6)
  TX_SET_ALL16,
}

/// Utility function that returns the log of the ratio of the col and row sizes.
#[inline]
pub fn get_rect_tx_log_ratio(col: usize, row: usize) -> i8 {
  debug_assert!(col > 0 && row > 0);
  col.ilog() as i8 - row.ilog() as i8
}

// performs half a butterfly
#[inline]
fn half_btf(w0: i32, in0: i32, w1: i32, in1: i32, bit: usize) -> i32 {
  // Ensure defined behaviour for when w0*in0 + w1*in1 is negative and
  //   overflows, but w0*in0 + w1*in1 + rounding isn't.
  let result = (w0 * in0).wrapping_add(w1 * in1);
  // Implement a version of round_shift with wrapping
  if bit == 0 {
    result
  } else {
    result.wrapping_add(1 << (bit - 1)) >> bit
  }
}

// clamps value to a signed integer type of bit bits
#[inline]
fn clamp_value(value: i32, bit: usize) -> i32 {
  let max_value: i32 = ((1i64 << (bit - 1)) - 1) as i32;
  let min_value: i32 = (-(1i64 << (bit - 1))) as i32;
  clamp(value, min_value, max_value)
}

#[inline]
fn round_shift_array<T>(arr: &mut [i32], size: usize, bit: i8)
where
  T: ISimd<i32>,
{
  if bit == 0 {
    return;
  }
  debug_assert!(size <= arr.len(), "{} > {}", size, arr.len());
  debug_assert_eq!(size % T::LANES, 0);

  let arr = T::slice_cast_mut(arr);

  if bit > 0 {
    let bit = bit as u32;
    for v in arr.iter_mut() {
      *v = v.round_shift(bit);
    }
  } else {
    for v in arr.iter_mut() {
      *v <<= T::_splat(-bit as _);
    }
  }
}

#[inline(never)]
pub fn forward_transform(
  residual: &mut [i16], coeffs: &mut [i32], tx_size: TxSize, tx_type: TxType, bit_depth: usize,
) {
  macro_rules! specialize_f {
    (@foreach_block $tx_size:expr,
     ($(($b_x:expr, $b_y:expr),)*),
     $f:ident $args:tt ) =>
    {
      // we have to create the match arms (ie `$pattern => $expression`)
      // without using a macro. See https://github.com/rust-lang/rust/issues/12832
      paste::expr! {
        match $tx_size {
        $(
          [<TX_$b_x X$b_y>] => {
            // call the function with this size.
            crate::util::[<Block $b_x x $b_y>]:: $f $args;
          },
        )*
          _ => {
            unreachable!("unhandled size: {:?}", $tx_size);
          },
        }
      }
    };
    // Instantiate $f for every size possible.
    (@forall $tx_size:expr, $f:ident $args:tt) => {
      specialize_f! { @foreach_block
        $tx_size,
        ((4, 4), (8, 8), (16, 16), (32, 32), (64, 64),
         (4, 8), (8, 16), (16, 32), (32, 64),
         (8, 4), (16, 8), (32, 16), (64, 32),
         (4, 16), (8, 32), (16, 64),
         (16, 4), (32, 8), (64, 16), ),
        $f $args
      }
    };
  }

  let bit_depth = bit_depth.try_into().expect("non-u8 bit depth");
  specialize_f!(@forall tx_size, fht(residual, coeffs, tx_type, bit_depth));
}
pub fn inverse_transform_add<P>(
  input: &[i32], output: &mut PlaneRegionMut<'_, P>,
  tx_size: TxSize, tx_type: TxType,
  bit_depth: usize,
  cpu: CpuFeatureLevel,
) where
  P: Pixel,
{
  use rkernels::transform::inverse::dispatch;

  let width = tx_size.width();
  let height = tx_size.height();
  assert!(
    input.len() >= width.min(32) * height.min(32),
    "input is smaller than the compute block: {} vs {}",
    input.len(), width.min(32) * height.min(32),
  );
  assert!(
    output.rect().height >= height && output.rect().width >= width,
    "output is smaller than the compute block: ({}, {}) vs expected ({}, {})",
    output.rect().width,
    output.rect().height,
    width,
    height,
  );
  let row_tx_idx = tx_type.row_tx().tbl_idx();
  let col_tx_idx = tx_type.col_tx().tbl_idx();

  let bd = bit_depth.try_into()
    .expect("non-u8 bit depth");

  let output_stride = output.plane_cfg.stride as i32;

  let r = unsafe {
    dispatch(input.as_ptr(), output.data_ptr_mut(),
             output_stride, bd, width.try_into().unwrap(),
             height.try_into().unwrap(),
             row_tx_idx, col_tx_idx,
             tx_size, cpu)
  };
  if let Some(()) = r {
    return;
  }

  unimplemented!("{:?}, {:?}", tx_type, tx_size);
}

#[cfg(test)]
mod test {
  use super::TxType::*;
  use super::*;
  use crate::frame::*;
  use rand::random;

  fn test_roundtrip<T: Pixel>(
    tx_size: TxSize, tx_type: TxType, tolerance: i16,
  ) -> bool {
    let mut src_storage = AlignedArray::new([T::cast_from(0); 64 * 64]);
    let src = &mut src_storage[..tx_size.area()];
    let mut dst =
      Plane::wrap(vec![T::cast_from(0); tx_size.area()], tx_size.width());
    let mut res_storage = AlignedArray::new([0i16; 64 * 64]);
    let res = &mut res_storage[..tx_size.area()];
    let mut freq_storage = AlignedArray::new([0i32; 64 * 64]);
    let freq = &mut freq_storage[..tx_size.area()];
    for ((r, s), d) in
      res.iter_mut().zip(src.iter_mut()).zip(dst.data.iter_mut())
    {
      *s = T::cast_from(random::<u8>());
      *d = T::cast_from(random::<u8>());
      *r = i16::cast_from(*s) - i16::cast_from(*d);
    }
    forward_transform(res, freq, tx_size, tx_type, 8);
    inverse_transform_add(freq, &mut dst.as_region_mut(),
                          tx_size, tx_type, 8,
                          CpuFeatureLevel::default());

    let ne = src
      .iter()
      .zip(dst.data.iter())
      .enumerate()
      .filter(|(_, (&s, &d))| {
        i16::abs(i16::cast_from(s) - i16::cast_from(d)) > tolerance
      })
      .map(|v| format!("{:?}", v))
      .collect::<Vec<_>>();
    if ne.len() != 0 {
      eprintln!(
        "tx_size = {:?}, tx_type = {:?}, tolerance = {}",
        tx_size, tx_type, tolerance
      );
      eprintln!("roundtrip mismatch: {:#?}", ne);
      false
    } else {
      true
    }
  }

  #[test]
  fn log_tx_ratios() {
    let combinations = [
      (TxSize::TX_4X4, 0),
      (TxSize::TX_8X8, 0),
      (TxSize::TX_16X16, 0),
      (TxSize::TX_32X32, 0),
      (TxSize::TX_64X64, 0),
      (TxSize::TX_4X8, -1),
      (TxSize::TX_8X4, 1),
      (TxSize::TX_8X16, -1),
      (TxSize::TX_16X8, 1),
      (TxSize::TX_16X32, -1),
      (TxSize::TX_32X16, 1),
      (TxSize::TX_32X64, -1),
      (TxSize::TX_64X32, 1),
      (TxSize::TX_4X16, -2),
      (TxSize::TX_16X4, 2),
      (TxSize::TX_8X32, -2),
      (TxSize::TX_32X8, 2),
      (TxSize::TX_16X64, -2),
      (TxSize::TX_64X16, 2),
    ];

    for &(tx_size, expected) in combinations.iter() {
      println!(
        "Testing combination {:?}, {:?}",
        tx_size.width(),
        tx_size.height()
      );
      assert!(
        get_rect_tx_log_ratio(tx_size.width(), tx_size.height()) == expected
      );
    }
  }

  fn roundtrips<T: Pixel>() {
    let combinations = [
      (TX_4X4, DCT_DCT, 0),
      (TX_4X4, ADST_DCT, 0),
      (TX_4X4, DCT_ADST, 0),
      (TX_4X4, ADST_ADST, 0),
      (TX_4X4, IDTX, 0),
      (TX_4X4, V_DCT, 0),
      (TX_4X4, H_DCT, 0),
      (TX_4X4, V_ADST, 0),
      (TX_4X4, H_ADST, 0),
      (TX_8X8, DCT_DCT, 1),
      (TX_8X8, ADST_DCT, 1),
      (TX_8X8, DCT_ADST, 1),
      (TX_8X8, ADST_ADST, 1),
      (TX_8X8, IDTX, 0),
      (TX_8X8, V_DCT, 0),
      (TX_8X8, H_DCT, 0),
      (TX_8X8, V_ADST, 0),
      (TX_8X8, H_ADST, 1),
      (TX_16X16, DCT_DCT, 1),
      (TX_16X16, ADST_DCT, 1),
      (TX_16X16, DCT_ADST, 1),
      (TX_16X16, ADST_ADST, 1),
      (TX_16X16, IDTX, 0),
      (TX_16X16, V_DCT, 1),
      (TX_16X16, H_DCT, 1),
      // 32x tranforms only use DCT_DCT and IDTX
      (TX_32X32, DCT_DCT, 2),
      (TX_32X32, IDTX, 0),
      // 64x tranforms only use DCT_DCT and IDTX
      //(TX_64X64, DCT_DCT, 0),
      (TX_4X8, DCT_DCT, 1),
      (TX_8X4, DCT_DCT, 1),
      (TX_4X16, DCT_DCT, 1),
      (TX_16X4, DCT_DCT, 1),
      (TX_8X16, DCT_DCT, 1),
      (TX_16X8, DCT_DCT, 1),
      (TX_8X32, DCT_DCT, 2),
      (TX_32X8, DCT_DCT, 2),
      (TX_16X32, DCT_DCT, 2),
      (TX_32X16, DCT_DCT, 2),
    ];
    let mut failed = false;
    for &(tx_size, tx_type, tolerance) in combinations.iter() {
      println!("Testing combination {:?}, {:?}", tx_size, tx_type);
      if !test_roundtrip::<T>(tx_size, tx_type, tolerance) {
        failed = true;
      }
    }
    if failed {
      panic!("Some roundtrips failed");
    }
  }

  #[test]
  fn roundtrips_u8() {
    roundtrips::<u8>();
  }

  #[test]
  fn roundtrips_u16() {
    roundtrips::<u16>();
  }
}
