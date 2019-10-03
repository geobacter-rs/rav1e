// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

cfg_if::cfg_if! {
  if #[cfg(nasm_x86_64)] {
    pub use crate::asm::x86::mc::*;
  } else if #[cfg(asm_neon)] {
    pub use crate::asm::aarch64::mc::*;
  } else {
    pub use self::native::*;
  }
}

use crate::cpu_features::CpuFeatureLevel;
use crate::frame::*;
use crate::tiling::*;
use crate::util::*;

use std::ops;

pub use rcore::mc::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionVector {
  pub row: i16,
  pub col: i16,
}

impl ops::Add<MotionVector> for MotionVector {
  type Output = MotionVector;

  fn add(self, _rhs: MotionVector) -> MotionVector {
    MotionVector { row: self.row + _rhs.row, col: self.col + _rhs.col }
  }
}

impl ops::Div<i16> for MotionVector {
  type Output = MotionVector;

  fn div(self, _rhs: i16) -> MotionVector {
    MotionVector { row: self.row / _rhs, col: self.col / _rhs }
  }
}

impl MotionVector {
  pub const fn quantize_to_fullpel(self) -> Self {
    Self { row: (self.row / 8) * 8, col: (self.col / 8) * 8 }
  }

  pub fn is_zero(self) -> bool {
    self.row == 0 && self.col == 0
  }
}

pub(crate) mod native {
  use std::convert::*;

  use super::*;
  use num_traits::*;

  unsafe fn run_filter<T: AsPrimitive<i32>>(
    src: *const T, stride: usize, filter: [i32; 8],
  ) -> i32 {
    filter
      .iter()
      .enumerate()
      .map(|(i, f)| {
        let p = src.add(i * stride);
        f * (*p).as_()
      })
      .sum::<i32>()
  }

  #[cold]
  pub fn put_8tap_ref<T: Pixel>(
    dst: &mut PlaneRegionMut<'_, T>, src: PlaneSlice<'_, T>, width: usize,
    height: usize, col_frac: i32, row_frac: i32, mode_x: FilterMode,
    mode_y: FilterMode, bit_depth: usize, _cpu: CpuFeatureLevel,
  ) {
    let ref_stride = src.plane.cfg.stride;
    let y_filter = get_filter(mode_y, row_frac, height);
    let x_filter = get_filter(mode_x, col_frac, width);
    let max_sample_val = ((1 << bit_depth) - 1) as i32;
    let intermediate_bits = 4 - if bit_depth == 12 { 2 } else { 0 };
    match (col_frac, row_frac) {
      (0, 0) => {
        for r in 0..height {
          let src_slice = &src[r];
          let dst_slice = &mut dst[r];
          dst_slice[..width].copy_from_slice(&src_slice[..width]);
        }
      }
      (0, _) => {
        let offset_slice = src.go_up(3);
        for r in 0..height {
          let src_slice = &offset_slice[r];
          let dst_slice = &mut dst[r];
          for c in 0..width {
            dst_slice[c] = T::cast_from(
              round_shift(
                unsafe {
                  run_filter(src_slice[c..].as_ptr(), ref_stride, y_filter)
                },
                7,
              )
              .max(0)
              .min(max_sample_val),
            );
          }
        }
      }
      (_, 0) => {
        let offset_slice = src.go_left(3);
        for r in 0..height {
          let src_slice = &offset_slice[r];
          let dst_slice = &mut dst[r];
          for c in 0..width {
            dst_slice[c] = T::cast_from(
              round_shift(
                round_shift(
                  unsafe { run_filter(src_slice[c..].as_ptr(), 1, x_filter) },
                  7 - intermediate_bits,
                ),
                intermediate_bits,
              )
              .max(0)
              .min(max_sample_val),
            );
          }
        }
      }
      (_, _) => {
        let mut intermediate = [0 as i16; 8 * (128 + 7)];

        let offset_slice = src.go_left(3).go_up(3);
        for cg in (0..width).step_by(8) {
          for r in 0..height + 7 {
            let src_slice = &offset_slice[r];
            for c in cg..(cg + 8).min(width) {
              intermediate[8 * r + (c - cg)] = round_shift(
                unsafe { run_filter(src_slice[c..].as_ptr(), 1, x_filter) },
                7 - intermediate_bits,
              ) as i16;
            }
          }

          for r in 0..height {
            let dst_slice = &mut dst[r];
            for c in cg..(cg + 8).min(width) {
              dst_slice[c] = T::cast_from(
                round_shift(
                  unsafe {
                    run_filter(
                      intermediate[8 * r + c - cg..].as_ptr(),
                      8,
                      y_filter,
                    )
                  },
                  7 + intermediate_bits,
                )
                .max(0)
                .min(max_sample_val),
              );
            }
          }
        }
      }
    }
  }

  #[cold]
  pub fn prep_8tap_ref<T: Pixel>(
    tmp: &mut [i16], src: PlaneSlice<'_, T>, width: usize, height: usize,
    col_frac: i32, row_frac: i32, mode_x: FilterMode, mode_y: FilterMode,
    bit_depth: usize, _cpu: CpuFeatureLevel,
  ) {
    let ref_stride = src.plane.cfg.stride;
    let y_filter = get_filter(mode_y, row_frac, height);
    let x_filter = get_filter(mode_x, col_frac, width);
    let intermediate_bits = 4 - if bit_depth == 12 { 2 } else { 0 };
    match (col_frac, row_frac) {
      (0, 0) => {
        for r in 0..height {
          let src_slice = &src[r];
          for c in 0..width {
            tmp[r * width + c] =
              i16::cast_from(src_slice[c]) << intermediate_bits;
          }
        }
      }
      (0, _) => {
        let offset_slice = src.go_up(3);
        for r in 0..height {
          let src_slice = &offset_slice[r];
          for c in 0..width {
            tmp[r * width + c] = round_shift(
              unsafe {
                run_filter(src_slice[c..].as_ptr(), ref_stride, y_filter)
              },
              7 - intermediate_bits,
            ) as i16;
          }
        }
      }
      (_, 0) => {
        let offset_slice = src.go_left(3);
        for r in 0..height {
          let src_slice = &offset_slice[r];
          for c in 0..width {
            tmp[r * width + c] = round_shift(
              unsafe { run_filter(src_slice[c..].as_ptr(), 1, x_filter) },
              7 - intermediate_bits,
            ) as i16;
          }
        }
      }
      (_, _) => {
        let mut intermediate = [0 as i16; 8 * (128 + 7)];

        let offset_slice = src.go_left(3).go_up(3);
        for cg in (0..width).step_by(8) {
          for r in 0..height + 7 {
            let src_slice = &offset_slice[r];
            for c in cg..(cg + 8).min(width) {
              intermediate[8 * r + (c - cg)] = round_shift(
                unsafe { run_filter(src_slice[c..].as_ptr(), 1, x_filter) },
                7 - intermediate_bits,
              ) as i16;
            }
          }

          for r in 0..height {
            for c in cg..(cg + 8).min(width) {
              tmp[r * width + c] = round_shift(
                unsafe {
                  run_filter(
                    intermediate[8 * r + c - cg..].as_ptr(),
                    8,
                    y_filter,
                  )
                },
                7,
              ) as i16;
            }
          }
        }
      }
    }
  }
  #[cold]
  pub fn mc_avg_ref<T: Pixel>(
    dst: &mut PlaneRegionMut<'_, T>, tmp1: &[i16], tmp2: &[i16], width: usize,
    height: usize, bit_depth: usize, _cpu: CpuFeatureLevel,
  ) {
    let max_sample_val = ((1 << bit_depth) - 1) as i32;
    let intermediate_bits = 4 - if bit_depth == 12 { 2 } else { 0 };
    for r in 0..height {
      let dst_slice = &mut dst[r];
      for c in 0..width {
        dst_slice[c] = T::cast_from(
          round_shift(
            tmp1[r * width + c] as i32 + tmp2[r * width + c] as i32,
            intermediate_bits + 1,
          )
          .max(0)
          .min(max_sample_val),
        );
      }
    }
  }

  pub fn put_8tap<T: Pixel>(
    dst: &mut PlaneRegionMut<'_, T>, src: PlaneSlice<'_, T>, width: usize,
    height: usize, col_frac: i32, row_frac: i32, mode_x: FilterMode,
    mode_y: FilterMode, bit_depth: usize, cpu: CpuFeatureLevel,
  ) {
    use rkernels::mc::put_8tap::dispatch;
    assert!(
      dst.rect().height >= height && dst.rect().width >= width,
      "dst is smaller than the compute block: ({}, {}) vs expected ({}, {})",
      dst.rect().width,
      dst.rect().height,
      width,
      height,
    );
    assert!(
      src.plane.data.len() >= height * src.plane.cfg.stride + width,
      "src is smaller than the compute block: {} vs expected {}",
      src.plane.data.len(),
      height * src.plane.cfg.stride + width,
    );
    let width = width.try_into().unwrap();
    let height = height.try_into().unwrap();

    let col_frac_i = (col_frac == 0) as usize;
    let row_frac_i = (row_frac == 0) as usize;

    let origin_dst = if cfg!(feature = "check_asm") {
      Some(dst.scratch_copy())
    } else {
      None
    };

    let r = unsafe {
      dispatch(dst.data_ptr_mut(),
               dst.plane_cfg.stride.try_into().unwrap(),
               src.as_ptr(),
               src.plane.cfg.stride.try_into().unwrap(),
               col_frac, row_frac,
               bit_depth as _,
               width, height,
               col_frac_i, row_frac_i,
               mode_x as usize, mode_y as usize,
               (width, height), cpu)
    };
    assert!(r.is_some(), "missing put_8tap kernel??");

    if let Some(mut origin_dst) = origin_dst {
      put_8tap_ref(
        &mut origin_dst.as_region_mut(),
        src,
        width as _,
        height as _,
        col_frac,
        row_frac,
        mode_x,
        mode_y,
        bit_depth,
        cpu,
      );

      let ne = origin_dst
        .as_region()
        .rows_iter()
        .flat_map(|row| row[..width as usize].iter().cloned())
        .zip({ dst.rows_iter().flat_map(|row| row[..width as usize].iter().cloned()) })
        .enumerate()
        .filter(|(_, (r, asm))| r != asm)
        .map(|v| format!("{:?}", v))
        .collect::<Vec<_>>();
      if ne.len() != 0 {
        eprintln!(
          "col_frac = {}, row_frac = {}, block = {:?}",
          col_frac, row_frac, (width, height)
        );
        eprintln!("put_8tap: ref vs asm mismatch: {:#?}", ne);
        panic!();
      }
    }
  }
  pub fn prep_8tap<T: Pixel>(
    tmp: &mut [i16], src: PlaneSlice<'_, T>, width: usize, height: usize,
    col_frac: i32, row_frac: i32, mode_x: FilterMode, mode_y: FilterMode,
    bit_depth: usize, cpu: CpuFeatureLevel,
  ) {
    use rkernels::mc::prep_8tap::dispatch;
    assert!(
      tmp.len() >= width * height,
      "tmp is smaller than the compute block"
    );
    assert!(
      src.plane.data.len() >= height * src.plane.cfg.stride + width,
      "src is smaller than the compute block: {} vs expected {}",
      src.plane.data.len(),
      height * src.plane.cfg.stride + width,
    );
    let width = width.try_into().unwrap();
    let height = height.try_into().unwrap();

    let col_frac_i = (col_frac == 0) as usize;
    let row_frac_i = (row_frac == 0) as usize;

    let origin_tmp =
      if cfg!(feature = "check_asm") { Some(tmp.to_owned()) } else { None };

    let r = unsafe {
      dispatch(tmp.as_mut_ptr(),
               src.as_ptr(),
               src.plane.cfg.stride.try_into().unwrap(),
               col_frac, row_frac,
               bit_depth as _,
               width, height,
               col_frac_i, row_frac_i,
               mode_x as usize, mode_y as usize,
               (width, height), cpu)
    };
    assert!(r.is_some(), "missing prep_8tap kernel??");

    if let Some(mut origin_tmp) = origin_tmp {
      prep_8tap_ref(
        &mut origin_tmp,
        src,
        width as _,
        height as _,
        col_frac,
        row_frac,
        mode_x,
        mode_y,
        bit_depth,
        cpu,
      );
      let ne = origin_tmp
        .into_iter()
        .enumerate()
        .zip(tmp.iter())
        .map(|((idx, r), &asm)| (idx, r, asm))
        .filter(|(_, r, asm)| r != asm)
        .map(|v| format!("{:?}", v))
        .collect::<Vec<_>>();
      if ne.len() != 0 {
        eprintln!(
          "col_frac = {}, row_frac = {}, block = {:?}",
          col_frac, row_frac, (width, height)
        );
        eprintln!("prep_8tap: ref vs asm mismatch: {:#?}", ne);
        panic!();
      }
    }
  }
  pub fn mc_avg<T: Pixel>(
    dst: &mut PlaneRegionMut<'_, T>, tmp1: &[i16], tmp2: &[i16], width: usize,
    height: usize, bit_depth: usize, cpu: CpuFeatureLevel,
  ) {
    use rkernels::mc::mc_avg::dispatch;
    assert!(
      tmp1.len() >= width * height,
      "tmp1 is smaller than the compute block"
    );
    assert!(
      tmp2.len() >= width * height,
      "tmp2 is smaller than the compute block"
    );
    assert!(
      dst.rect().height >= height && dst.rect().width >= width,
      "dst is smaller than the compute block"
    );
    let width = width.try_into().unwrap();
    let height = height.try_into().unwrap();

    let origin_dst = if cfg!(feature = "check_asm") {
      Some(dst.scratch_copy())
    } else {
      None
    };

    let r = unsafe {
      dispatch(dst.data_ptr_mut(),
               dst.plane_cfg.stride.try_into().unwrap(),
               tmp1.as_ptr(), tmp2.as_ptr(),
               bit_depth as _,
               width, height,
               (width, height), cpu)
    };
    assert!(r.is_some(), "missing mc_avg kernel??");

    if let Some(mut origin_dst) = origin_dst {
      mc_avg_ref(
        &mut origin_dst.as_region_mut(),
        tmp1,
        tmp2,
        width as _,
        height as _,
        bit_depth,
        cpu,
      );
      let ne = origin_dst
        .as_region()
        .rows_iter()
        .flat_map(|row| row[..width as usize].iter().cloned())
        .zip({ dst.rows_iter().flat_map(|row| row[..width as usize].iter().cloned()) })
        .enumerate()
        .filter(|(_, (r, asm))| r != asm)
        .map(|v| format!("{:?}", v))
        .collect::<Vec<_>>();
      if ne.len() != 0 {
        eprintln!("block = {:?}", (width, height));
        eprintln!("put_8tap: ref vs asm mismatch: {:#?}", ne);
        panic!();
      }
    }
  }
}
