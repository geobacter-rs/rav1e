// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]

use crate::util::*;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Ord, Eq)]
pub enum PredictionMode {
  DC_PRED = 0,     // Average of above and left pixels
  V_PRED,      // Vertical
  H_PRED,      // Horizontal
  D45_PRED,    // Directional 45  deg = round(arctan(1/1) * 180/pi)
  D135_PRED,   // Directional 135 deg = 180 - 45
  D117_PRED,   // Directional 117 deg = 180 - 63
  D153_PRED,   // Directional 153 deg = 180 - 27
  D207_PRED,   // Directional 207 deg = 180 + 27
  D63_PRED,    // Directional 63  deg = round(arctan(2/1) * 180/pi)
  SMOOTH_PRED, // Combination of horizontal and vertical interpolation
  SMOOTH_V_PRED,
  SMOOTH_H_PRED,
  PAETH_PRED,
  UV_CFL_PRED,
  NEARESTMV,
  NEAR0MV,
  NEAR1MV,
  NEAR2MV,
  GLOBALMV,
  NEWMV,
  // Compound ref compound modes
  NEAREST_NEARESTMV,
  NEAR_NEARMV,
  NEAREST_NEWMV,
  NEW_NEARESTMV,
  NEAR_NEWMV,
  NEW_NEARMV,
  GLOBAL_GLOBALMV,
  NEW_NEWMV,
}

#[derive(Copy, Clone, Debug)]
pub enum PredictionVariant {
  NONE = 0,
  LEFT,
  TOP,
  BOTH,
}

impl PredictionVariant {
  pub fn new(x: usize, y: usize) -> Self {
    match (x, y) {
      (0, 0) => PredictionVariant::NONE,
      (_, 0) => PredictionVariant::LEFT,
      (0, _) => PredictionVariant::TOP,
      _ => PredictionVariant::BOTH,
    }
  }
}

impl Default for PredictionMode {
  fn default() -> Self {
    PredictionMode::DC_PRED
  }
}

impl PredictionMode {
  pub fn is_compound(self) -> bool {
    self >= PredictionMode::NEAREST_NEARESTMV
  }
  pub fn has_near(self) -> bool {
    (self >= PredictionMode::NEAR0MV && self <= PredictionMode::NEAR2MV)
      || self == PredictionMode::NEAR_NEARMV
      || self == PredictionMode::NEAR_NEWMV
      || self == PredictionMode::NEW_NEARMV
  }
  pub fn is_intra(self) -> bool {
    self < PredictionMode::NEARESTMV
  }

  pub fn is_cfl(self) -> bool {
    self == PredictionMode::UV_CFL_PRED
  }

  pub fn is_directional(self) -> bool {
    self >= PredictionMode::V_PRED && self <= PredictionMode::D63_PRED
  }
}

pub trait Dim {
  const W: usize;
  const H: usize;
}

impl<T> Dim for T
  where
    T: Block,
{
  const W: usize = T::Hori::LENGTH;
  const H: usize = T::Vert::LENGTH;
}
