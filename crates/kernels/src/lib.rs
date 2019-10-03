// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(unused_imports)]
#![allow(dead_code)]

pub mod dist {
  pub mod sad {
    include!(concat!(env!("OUT_DIR"), "/sad_kernels.rs"));
  }
  pub mod satd {
    include!(concat!(env!("OUT_DIR"), "/satd_kernels.rs"));
  }
}
pub mod lrf {
  include!(concat!(env!("OUT_DIR"), "/lrf_kernels.rs"));
}
pub mod mc {
  pub mod put_8tap {
    include!(concat!(env!("OUT_DIR"), "/put_8tap_kernels.rs"));
  }
  pub mod prep_8tap {
    include!(concat!(env!("OUT_DIR"), "/prep_8tap_kernels.rs"));
  }
  pub mod mc_avg {
    include!(concat!(env!("OUT_DIR"), "/mc_avg_kernels.rs"));
  }
}
pub mod predict {
  include!(concat!(env!("OUT_DIR"), "/pred_kernels.rs"));
}
pub mod transform {
  pub mod inverse {
    include!(concat!(env!("OUT_DIR"), "/inv_tx_add_kernels.rs"));
  }
}
