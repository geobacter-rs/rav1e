// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::collections::btree_map::*;
use std::io::Write;

use super::*;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum FilterMode {
  REGULAR = 0,
  SMOOTH = 1,
  SHARP = 2,
  BILINEAR = 3,
}
impl FilterMode {
  fn modes() -> &'static [Self] {
    const C: &'static [FilterMode] = &[
      FilterMode::REGULAR,
      FilterMode::SMOOTH,
      FilterMode::SHARP,
      FilterMode::BILINEAR,
    ];
    C
  }
  fn fn_suffix(&self) -> &'static str {
    match self {
      FilterMode::REGULAR => "reg",
      FilterMode::SMOOTH => "smooth",
      FilterMode::SHARP => "sharp",
      FilterMode::BILINEAR => "bilinear",
    }
  }
}
impl ToTokens for FilterMode {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let t = match self {
      FilterMode::REGULAR => quote!(FilterMode::REGULAR),
      FilterMode::SMOOTH => quote!(FilterMode::SMOOTH),
      FilterMode::SHARP => quote!(FilterMode::SHARP),
      FilterMode::BILINEAR => quote!(FilterMode::BILINEAR),
    };
    tokens.extend(t);
  }
}
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct BlockFilterMode {
  pub x: FilterMode,
  pub y: FilterMode,
}
impl BlockFilterMode {
  fn modes() -> Vec<Self> {
    let modes = FilterMode::modes();
    let mut out = Vec::with_capacity(modes.len() * modes.len());
    for &mode_x in modes.iter() {
      for &mode_y in modes.iter() {
        out.push(BlockFilterMode {
          x: mode_x,
          y: mode_y,
        });
      }
    }
    out
  }
  fn fn_suffix(&self) -> String {
    format!("{}_{}", self.x.fn_suffix(), self.y.fn_suffix())
  }
  fn module_name(&self) -> Ident {
    Ident::new(&self.fn_suffix(), Span::call_site())
  }

  fn reg_reg() -> Self {
    BlockFilterMode {
      x: FilterMode::REGULAR,
      y: FilterMode::REGULAR,
    }
  }
  fn is_reg_reg(&self) -> bool {
    self == &Self::reg_reg()
  }

  // don't generate extra kernels for these unused permutations
  fn canonicalize(&mut self, frac: EightTapFrac) -> bool {
    if frac.0 {
      self.x = FilterMode::REGULAR;
    }
    if frac.1 {
      self.y = FilterMode::REGULAR;
    }
    frac.0 || frac.1
  }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct EightTapFrac(bool, bool);
impl EightTapFrac {
  fn fracs() -> &'static [Self] {
    const C: &'static [EightTapFrac] = &[
      EightTapFrac(true, true),
      EightTapFrac(true, false),
      EightTapFrac(false, true),
      EightTapFrac(false, false),
    ];
    C
  }
  fn fn_suffix(&self) -> &'static str {
    match self {
      EightTapFrac(true, true) => "0_0",
      EightTapFrac(true, false) => "0_y",
      EightTapFrac(false, true) => "x_0",
      EightTapFrac(false, false) => "x_y",
    }
  }
  fn module_name(&self) -> Ident {
    let name = format!("f_{}", self.fn_suffix());
    Ident::new(&name, Span::call_site())
  }

  fn is_copy(&self) -> bool {
    self == &EightTapFrac(true, true)
  }
}

impl TableEntry {
  fn new_8tap_table_entry<T>(isa: IsaFeature,
                             frac: EightTapFrac,
                             bfm: BlockFilterMode,
                             b: Block,
                             path: T,
                             f: Result<Func, Ident>)
    -> TableEntry
    where T: Into<Rc<TokenStream>>,
  {
    let feature_idx = isa.index();
    let b_enum = b.table_idx();
    let row_frac = frac.0 as usize;
    let col_frac = frac.1 as usize;
    let x_mode = bfm.x as usize;
    let y_mode = bfm.y as usize;
    let idx = quote! {
      [#feature_idx][#b_enum][#row_frac][#col_frac][#x_mode][#y_mode]
    };
    let (name, f) = match f {
      Ok(f) => (f.name(), Some(f)),
      Err(name) => (name, None),
    };
    TableEntry {
      indices: Rc::new(idx),
      name,
      path: path.into(),
      func: f,
    }
  }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct SetKey {
  kind: &'static str,
  frac: EightTapFrac,
  bfm: BlockFilterMode,
}

type PxFunctions = BTreeMap<EightTapFrac, FracFunctions>;
struct FracFunctions {
  /// If None, we've already written all the kernels to the module.
  /// No more kernels are allowed.
  out: Option<Module>,

  frac: EightTapFrac,
  funcs: BTreeMap<BlockFilterMode, EightTapFunctions>,
}
struct EightTapFunctions {
  /// If None, we've already written all the kernels to the module.
  /// No more kernels are allowed.
  out: Option<Module>,

  bfm: BlockFilterMode,
  frac: EightTapFrac,

  put_internal: BTreeMap<Block, TableEntry>,
  prep_internal: BTreeMap<Block, TableEntry>,
  put: BTreeMap<Block, TableEntry>,
  prep: BTreeMap<Block, TableEntry>,
}
#[derive(Default)]
struct AvgFunctions {
  /// per lane width
  avg_internal: BTreeMap<usize, Ident>,
  avg: BTreeMap<Block, TableEntry>,
}

type Ctx8Tap<'a> = super::Ctx<'a, PxFunctions, &'a mut BTreeMap<PixelType, PxFunctions>, SetKey>;
type CtxAvg<'a> = super::Ctx<'a, AvgFunctions, &'a mut BTreeMap<PixelType, AvgFunctions>, ()>;

impl<'a> Ctx8Tap<'a> {
  fn mc_funcs(&self, bfm: BlockFilterMode, frac: EightTapFrac) -> Option<&EightTapFunctions> {
    self.funcs.get(&frac)?
      .funcs
      .get(&bfm)
  }
  fn funcs_mut(&mut self, bfm: BlockFilterMode,
               frac: EightTapFrac) -> &mut EightTapFunctions {
    let kind = self.kind;
    let bfmfs = match self.funcs.entry(frac) {
      Entry::Occupied(o) => o.into_mut(),
      Entry::Vacant(v) => {
        let frac_module = self.out
          .new_child(frac.module_name());
        let insert = FracFunctions {
          out: Some(frac_module),
          frac,
          funcs: Default::default()
        };
        v.insert(insert)
      },
    };
    match bfmfs.funcs.entry(bfm) {
      Entry::Occupied(o) => o.into_mut(),
      Entry::Vacant(v) => {
        assert!(bfmfs.out.is_some());

        let mut bfm_module = bfmfs.out
          .as_mut()
          .unwrap()
          .new_child_file(kind, bfm.module_name());
        bfm_module.extend(quote! {
          use rcore::mc::{FilterMode, get_filter, };
        });
        StdImports.to_tokens(&mut bfm_module);
        let insert = EightTapFunctions {
          out: Some(bfm_module),

          bfm,
          frac,

          put_internal: Default::default(),
          prep_internal: Default::default(),

          put: Default::default(),
          prep: Default::default(),
        };
        v.insert(insert)
      },
    }
  }
  fn new_mc_func<T>(&self, name: T, args: TokenStream,
                    params: Vec<(TokenStream, TokenStream)>,
                    public: bool)
    -> Func
    where T: Display,
  {
    let name = format!("{}_{}_{}_{}_{}",
                       name,
                       self.frac.fn_suffix(),
                       self.bfm.fn_suffix(),
                       self.px.type_str(),
                       self.isa.fn_suffix());
    let name = Ident::new(&name, Span::call_site());

    let mut path = vec![Ident::new("crate", Span::call_site())];
    let self_out = self.self_funcs().out.as_ref().unwrap();
    for &(ref parent, _) in self_out.parents.iter() {
      path.push(parent.clone());
    }
    path.push(self_out.name.clone());
    path.push(name.clone());

    Func {
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
    }
  }
  fn insert_func(set: &mut BTreeMap<Block, TableEntry>, b: Block,
                 entry: TableEntry)
    -> TableEntry
  {
    let out = entry.clone();
    assert!(set.insert(b, entry).is_none());
    out
  }

  fn self_funcs_mut(&mut self) -> &mut EightTapFunctions {
    self.funcs_mut(self.bfm, self.frac)
  }
  fn self_funcs(&self) -> &EightTapFunctions {
    self.mc_funcs(self.bfm, self.frac)
      .expect("funcs not yet created for self?")
  }

  fn eight_tap_redir(&self, origin_b: Block) -> Option<(Block, BlockFilterMode)> {
    // the only difference is when we can exploit wider SIMD.
    // Thus the height and bfm are ignored.

    let mut simd_width = self.px_simd_width()
      .min(origin_b.w());

    match self.frac {
      EightTapFrac(true, true) => {},
      _ => {
        simd_width = 8;
      },
    }

    // a square block will always have an index
    let b = Block(simd_width, simd_width);
    let mut bfm = self.bfm;

    if bfm.canonicalize(self.frac) && self.bfm != bfm || b != origin_b {
      Some((b, bfm))
    } else {
      None
    }
  }

  /// In the (false, false) case, I've diverged slightly from the
  /// reference implementation. In the first loop over the rows, I've
  /// swapped the write to intermediates so that we write to the
  /// corresponding transposed position. This allows the second row
  /// loop to load the row from intermediates with a stride of 1,
  /// saving us from having to use a vector gather.
  /// Ditto for `Self::prep_body`.
  /// TODO: for the smaller blocks, it'd be faster to compute the transpose
  /// rather than use a gather.
  fn put_internal(&mut self, b: Block) -> TableEntry {
    let origin_b = b;
    let (b, bfm) = self.eight_tap_redir(origin_b)
      .unwrap_or((origin_b, self.bfm));
    let isa = self.isa;
    let frac = self.frac;
    let cur_bfm = self.bfm;

    let entry = {
      self.funcs_mut(bfm, self.frac)
        .put_internal
        .get(&b)
        .map(|f| {
          TableEntry::new_8tap_table_entry(isa,
                                           frac,
                                           cur_bfm,
                                           b,
                                           f.path.clone(),
                                           Err(f.name.clone()))
        })
    };
    {
      let self_funcs = self.self_funcs_mut();
      if let Some(entry) = entry {
        let map_entry = self_funcs
          .put_internal
          .entry(b);
        match map_entry {
          Entry::Occupied(o) => {
            return o.get().clone();
          },
          Entry::Vacant(v) => {
            return v.insert(entry)
              .clone();
          }
        }
      } else {
        assert!(self_funcs.out.is_some());
      }
    }

    // Note: width is only the lower bound of possible sizes
    let width = b.w();

    let calc_ty = PrimType::I32;
    let px = match self.px {
      PixelType::U8 => PrimType::U8,
      PixelType::U16 => PrimType::U16,
    };
    // the simd width:
    let step = 8usize;
    let calc_simd = SimdType::new(calc_ty, step);
    let px_simd = SimdType::new(px, step);

    let mut idxs = TokenStream::default();
    for i in 0..step {
      let i = i as i32;
      idxs.extend(quote! {
        #i,
      });
    }
    let idxs = quote!(Simd::<[i32; #step]>::new(#idxs));

    let mode_x = bfm.x;
    let mode_y = bfm.y;

    let x_filter = Ident::new("x_filter", Span::call_site());
    let y_filter = Ident::new("y_filter", Span::call_site());
    let intermediate = Ident::new("intermediate", Span::call_site());

    let load_unit_stride_src = |dst: &mut TokenStream, px, src: &Plane| {
      SimdType::new(px, step)
        .uload(src)
        .cast(calc_simd)
        .let_(dst, "t")
    };
    let load_src = |dst: &mut TokenStream, px, src: &Plane| {
      let stride = src.stride();
      let t = quote! {{
        let ptrs = Simd::<[*const #px; #step]>::splat(#src);
        let idxs = #idxs * #stride;
        ptrs.add(idxs.cast())
          .read(Simd::<[m8; #step]>::splat(true),
                Default::default())
      }};
      let ty = SimdType::new(px, step);
      SimdValue::from(ty, t)
        .cast(calc_simd)
        .let_(dst, "t")
    };

    let run_filter = |src, filter, bits: &[_]| {
      let mut t = quote! {
        (#filter * #src).wrapping_sum() as i32
      };
      for bits in bits.iter() {
        t = quote!(round_shift(#t, #bits));
      }

      t
    };

    let args = quote! {(mut dst: *mut #px, dst_stride: i32,
                       mut src: *const #px, src_stride: i32,
                       #intermediate: &mut [i16],
                       col_frac: i32, row_frac: i32,
                       bit_depth: u8,
                       width: u16, height: u16, )};
    let mut func = self.new_mc_func(
      format_args!("put_8tap_internal_{}", b.fn_suffix()),
      args, vec![], true);

    let dst = Plane::new_named("dst", quote!(dst_stride));
    let src = Plane::new_named("src", quote!(src_stride));

    if px == PrimType::U8 {
      let v = ((1 << 8) - 1) as i32;
      func.extend(quote! {
        #[cfg(debug_assertions)]
        {
          if bit_depth != 8 {
            panic!("put_8tap expects 8bit depth on u8 pixels");
          }
        }
        let max_sample_val = #v;
        let intermediate_bits = 4;
      });
    } else {
      func.extend(quote! {
        let max_sample_val = ((1 << bit_depth) - 1) as i32;
        let intermediate_bits = 4 - if bit_depth == 12 { 2 } else { 0 };
      });
    }
    func.extend(quote! {
      let #x_filter = get_filter(#mode_x, col_frac, width as _);
      let #y_filter = get_filter(#mode_y, row_frac, height as _);
      let #x_filter = <#calc_simd> ::from(#x_filter);
      let #y_filter = <#calc_simd> ::from(#y_filter);
    });

    let mut bl = BlockLoop::std_u8();

    match self.frac {
      EightTapFrac(true, true) => {
        func.inline_always();

        bl.add_loop_var(src);
        bl.add_loop_var(dst);

        let step = b.w().min(self.px_simd_width());

        let px_simd = SimdType::new(px, step);
        bl.gen(&mut *func, 1, step as _, |out, _, _, vars| {
          let src = &vars[0];
          let dst = &vars[1];
          px_simd.uload(&src)
            .ustore(out, &dst);
        });
      }
      EightTapFrac(true, false) => {
        func.extend(quote! {
          //let src = src.go_up(3);
          src = src.offset((src_stride * -3) as isize);
        });

        bl.add_loop_var(src);
        bl.add_loop_var(dst);

        bl.gen(&mut *func, 1, 1, |out, _, _, vars| {
          let src = &vars[0];
          let dst = &vars[1];
          let t = load_src(out, px, src);
          let t = run_filter(t, &y_filter, &[quote!(7)]);
          let mut t = quote!(#t.max(0));
          if px == PrimType::U16 {
            t = quote!(#t.min(max_sample_val as _));
          }
          out.extend(quote! {
            *#dst = #t as #px;
          });
        });
      }
      EightTapFrac(false, true) => {
        func.extend(quote! {
          //let src = src.go_left(3);
          src = src.offset(-3);
        });

        bl.add_loop_var(src);
        bl.add_loop_var(dst);

        let bits = &[quote!(7 - intermediate_bits), quote!(intermediate_bits)];
        bl.gen(&mut *func, 1, 1, |out, _, _, vars| {
          let src = &vars[0];
          let dst = &vars[1];
          let t = load_unit_stride_src(out, px, src);
          let t = run_filter(t, &x_filter, bits);
          let mut t = quote!(#t.max(0));
          if px == PrimType::U16 {
            t = quote!(#t.min(max_sample_val as _));
          }
          out.extend(quote! {
            *#dst = #t as #px;
          });
        });
      }
      EightTapFrac(false, false) => {
        if width <= 8 {
          // Only case where `cg + c < width as i32` is true.
          func.inline_always();
        }

        func.extend(quote! {
          src = src.offset((src_stride * -3 - 3) as _);

          // stored in the "real" kernal function:
          //let mut intermediate: AlignedArray<[i16; 8 * (#height + 7)]> =
          //  AlignedArray::uninitialized();
          let mut iptr = intermediate.as_mut_ptr();
          let height = height as u16;
        });

        let iptr = Ident::new("iptr", Span::call_site());

        let mut cg_loop = Loop::new("cg", quote!(width),
                                    PrimType::U16);
        let cg_src = Plane::new_stride(src.name(),
                                       quote!(1i32));
        cg_loop.add_loop_var(cg_src);
        let cg_dst = Plane::new_stride(dst.name(),
                                       quote!(1i32));
        cg_loop.add_loop_var(cg_dst);
        let mut r_loop1 = Loop::new("r", quote!(height + 7),
                                    PrimType::U16);
        let mut r_loop2 = Loop::new("r", quote!(height),
                                    PrimType::U16);

        let iplane = Plane::new_stride(&iptr, quote!(1i32));
        r_loop1.add_loop_var(iplane.clone());
        r_loop2.add_loop_var(iplane);

        let c_loop = Loop::new("c", quote!(8.min(width)),
                               PrimType::U16);

        let bits1 = &[quote!(7 - intermediate_bits)];
        let bits2 = &[quote!(7 + intermediate_bits)];

        cg_loop.gen(&mut *func, 8, |out, _, vars| {
          let cg_src = &vars[0];
          let cg_dst = &vars[1];

          let src = Plane::new_stride(cg_src.name(),
                                      src.stride());
          let dst = Plane::new_stride(cg_dst.name(),
                                      dst.stride());
          r_loop1.add_loop_var(src.clone());

          let mut rc_loop = c_loop.clone();
          r_loop1.gen(out, 1, |out, _, vars| {
            let iptr = &vars[0];
            let src = &vars[1];

            let iplane = Plane::new_stride(iptr.name(),
                                           quote!(((height + 7) as i32)));
            rc_loop.add_loop_var(iplane);
            let src = Plane::new_stride(src.name(),
                                        quote!(1i32));
            rc_loop.add_loop_var(src);

            rc_loop.gen(out, 1, |out, _, vars| {
              let iptr = &vars[0];
              let src = &vars[1];

              let t = px_simd
                .uload(src)
                .cast(calc_simd)
                .let_(out, "t");

              let t = run_filter(t, &x_filter, bits1);
              let write = quote! {
                *#iptr = #t as i16;
              };
              if 8 > width {
                out.extend(quote! {
                  if cg + c < width as i32 {
                    #write
                  }
                });
              } else {
                out.extend(write);
              }
            });
          });

          r_loop2.add_loop_var(dst.clone());

          let mut rc_loop = c_loop.clone();
          r_loop2.gen(out, 1, |out, _, vars| {
            let iptr = &vars[0];
            let dst = &vars[1];

            let iplane = Plane::new_stride(iptr.name(),
                                           quote!(((height + 7) as i32)));
            rc_loop.add_loop_var(iplane);
            let dst = Plane::new_stride(dst.name(), quote!(1i32));
            rc_loop.add_loop_var(dst);

            rc_loop.gen(out, 1, |out, _, vars| {
              let iptr = &vars[0];
              let dst = &vars[1];

              let t = SimdType::new(PrimType::I16, step)
                .uload(iptr)
                .cast(calc_simd)
                .let_(out, "t");

              let t = run_filter(t, &y_filter, bits2);
              let mut t = quote!(#t.max(0));
              if px == PrimType::U16 {
                t = quote!(#t.min(max_sample_val as _));
              }
              let write = quote! {
                *#dst = #t as #px;
              };
              if 8 > width {
                out.extend(quote! {
                  if cg + c < width as i32 {
                    #write
                  }
                });
              } else {
                out.extend(write);
              }
            });
          });
        });
      }
    }

    func.to_tokens(self.self_funcs_mut()
      .out
      .as_mut()
      .unwrap());

    let entry =
      TableEntry::new_8tap_table_entry(self.isa,
                                       self.frac,
                                       self.bfm,
                                       b,
                                       func.path(),
                                       Ok(func));
    Self::insert_func(&mut self.self_funcs_mut().put_internal,
                      b, entry)
  }
  fn put_8tap(&mut self, b: Block) -> TableEntry {
    let entry = {
      self.self_funcs_mut()
        .put
        .get(&b)
        .map(|entry| entry.clone())
    };
    if let Some(entry) = entry {
      return entry;
    }

    let px = self.px;
    let args = quote! {(mut dst: *mut #px, dst_stride: i32,
                       mut src: *const #px, src_stride: i32,
                       col_frac: i32, row_frac: i32,
                       bit_depth: u8,
                       _width: u16, _height: u16, )};
    let mut func = self.new_mc_func(
      format_args!("put_8tap_{}", b.fn_suffix()),
      args, vec![], true);

    let width = b.w();
    let height = b.h();

    let internal = self.put_internal(b);
    let internal = &internal.path;

    let intermediate = if let EightTapFrac(false, false) = self.frac {
      // don't use stack space if this won't be used anyway.
      func.extend(quote! {
        let mut intermediate: AlignedArray<[i16; 8 * (#height + 7)]> =
          AlignedArray::uninitialized();
      });
      quote!(&mut *intermediate)
    } else {
      func.extend(quote! {
        let mut intermediate: [i16; 0] = [0; 0];
      });
      quote!(&mut intermediate)
    };

    func.extend(quote! {
      #internal(dst, dst_stride, src, src_stride, #intermediate,
                col_frac, row_frac, bit_depth, #width as _, #height as _)
    });

    func.to_tokens(self.self_funcs_mut()
      .out
      .as_mut()
      .unwrap());

    let entry =
      TableEntry::new_8tap_table_entry(self.isa,
                                       self.frac,
                                       self.bfm,
                                       b,
                                       func.path(),
                                       Ok(func));
    Self::insert_func(&mut self.self_funcs_mut().put,
                      b, entry)
  }
  fn prep_internal(&mut self, b: Block) -> TableEntry {
    let origin_b = b;
    let (b, bfm) = self.eight_tap_redir(origin_b)
      .unwrap_or((origin_b, self.bfm));
    let isa = self.isa;
    let frac = self.frac;
    let cur_bfm = self.bfm;

    let entry = {
      self.funcs_mut(bfm, self.frac)
        .prep_internal
        .get(&b)
        .map(|f| {
          TableEntry::new_8tap_table_entry(isa,
                                           frac,
                                           cur_bfm,
                                           b,
                                           f.path.clone(),
                                           Err(f.name.clone()))
        })
    };
    {
      let self_funcs = self.self_funcs_mut();
      if let Some(entry) = entry {
        let map_entry = self_funcs
          .prep_internal
          .entry(b);
        match map_entry {
          Entry::Occupied(o) => {
            return o.get().clone();
          },
          Entry::Vacant(v) => {
            return v.insert(entry)
              .clone();
          }
        }
      } else {
        assert!(self_funcs.out.is_some());
      }
    }

    // Note: width is only the lower bound of possible sizes
    let width = b.w();

    let calc_ty = PrimType::I32;
    let px = match self.px {
      PixelType::U8 => PrimType::U8,
      PixelType::U16 => PrimType::U16,
    };
    // the simd width:
    let step = 8usize;
    let calc_simd = SimdType::new(calc_ty, step);
    let px_simd = SimdType::new(px, step);

    let mut idxs = TokenStream::default();
    for i in 0..step {
      let i = i as i32;
      idxs.extend(quote! {
        #i,
      });
    }
    let idxs = quote!(Simd::<[i32; #step]>::new(#idxs));

    let mode_x = bfm.x;
    let mode_y = bfm.y;

    let x_filter = Ident::new("x_filter", Span::call_site());
    let y_filter = Ident::new("y_filter", Span::call_site());
    let intermediate = Ident::new("intermediate", Span::call_site());

    let load_unit_stride_src = |dst: &mut TokenStream, px, src: &Plane| {
      SimdType::new(px, step)
        .uload(src)
        .cast(calc_simd)
        .let_(dst, "t")
    };
    let load_src = |dst: &mut TokenStream, px, src: &Plane| {
      let stride = src.stride();
      let t = quote! {{
        let ptrs = Simd::<[*const #px; #step]>::splat(#src);
        let idxs = #idxs * #stride;
        ptrs.add(idxs.cast())
          .read(Simd::<[m8; #step]>::splat(true),
                Default::default())
      }};
      let ty = SimdType::new(px, step);
      SimdValue::from(ty, t)
        .cast(calc_simd)
        .let_(dst, "t")
    };

    let run_filter = |src, filter, bits: &[_]| {
      let mut t = quote! {
        (#filter * #src).wrapping_sum() as i32
      };
      for bits in bits.iter() {
        t = quote!(round_shift(#t, #bits));
      }

      t
    };

    let args = quote! {(mut tmp: *mut i16,
                       mut src: *const #px, src_stride: i32,
                       #intermediate: &mut [i16],
                       col_frac: i32, row_frac: i32,
                       bit_depth: u8,
                       width: u16, height: u16, )};
    let mut func = self.new_mc_func(
      format_args!("prep_8tap_internal_{}", b.fn_suffix()),
      args, vec![], true);

    let dst = Plane::new_named("tmp", quote!(width as i32));
    let src = Plane::new_named("src", quote!(src_stride));

    if px == PrimType::U8 {
      let v = ((1 << 8) - 1) as i32;
      func.extend(quote! {
        #[cfg(debug_assertions)]
        {
          if bit_depth != 8 {
            panic!("prep_8tap expects 8bit depth on u8 pixels");
          }
        }
        let max_sample_val = #v;
        let intermediate_bits = 4;
      });
    } else {
      func.extend(quote! {
        let max_sample_val = ((1 << bit_depth) - 1) as i32;
        let intermediate_bits = 4 - if bit_depth == 12 { 2 } else { 0 };
      });
    }
    func.extend(quote! {
      let #x_filter = get_filter(#mode_x, col_frac, width as _);
      let #y_filter = get_filter(#mode_y, row_frac, height as _);
      let #x_filter = <#calc_simd> ::from(#x_filter);
      let #y_filter = <#calc_simd> ::from(#y_filter);
    });

    let bits = &[quote!(7 - intermediate_bits)];

    let mut bl = BlockLoop::std_u8();

    match self.frac {
      EightTapFrac(true, true) => {
        func.inline_always();

        bl.add_loop_var(src);
        bl.add_loop_var(dst);

        let step = b.w().min(self.px_simd_width());

        let px_simd = SimdType::new(px, step);
        bl.gen(&mut *func, 1, step as _, |out, _, _, vars| {
          let src = &vars[0];
          let dst = &vars[1];
          px_simd.uload(&src)
            .shl(quote!(intermediate_bits))
            .astore(out, &dst);
        });
      }
      EightTapFrac(true, false) => {
        func.extend(quote! {
          //let src = src.go_up(3);
          src = src.offset((src_stride * -3) as isize);
        });

        bl.add_loop_var(src);
        bl.add_loop_var(dst);

        bl.gen(&mut *func, 1, 1, |out, _, _, vars| {
          let src = &vars[0];
          let dst = &vars[1];
          let t = load_src(out, px, src);
          let t = run_filter(t, &y_filter, bits);
          out.extend(quote! {
            *#dst = #t as i16;
          });
        });
      }
      EightTapFrac(false, true) => {
        func.extend(quote! {
          //let src = src.go_left(3);
          src = src.offset(-3);
        });

        bl.add_loop_var(src);
        bl.add_loop_var(dst);

        bl.gen(&mut *func, 1, 1, |out, _, _, vars| {
          let src = &vars[0];
          let dst = &vars[1];
          let t = load_unit_stride_src(out, px, src);
          let t = run_filter(t, &x_filter, bits);
          out.extend(quote! {
            *#dst = #t as i16;
          });
        });
      }
      EightTapFrac(false, false) => {
        if width <= 8 {
          // Only case where `cg + c < width as i32` is true.
          func.inline_always();
        }

        func.extend(quote! {
          src = src.offset((src_stride * -3 - 3) as _);

          // stored in the "real" kernal function:
          //let mut intermediate: AlignedArray<[i16; 8 * (#height + 7)]> =
          //  AlignedArray::uninitialized();
          let mut iptr = intermediate.as_mut_ptr();
          let height = height as u16;
        });

        let iptr = Ident::new("iptr", Span::call_site());

        let mut cg_loop = Loop::new("cg", quote!(width),
                                    PrimType::U16);
        let cg_src = Plane::new_stride(src.name(),
                                       quote!(1i32));
        cg_loop.add_loop_var(cg_src);
        let cg_dst = Plane::new_stride(dst.name(),
                                       quote!(1i32));
        cg_loop.add_loop_var(cg_dst);
        let mut r_loop1 = Loop::new("r", quote!(height + 7),
                                    PrimType::U16);
        let mut r_loop2 = Loop::new("r", quote!(height),
                                    PrimType::U16);

        let iplane = Plane::new_stride(&iptr, quote!(1i32));
        r_loop1.add_loop_var(iplane.clone());
        r_loop2.add_loop_var(iplane);

        let c_loop = Loop::new("c", quote!(8.min(width)),
                               PrimType::U16);

        let bits1 = bits;
        let bits2 = &[quote!(7)];

        cg_loop.gen(&mut *func, 8, |out, _, vars| {
          let cg_src = &vars[0];
          let cg_dst = &vars[1];

          let src = Plane::new_stride(cg_src.name(),
                                      src.stride());
          let dst = Plane::new_stride(cg_dst.name(),
                                      dst.stride());
          r_loop1.add_loop_var(src.clone());

          let mut rc_loop = c_loop.clone();
          r_loop1.gen(out, 1, |out, _, vars| {
            let iptr = &vars[0];
            let src = &vars[1];

            let iplane = Plane::new_stride(iptr.name(),
                                           quote!(((height + 7) as i32)));
            rc_loop.add_loop_var(iplane);
            let src = Plane::new_stride(src.name(),
                                        quote!(1i32));
            rc_loop.add_loop_var(src);

            rc_loop.gen(out, 1, |out, _, vars| {
              let iptr = &vars[0];
              let src = &vars[1];

              let t = px_simd
                .uload(src)
                .cast(calc_simd)
                .let_(out, "t");

              let t = run_filter(t, &x_filter, bits1);
              let write = quote! {
                *#iptr = #t as i16;
              };
              if 8 > width {
                out.extend(quote! {
                  if cg + c < width as i32 {
                    #write
                  }
                });
              } else {
                out.extend(write);
              }
            });
          });

          r_loop2.add_loop_var(dst.clone());

          let mut rc_loop = c_loop.clone();
          r_loop2.gen(out, 1, |out, _, vars| {
            let iptr = &vars[0];
            let dst = &vars[1];

            let iplane = Plane::new_stride(iptr.name(),
                                           quote!(((height + 7) as i32)));
            rc_loop.add_loop_var(iplane);
            let dst = Plane::new_stride(dst.name(), quote!(1i32));
            rc_loop.add_loop_var(dst);

            rc_loop.gen(out, 1, |out, _, vars| {
              let iptr = &vars[0];
              let dst = &vars[1];

              let t = SimdType::new(PrimType::I16, step)
                .uload(iptr)
                .cast(calc_simd)
                .let_(out, "t");

              let t = run_filter(t, &y_filter, bits2);
              let write = quote! {
                *#dst = #t as i16;
              };
              if 8 > width {
                out.extend(quote! {
                  if cg + c < width as i32 {
                    #write
                  }
                });
              } else {
                out.extend(write);
              }
            });
          });
        });
      }
    }

    func.to_tokens(self.self_funcs_mut()
      .out
      .as_mut()
      .unwrap());

    let entry =
      TableEntry::new_8tap_table_entry(self.isa,
                                       self.frac,
                                       self.bfm,
                                       b,
                                       func.path(),
                                       Ok(func));
    Self::insert_func(&mut self.self_funcs_mut().prep_internal,
                      b, entry)
  }
  fn prep_8tap(&mut self, b: Block) -> TableEntry {
    let entry = {
      self.self_funcs_mut()
        .prep
        .get(&b)
        .map(|entry| entry.clone())
    };
    if let Some(entry) = entry {
      return entry;
    }

    let px = self.px;
    let args = quote! {(mut tmp: *mut i16,
                       mut src: *const #px, src_stride: i32,
                       col_frac: i32, row_frac: i32,
                       bit_depth: u8,
                       _width: u16, _height: u16, )};
    let mut func = self.new_mc_func(
      format_args!("prep_8tap_{}", b.fn_suffix()),
      args, vec![], true);

    let width = b.w();
    let height = b.h();

    let internal = self.prep_internal(b);
    let internal = &internal.path;

    let intermediate = if let EightTapFrac(false, false) = self.frac {
      // don't use stack space if this won't be used anyway.
      func.extend(quote! {
        let mut intermediate: AlignedArray<[i16; 8 * (#height + 7)]> =
          AlignedArray::uninitialized();
      });
      quote!(&mut *intermediate)
    } else {
      func.extend(quote! {
        let mut intermediate: [i16; 0] = [0; 0];
      });
      quote!(&mut intermediate)
    };

    func.extend(quote! {
      #internal(tmp, src, src_stride, #intermediate,
                col_frac, row_frac, bit_depth, #width as _, #height as _)
    });

    func.to_tokens(self.self_funcs_mut()
      .out
      .as_mut()
      .unwrap());

    let entry =
      TableEntry::new_8tap_table_entry(self.isa,
                                       self.frac,
                                       self.bfm,
                                       b,
                                       func.path(),
                                       Ok(func));
    Self::insert_func(&mut self.self_funcs_mut().prep,
                      b, entry)
  }
}
impl<'a> CtxAvg<'a> {
  fn mc_avg_args(&self) -> TokenStream {
    let px = self.px;
    quote!{(
      dst: *mut #px,
      dst_stride: i32,
      tmp1: *const i16, tmp2: *const i16,
      bd: u8,
      width: u16,
      height: u16,
    )}
  }
  fn mc_avg_internal(&mut self, simd_lanes: usize) -> Ident {
    assert!(self.simd_width(PrimType::I32) >= simd_lanes);
    if let Some(f) = self.funcs.avg_internal.get(&simd_lanes) {
      return f.clone();
    }

    let load_simd = SimdType::new(PrimType::I16, simd_lanes);
    let calc_simd = SimdType::new(PrimType::I32, simd_lanes);
    let px_simd = SimdType::new(self.px.into(), simd_lanes);

    let mut func = self.new_func(format_args!("mc_avg_internal_{}", simd_lanes),
                                 self.mc_avg_args(),
                                 vec![], true);
    if self.px == PixelType::U8 {
      func.extend(quote! {
        debug_assert_eq!(1u32 << bd, 1u32 << 8,
                         "mc_avg expects 8bit depth on u8 pixels; got {}",
                         bd);
        let bd = 8;
      });
    }

    func.extend(quote! {
      let zero = Simd::<[i32; #simd_lanes]>::splat(0);
      let max_sample_val = ((1 << bd) - 1) as i32;
      let intermediate_bits = 4 - if bd == 12 { 2 } else { 0 };
    });

    let tmp1 = Plane::new_named("tmp1", quote!((width as i32)));
    let tmp2 = Plane::new_named("tmp2", quote!((width as i32)));
    let dst = Ident::new("dst", Span::call_site());
    let dst = Plane::new(&dst);

    let mut l = BlockLoop::std();
    l.add_loop_var(dst);
    l.add_loop_var(tmp1);
    l.add_loop_var(tmp2);

    l.gen(&mut *func, 1, simd_lanes as _, |body, _, _, vars| {
      let dst = &vars[0];
      let tmp1 = &vars[1];
      let tmp2 = &vars[2];

      let t1 = load_simd.aload(tmp1)
        .cast(calc_simd)
        .let_(body, "t1");
      let t2 = load_simd.aload(tmp2)
        .cast(calc_simd)
        .let_(body, "t2");
      let t = (&t1 + &t2)
        .let_(body, "t");
      let t = t.round_shift(quote!(intermediate_bits + 1))
        .let_(body, "t");
      let t = if self.px == PixelType::U8 {
        // here, we just need to ensure negative numbers
        // don't get wrapped to positive values.
        // XXX assumes u8 <=> 8-bit depth.
        t.max(&SimdValue::from(t.ty(), quote!(zero)))
      } else {
        t.clamp(quote!(0), quote!(max_sample_val))
      }
        .let_(body, "t");
      let t = t.cast(px_simd)
        .let_(body, "t");
      t.ustore(body, &dst);
    });

    func.to_tokens(&mut **self.out);

    assert!({
      self.funcs.avg_internal
        .insert(simd_lanes, func.name())
        .is_none()
    });
    func.name()
  }
  fn mc_avg(&mut self, b: Block) -> TableEntry {
    if let Some(f) = self.funcs.avg.get(&b) {
      return f.clone();
    }

    let simd_lanes = b.w().min(self.simd_width(PrimType::I32));
    let internal = self.mc_avg_internal(simd_lanes);
    let (name, path) = if self.hot_block(b) {
      let mut func = self.new_func(format_args!("mc_avg_{}", b.fn_suffix()),
                                   self.mc_avg_args(), vec![], true);
      Self::block_size_check(&mut *func, b);

      let w = b.w();
      let h = b.h();

      func.extend(quote! {
        #internal(dst, dst_stride, tmp1, tmp2, bd, #w as _, #h as _)
      });

      func.to_tokens(&mut **self.out);
      (func.name(), func.path())
    } else {
      let p = self.out.item_path(&internal);
      (internal, Rc::new(p))
    };

    let feature_idx = self.isa.index();
    let b_enum = b.table_idx();
    let idx = quote! {
      [#feature_idx][#b_enum]
    };
    let entry = TableEntry {
      indices: Rc::new(idx),
      name,
      path,
      func: None,
    };
    let out = entry.clone();
    assert!({
      self.funcs.avg
        .insert(b, entry)
        .is_none()
    });
    out
  }
}

pub fn put_8tap_kernels(file: &mut dyn Write) {
  write_prelude(file);

  let args = vec![
    (quote!(dst), quote!(*mut T)),
    (quote!(dst_stride), quote!(i32)),
    (quote!(src), quote!(*const T)),
    (quote!(src_stride), quote!(i32)),
    (quote!(col_frac), quote!(i32)),
    (quote!(row_frac), quote!(i32)),
    (quote!(bd), quote!(u8)),
    (quote!(_width), quote!(u16)),
    (quote!(_height), quote!(u16)),
  ];
  let ret = None;
  let table_array_sizes = vec![
    quote!(4),
    quote!(4),
    quote!(2),
    quote!(2),
  ];
  let mut kernels =
    KernelDispatch::new("Put8TapF", args, ret, "PUT_8TAP",
                        table_array_sizes);

  let mut native: BTreeMap<PixelType, PxFunctions> = BTreeMap::default();

  let from_crate_root = &["mc", "put_8tap"];
  for isa in IsaFeature::sets() {
    if IsaFeature::Sse2 < isa && isa < IsaFeature::Sse4_1 ||
      IsaFeature::Sse4_1 < isa && isa < IsaFeature::Avx2 {
      // no new instructions for us here.
      continue;
    }

    let mut isa_module = Module::new_root(from_crate_root,
                                          isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child(px.module_name());

      let mut ctx = Ctx8Tap {
        out: &mut px_module,
        px,
        isa,
        key: SetKey {
          kind: "put_8tap",
          bfm: BlockFilterMode::reg_reg(),
          frac: EightTapFrac(true, true),
        },

        native: &mut native,

        funcs: PxFunctions::default(),
      };

      for &frac in EightTapFrac::fracs().iter() {
        for bfm in BlockFilterMode::modes().into_iter() {
          ctx.key.frac = frac;
          ctx.key.bfm = bfm;

          for block in Block::blocks_iter() {
            let TableEntry {
              indices,
              path,
              ..
            } = ctx.put_8tap(block);
            kernels.push_kernel(px, indices, path);
          }
        }
      }

      for frac_funcs in ctx.funcs.values_mut() {
        let mut frac_module = frac_funcs.out.take().unwrap();
        for bfm_funcs in frac_funcs.funcs.values_mut() {
          let bfm_module = bfm_funcs.out.take().unwrap();
          bfm_module.finish_child(&mut frac_module);
        }
        frac_module.finish_child(ctx.out);
      }

      ctx.finish();

      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  println!("generated {} put_8tap kernels", kernels.len());

  let tables = kernels.tables();
  writeln!(file, "{}", tables).expect("write put_8tap kernel tables");
}
pub fn prep_8tap_kernels(file: &mut dyn Write) {
  write_prelude(file);

  let args = vec![
    (quote!(tmp), quote!(*mut i16)),
    (quote!(src), quote!(*const T)),
    (quote!(stride), quote!(i32)),
    (quote!(col_frac), quote!(i32)),
    (quote!(row_frac), quote!(i32)),
    (quote!(bd), quote!(u8)),
    (quote!(_width), quote!(u16)),
    (quote!(_height), quote!(u16)),
  ];
  let ret = None;
  let table_array_sizes = vec![
    quote!(4),
    quote!(4),
    quote!(2),
    quote!(2),
  ];
  let mut kernels =
    KernelDispatch::new("Prep8TapF", args, ret, "PREP_8TAP",
                        table_array_sizes);

  let mut native: BTreeMap<PixelType, PxFunctions> = BTreeMap::default();

  let from_crate_root = &["mc", "prep_8tap"];
  for isa in IsaFeature::sets() {
    if IsaFeature::Sse2 < isa && isa < IsaFeature::Sse4_1 ||
      IsaFeature::Sse4_1 < isa && isa < IsaFeature::Avx2 {
      // no new instructions for us here.
      continue;
    }
    let mut isa_module = Module::new_root(from_crate_root,
                                          isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child(px.module_name());

      let mut ctx = Ctx8Tap {
        out: &mut px_module,
        px,
        isa,
        key: SetKey {
          kind: "prep_8tap",
          bfm: BlockFilterMode::reg_reg(),
          frac: EightTapFrac(true, true),
        },

        native: &mut native,

        funcs: PxFunctions::default(),
      };

      for &frac in EightTapFrac::fracs().iter() {
        for bfm in BlockFilterMode::modes().into_iter() {
          ctx.key.frac = frac;
          ctx.key.bfm = bfm;

          for block in Block::blocks_iter() {
            let TableEntry {
              indices,
              path,
              ..
            } = ctx.prep_8tap(block);
            kernels.push_kernel(px, indices, path);
          }
        }
      }

      for frac_funcs in ctx.funcs.values_mut() {
        let mut frac_module = frac_funcs.out.take().unwrap();
        for bfm_funcs in frac_funcs.funcs.values_mut() {
          let bfm_module = bfm_funcs.out.take().unwrap();
          bfm_module.finish_child(&mut frac_module);
        }
        frac_module.finish_child(ctx.out);
      }

      ctx.finish();

      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  println!("generated {} prep_8tap kernels", kernels.len());

  let tables = kernels.tables();
  writeln!(file, "{}", tables).expect("write prep_8tap kernel tables");
}
pub fn mc_avg_kernels(file: &mut dyn Write) {
  write_prelude(file);

  let args = vec![
    (quote!(dst), quote!(*mut T)),
    (quote!(dst_stride), quote!(i32)),
    (quote!(tmp1), quote!(*const i16)),
    (quote!(tmp2), quote!(*const i16)),
    (quote!(bit_depth), quote!(u8)),
    (quote!(width), quote!(u16)),
    (quote!(height), quote!(u16)),
  ];
  let ret = None;
  let mut kernels = KernelDispatch::new("McAvgF", args, ret, "MC_AVG", vec![]);

  let mut native: BTreeMap<PixelType, AvgFunctions> = BTreeMap::default();

  let from_crate_root = &["mc", "mc_avg"];
  for isa in IsaFeature::sets() {
    if IsaFeature::Sse2 < isa && isa < IsaFeature::Sse4_1 ||
      IsaFeature::Sse4_1 < isa && isa < IsaFeature::Avx2 {
      // no new instructions for us here.
      continue;
    }
    let mut isa_module = Module::new_root(from_crate_root,
                                          isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child(px.module_name());
      StdImports.to_tokens(&mut px_module);

      let mut ctx = CtxAvg {
        out: &mut px_module,
        px,
        isa,
        key: (),

        native: &mut native,

        funcs: AvgFunctions::default(),
      };
      for block in Block::blocks_iter() {
        let entry = ctx.mc_avg(block);
        kernels.push_kernel(px, entry.idx(), entry.path());
      }
      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  println!("generated {} mc_avg kernels", kernels.len());

  let tables = kernels.tables();
  writeln!(file, "{}", tables).expect("write mc_avg kernel tables");
}
