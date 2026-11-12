Triton warpAffine {#tutorial_triton_warpaffine}
=============

@note PROPOSED optional module. The module name, public API and build switches
are part of a proposal under review and may change. For a quick start see the
module's `README.md`; this page is the full usage guide.

@see @ref triton "Triton API reference".

What it computes
----------------

Selected CPU vector-path semantics for `INTER_NEAREST` and `INTER_LINEAR`:
OpenCV's CPU `cv::warpAffine` as produced by its SIMD body (OpenCV 5.0.0: float32
coordinates and float32 weights, one `saturate_cast<uchar|ushort>` for `CV_8U` /
`CV_16U`), and the **natural-width** call for `INTER_CUBIC`, which is a different
code path with no width dependence (see below).

`cv::cuda::warpAffine` computes something else -- it still runs the older
fixed-point remap tables -- so it is a performance baseline only (`perf/`),
never a correctness reference, and this module is **not a drop-in replacement
for `cv::cuda::warpAffine`**.

Which CPU path is the reference, and why that question exists
-------------------------------------------------------------

For fixed `(src, M, dsize, flags)` this build is deterministic. What is **not**
stable is agreement between two *destination widths* on the region they share.
In `imgproc/src/warp_kernels.simd.hpp` the SIMD body computes the source
coordinate of a column as

    v_fma(M0, x, (float)(y*M1 + M2))

while the scalar tail -- whatever is left past the vector width -- computes

    x*M0 + y*M1 + M2

One ulp apart. Since the tail's length is a function of the destination width,
different widths apply those two expressions to different sets of columns.

How large the disagreement is depends on the interpolation, and the two cases are
not the same:

* **NEAREST** rounds the coordinate to pick one pixel. An ulp can flip that
  rounding at a .5 tie, selecting the **adjacent source pixel**, whose value is
  unrelated -- so the difference is unbounded. Measured worst 239 of 255 with
  `M=[[1,1.9,-20],[0.4,1,-10]]` over a 37x53 source, up to 3 pixels per case,
  240 of 320 (seed, width) combinations affected. Destination width 40 gives 193
  where widths 48, 64 and 128 give 65.
* **LINEAR** keeps both taps and reweights them. An ulp in the coordinate moves
  the fractional weight by ~1e-7, so the interpolated float moves by about one
  ulp of itself, and the uint8 rounding absorbs nearly all of it: measured worst
  **1** level, up to 4 pixels per case, 157 of 320 combinations affected.

Reproducing that would mean reproducing which columns land in the tail, i.e.
depending on a property of the caller's allocation that has nothing to do with
what was asked. This module therefore defines its reference as the **vector
path**: `cv::warpAffine` evaluated at a destination width rounded up to a
multiple of 128 (past any SIMD width, so every compared column is in the body),
then cropped. That is a pure function of `(src, M, dsize, flags)`, and it is
stable: padding by 128 and by 256 give identical crops.

Why 128 suffices is an argument about the code, not only a measurement: the body
evaluates `v_fma(M0, x, Mx)` per lane with `x` the lane's own column, so a given
column gets the same float on SSE, AVX2 and AVX-512. Only the tail's *length* is
ISA-dependent, and it is at most 32 lanes with an unroll factor up to 64 on any
`CV_SIMD` build. Padding past that puts every compared column in the body, which
is what makes the reference ISA-independent.

Against a **natural-width** call the difference is confined to columns whose
sampled coordinate is critical, and its size depends on the interpolation: for
NEAREST it is an adjacent source pixel's value (unbounded, measured up to 239 of
255), for LINEAR at most 1 level. Neither is covered by the contract's
tolerances, which apply to the kernel against the *selected* reference only.

The same class of defect is why this module refuses `HSV2BGR`: there the CPU's
SIMD body and scalar tail disagree on the rounding of a whole conversion, and an
implementation cannot be "CPU-exact" against a reference that is not
self-consistent.

### `INTER_CUBIC` is the exception, and it is measured

`cv::warpAffine` does not route `INTER_CUBIC` through the invoker family above.
It goes through `genericWarp()` -> `getBicubicWarpFunc_()` ->
`bicubicVec`/`bicubicRef` (`imgwarp.cpp`, `warp_kernels.simd.hpp`), which
computes a destination column's coordinate as

    M_x + M(0,0)*x        with  M_x = float(y*M(1) + M(2))

from the **absolute** `x`, and with no `fma`. There is no SIMD-body/scalar-tail
split, so there is nothing for the padding rule to remove: over 80 seeds x 4
widths, **0 of 320** (seed, width) pairs differ for `CV_8U`, `CV_16U` and
`CV_32F`. Padding the destination would change nothing.

The CUBIC reference in this module is therefore the **natural-width call**, not
the padded one. Applying the vector-path rule here would have hidden the fact
that this path has no width dependence, and it would have implied a compatibility
cost that does not exist. `Triton_WarpAffine.CpuCubicIsWidthInvariant` measures
the property, including `CV_16U`, and uses NEAREST as the contrast.

CUBIC's arithmetic, all from the CPU source: the coordinate is clamped in float
to `[-max(dim,16), 2*max(dim,16)]` before the floor; `ix0 = floor(sx) - 1` and 16
taps follow; the weights use `A = -0.75` (`w0 = A*(a*b^2)`, `w3 = A*(a^2*b)`,
`w1 = a^2*((A+2)*a - (A+3)) + 1`, `w2 = 1 - w0 - w1 - w3`); each tap accumulates
as `fma(v, wx, rowsum)` and each row as `fma(rowsum, wy, acc)`. A whole 4x4
neighbourhood outside the source under `BORDER_CONSTANT` takes the border value.

`CV_16U` is ordinary by comparison: `warpAffine{Nearest,Linear}Invoker_16UC*` run
the same coordinate chain and the same fma interpolation as the `CV_8U` ones, load
`uint16`, convert to float32, and store with `v_pack_u(v_round(...))`, i.e.
`saturate_cast<ushort>` (round, then clamp to `[0,65535]`). It inherits the
vector-path rule for NEAREST/LINEAR, and its width sensitivity is the same shape:
NEAREST up to an adjacent pixel (measured worst 59,677 of 65,535), LINEAR at most
1 level.

Accuracy contract
-----------------

| depth | bound | meaning |
|---|---|---|
| `CV_8U` | bit-exact | `got == ref` on every element |
| `CV_16U` | `max_abs` 1 | `|got - ref| <= 1` on every element |
| `CV_32F` | relative with absolute cap | `|got - ref| <= min(5e-5 * max(|ref|, 1.0), 1e-4)` on every element |

The float32 bound is **per element, with a hard absolute ceiling of 1e-4**;
the relative component can only make that limit stricter. Float16 is not supported.
It is never normalised by the image maximum: an image maximum lets a large
element hide a small one's error, and a formula that does that is not a contract.
The floor of 1.0 is what `|ref|` looks like near zero, where a purely relative
bound cannot be met in float32.

Passing the bound is not the same as being bit-identical, so the accuracy test
reports, per dtype: elements compared, elements not bit-identical, elements
outside the bound, max |diff|, max normalised error (difference over bound), and
max ULP of the reference (computed from the reference's own `nextafterf`
spacing). Against the selected reference this module currently reports 0
non-identical elements for both dtypes; those figures come from the run in
`runs/warpaffine-0001/logs/`, not from this page.

`CV_8U` is not claimed bit-exact against a natural-width CPU call (the tie issue
above, where the difference is an adjacent pixel's value, not one level). Against
the selected reference it must now match exactly; one-level differences fail.

Supported subset (this milestone)
---------------------------------

| dimension | supported |
|---|---|
| depth | `CV_8U`, `CV_16U`, `CV_32F` |
| channels | 1, 3, 4 |
| interpolation | `INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC` |
| border modes | `BORDER_CONSTANT`, `BORDER_REPLICATE`, `BORDER_REFLECT`, `BORDER_REFLECT_101`, `BORDER_WRAP` |
| map direction | forward, and `WARP_INVERSE_MAP` |
| `borderValue` | a `Scalar` of up to four components, one per channel; kept as `double` to the kernel, so a `CV_32F` border of 0.5 is 0.5 |
| layout | ROI source/destination, padded row stride, 1-pixel axes (sampled with `BORDER_REPLICATE` as on the CPU), output reuse |
| platform | Linux, CUDA (driver API), `sm_90` tested |

`INTER_LANCZOS4` raises `cv::Error::StsUnsupportedFormat`, and so do other depths
and channel counts, and overlapping input/output. LANCZOS4's 8x8 weight table and
clamped tap neighbourhood are a separate contract; approximating them behind an
unmeasured tolerance is not acceptable. There is **no implicit fallback**: a call
that returns has submitted the computation.

`CV_16U` saturates at the store, and so does the border value: a `borderValue` of
40000 survives, and a fractional border is rounded rather than truncated
(`77.6` becomes `78`, which is what `saturate_cast` does).

How a call runs
---------------

1. The matrix is converted to `double` and inverted by `cv::warpAffine`'s
   closed form unless `WARP_INVERSE_MAP` is set; the request is validated
   (types, steps, int32 index range, overlap).
2. The compiled kernel for the request's compile key (kernel, signature
   dtypes, the constexprs, launch options, target, Triton version, kernel
   source hash) is taken from an in-process cache, else from the on-disk cache
   after verifying its manifest and binary hash, else compiled by
   `tools/compile_helper.py` in a separate, Torch-free process. The six matrix
   entries and the four border components are **runtime** arguments, never
   constexprs: a matrix or border value must not cause a new compilation, and
   a border value must not be narrowed to an integer.
3. The kernel is launched on the caller's stream, with one program per
   `(BLOCK_H rows, BLOCK_W pixels)` tile of the destination. The tile shape is
   chosen per interpolation mode and is part of the compile key, so the two
   shapes are separate artifacts rather than one kernel branched at run time:
   `NEAREST` uses `1x256`, while `LINEAR` and `CUBIC` use `2x64`.

   On a rotating geometry a squarer tile measured 1.10-1.17x faster at 1080p and
   4K, and `NEAREST` measured 1.23-1.42x faster on a wider row. A plain translate
   was unchanged in both cases. The suggested explanation -- not a profiled
   result -- is that a rotation maps an output row onto a diagonal, so 128 lanes
   of one row touch on the order of 56 source rows, and a squarer tile halves
   that spread while a single-tap kernel has little spread to reduce. Treat the
   explanation as a hypothesis and the timings as measurements on one device
   class; neither is a cross-platform promise. The tile does not affect the
   accuracy contract described in the module README.
4. A completion event orders the work against later reads on other streams.
