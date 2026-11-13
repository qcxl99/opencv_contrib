Triton warpAffine {#tutorial_triton_warpaffine}
=============

@note PROPOSED optional module. The module name, public API and build switches
are part of a proposal under review and may change. For a quick start see the
module's `README.md`; this page is the full usage guide.

@see @ref triton "Triton API reference".
@see @ref tutorial_triton_h20_first_use "H20 build, install and first-use walkthrough".

What it computes
----------------

The reference is the linked OpenCV CPU build's **selected vector path** for
`INTER_NEAREST` and `INTER_LINEAR` (including the `INTER_AREA` alias), and its
**natural-width** `INTER_CUBIC` path. It is not a portable promise about every
OpenCV CPU build. `cv::cuda::warpAffine` is a timing baseline, not the numerical
reference; this module is not a drop-in replacement for that implementation.

Why the reference path matters
-----------------------------

For fixed input, matrix, output size and flags, the selected CPU build is
deterministic. Its NN/LINEAR SIMD body and scalar tail can nevertheless compute
slightly different coordinates in the overlapping region of two output widths:

    vector: v_fma(M0, x, float(y*M1 + M2))
    scalar: x*M0 + y*M1 + M2

An ulp near a half-integer may select the adjacent source pixel under NEAREST,
whose value can differ over the full pixel range. There is no small-error bound
against a natural-width call. LINEAR also may differ; the tolerance below is
against the **selected reference**, not a guarantee for all natural-width calls.

The test/reference helper computes NN/LINEAR/AREA at a padded destination width
and crops it, keeping compared columns in the selected vector body. Padding
removes the identified tail-boundary dependence on this linked build; it does
not establish ISA independence. Compiler contraction, SIMD lowering and CPU
dispatch can change the reference and must be validated for another build.

The module identifies fused versus separate multiply/add using fixed probes of
the linked CPU implementation, separately by depth/channel/useOptimized state.
It does not infer the arithmetic from hardware FMA support or the module's
compiler flags. This bounded first-use identification does not compute the
user's output on the CPU. Unidentifiable reference arithmetic is refused.

CUBIC uses a separate CPU implementation
---------------------------------------

The linked `cv::warpAffine` routes CUBIC through genericWarp()/bicubicVec/bicubicRef
in imgwarp.cpp/warp_kernels.simd.hpp. Coordinates use absolute destination x:

    M_x + M(0,0)*x, with M_x = float(y*M(1) + M(2))

The reference uses the natural-width CPU call. The module tests check width
invariance on that build; this is not a cross-build guarantee. The coordinate is
clamped to `[-max(dim,16), 2*max(dim,16)]` before floor, followed by a 4x4 tap
window and cubic weights with A=-0.75. Each row/tap preserves the selected
reference arithmetic. A wholly outside window under CONSTANT uses borderValue.

The 16U path follows the selected reference's coordinate/interpolation rules,
loads uint16 and rounds/saturates the result into `[0,65535]`. It uses the same
reference distinction and the depth-specific bound below.

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
spacing). Actual statistics belong to the results for the tested source/build;
passing a tolerance does not establish bit identity.

`CV_8U` is not claimed bit-exact against a natural-width CPU call (the tie issue
above, where the difference is an adjacent pixel's value, not one level). Against
the selected reference it must now match exactly; one-level differences fail.

Supported subset
---------------------------------

| dimension | supported |
|---|---|
| depth | `CV_8U`, `CV_16U`, `CV_32F` |
| channels | 1, 3, 4 |
| interpolation | `INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC`; `INTER_AREA` aliases LINEAR |
| border modes | `BORDER_CONSTANT`, `BORDER_REPLICATE`, `BORDER_REFLECT`, `BORDER_REFLECT_101`, `BORDER_WRAP`; `BORDER_TRANSPARENT` for NEAREST/LINEAR/AREA |
| map direction | forward, and `WARP_INVERSE_MAP` |
| `borderValue` | a `Scalar` of up to four components, one per channel; integer depths round and saturate it, while a `CV_32F` border of 0.5 stays 0.5 |
| layout | ROI source/destination, padded row stride, 1-pixel axes (sampled with `BORDER_REPLICATE` as on the CPU), output reuse |
| platform | Linux; CUDA GpuMat/device buffers and HIP device buffers (see module build instructions) |

`INTER_AREA` is normalized to LINEAR, as in CPU `warpAffine`; it is not the
area-integration algorithm used by `resize`. Both flags use the same normalized
mode and compile key.

`BORDER_TRANSPARENT` uses the destination's existing value. For NEAREST, an
out-of-range rounded source point leaves that destination pixel unchanged.
For LINEAR/AREA, each out-of-range tap is replaced with the old destination
pixel and the original interpolation weights are retained; if the whole 2x2
neighbourhood is outside, the pixel stays unchanged. This is destination-fill
blending, not weight renormalization or a requirement that every tap be inside.
Supply an initialized destination of the required size/type: a newly allocated
output has no defined background. `borderValue` does not provide that background.
`INTER_CUBIC` with `BORDER_TRANSPARENT` is explicitly refused.

The public module GpuMat and device-buffer entries serve these routes.

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

1. Shared entry resolution validates source depth/channels, flags, matrix and
   border options before output allocation. It resolves the requested destination
   size and checks its sampling expressions through a pure output-admission API;
   this step performs no device work or CPU arithmetic calibration. The matrix is
   inverted with the CPU closed form unless `WARP_INVERSE_MAP` is set. After
   allocation, the planner validates the actual destination type, pitch, allocation
   span, overlap and context, and identifies the linked CPU arithmetic path.
2. The compiled kernel for the request's compile key (kernel, signature
   dtypes, the constexprs, launch options, target, Triton version, kernel
   source hash) is taken from an in-process cache, else from the on-disk cache
   after verifying its manifest and binary hash, else compiled by
   `tools/compile_helper.py` in a separate, Torch-free process. The six matrix
   entries and the four border components are **runtime** arguments, never
   constexprs: a matrix or border value must not cause a new compilation, and
   a border value must not be narrowed to an integer.
3. The kernel is launched on the caller's stream, with one program per
   `(BLOCK_H rows, BLOCK_W pixels)` tile of the destination.
4. A completion event orders the work against later reads on other streams.


### CPU arithmetic and HIP launch tiles

The vendored kernels include the current `FUSED` constexpr for both mapped
coordinates and interpolation accumulation. The native planner observes the
linked OpenCV CPU vector path at rounding-sensitive coordinates for each
dtype, channel count and `cv::useOptimized()` state, instead
of inferring that path solely from CPU hardware features. A reference outside
both supported arithmetic forms is refused. The selected value participates
in the compile key, so fused and split binaries cannot alias in the cache.
The `TRANSPARENT` constexpr also participates in the key, separately from the
normalized interpolation mode; matrices and border values remain runtime inputs.

Integer border components are rounded and saturated to the destination depth;
floating components keep their precision. Unknown flags, nonfinite matrix entries and sampling matrices outside finite
float32 range are refused before device work. Finite matrix entries alone do
not guarantee representable coordinates: the planner bounds the absolute affine
terms over the destination rectangle, rejects float32 intermediate overflow, and
requires NEAREST/LINEAR/AREA coordinates to fit the safe int32 sampling domain.
This conservative bound can reject maps whose large terms cancel. CUBIC clamps
finite float32 coordinates before int32 conversion. Float32 border components
must also fit finite float32. HIP uses measured launch shapes with
small/narrow-image and C3 NEAREST fallback conditions. CUDA uses `1×256` tiles
for NEAREST and `2×64` for LINEAR/CUBIC (including the AREA alias). These are
tuning choices, not changes to interpolation meaning.

Python `resize`/`warpAffine` use a nullable stream bridge so a HIP build raises
a catchable error for its unavailable GpuMat adapter. HIP numerical examples use
`opencv2/triton/device.hpp`; the CUDA Python call shape remains available.

### Performance coverage

`perf/perf_warpaffine.cpp` keeps opaque NEAREST/LINEAR/CUBIC comparisons with
native CUDA separate from the AREA/TRANSPARENT group. The latter declares cases for 1080p
and 4K rotated sampling, uint8 C3, uint16 C1 and float32 C1, with REPLICATE and
TRANSPARENT borders. Its destination is initialized before warmup and reused
through the timed loop, so reading and updating previous destination values is
part of the measured workload. These rows provide no CUDA speedup claim for
transparent semantics. They are warmed public-call timings, not isolated kernel
or graph-replay measurements.
