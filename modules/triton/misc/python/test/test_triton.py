#!/usr/bin/env python
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests, unittest

# Two independent things are checked here, and they are deliberately NOT in the
# same test:
#
#   1. The binding exists and is callable, returns the requested shape, dtype and
#      channel count, and raises on what the contract rejects. This holds on ANY
#      build that has a Triton backend -- it needs no particular CPU build.
#   2. The numbers match CPU cv::resize of the PINNED reference configuration
#      (OpenCV 5.0.0 generic resize.cpp path, SSE3 baseline, no IPP, no custom
#      HAL). cv::resize of an arbitrary build is that reference only if the
#      build is configured so.
#
# An earlier version put both behind the reference-build check, so on an
# ordinary build EVERY python test skipped and a broken binding would have gone
# unnoticed. The C++ suite covers the numbers on any build against a portable
# oracle (test/test_resize.cpp); Python is here to prove the binding itself.

_F32_ATOL = {cv.INTER_NEAREST: 0.0, cv.INTER_LINEAR: 4.948e-07, cv.INTER_CUBIC: 1.401e-06, cv.INTER_AREA: 8.419e-07}


def _is_reference_build():
    info = cv.getBuildInformation()
    fields = {}
    for line in info.splitlines():
        k, sep, v = line.strip().partition(":")
        if sep and k.strip() not in fields:
            fields[k.strip()] = v.strip()
    return (fields.get("Baseline") == "SSE SSE2 SSE3" and fields.get("Custom HAL", "").startswith("NO")
            and fields.get("Intel IPP", "NO") in ("NO", ""))


class _TritonBase(NewOpenCVTests):
    def setUp(self):
        super(_TritonBase, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")
        if not cv.triton.haveBackend():
            self.skipTest("No Triton backend available")


class triton_hip_binding_test(NewOpenCVTests):
    """HIP queries work, and CUDA-only Python adapters raise instead of aborting."""

    def setUp(self):
        super(triton_hip_binding_test, self).setUp()
        if not hasattr(cv, "triton") or cv.triton.backendName() != "hip-jit":
            self.skipTest("requires the HIP Triton module")

    def test_gpumat_adapters_raise_safely(self):
        gpu = cv.cuda_GpuMat()
        matrix = np.array([[1., 0., 0.], [0., 1., 0.]], np.float64)
        for fn, args in ((cv.triton.resize, (gpu, (32, 24))),
                         (cv.triton.warpAffine, (gpu, matrix, (32, 24)))):
            for kwargs in ({}, {"stream": None}):
                with self.assertRaises(cv.error) as raised:
                    fn(*args, **kwargs)
                self.assertEqual(raised.exception.code, -213)  # StsNotImplemented


class triton_binding_test(_TritonBase):
    """The binding works. No reference configuration required."""

    def test_module_is_exposed(self):
        self.assertTrue(hasattr(cv, "triton"))
        self.assertIsInstance(cv.triton.backendName(), str)
        self.assertNotEqual(cv.triton.backendName(), "")

    def test_returns_requested_shape_dtype_and_channels(self):
        rng = np.random.RandomState(0)
        for channels in (1, 3, 4):
            shape = (60, 80) if channels == 1 else (60, 80, channels)
            # Both depths the contract accepts: a float32 upload/download path
            # that silently returned uint8 (or the wrong channel count) would
            # not have been visible from a uint8-only check.
            for img in (rng.randint(0, 256, shape).astype(np.uint8),
                        (rng.rand(*shape) * 4 - 2).astype(np.float32)):
                for dsize in ((40, 30), (120, 90)):
                    got = cv.triton.resize(cv.cuda_GpuMat(img), dsize,
                                           interpolation=cv.INTER_LINEAR).download()
                    expect = ((dsize[1], dsize[0]) if channels == 1
                              else (dsize[1], dsize[0], channels))
                    self.assertEqual(got.shape, expect)
                    self.assertEqual(got.dtype, img.dtype)

    def test_rejected_inputs_raise(self):
        u8 = np.zeros((32, 32, 3), np.uint8)
        for interp in (cv.INTER_LANCZOS4,):
            with self.assertRaises(cv.error):
                cv.triton.resize(cv.cuda_GpuMat(u8), (16, 16), interpolation=interp)


class triton_numeric_test(_TritonBase):
    """The numbers match the pinned CPU reference. Skips on other builds."""

    def setUp(self):
        super(triton_numeric_test, self).setUp()
        if not _is_reference_build():
            self.skipTest("cv.resize of this build is not the reference configuration; "
                          "the C++ suite compares against the portable oracle instead")

    def _check(self, np_mat, dsize, interp):
        src = cv.cuda_GpuMat(np_mat)
        got = cv.triton.resize(src, dsize, interpolation=interp).download()
        ref = cv.resize(np_mat, dsize, interpolation=interp)
        self.assertEqual(got.shape, ref.shape)
        d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        self.assertLessEqual(float(np.nanmax(d)), 0.0 if np_mat.dtype == np.uint8 else _F32_ATOL[interp])

    def test_resize_matches_cpu(self):
        rng = np.random.RandomState(0)
        for channels in (1, 3, 4):
            shape = (120, 160) if channels == 1 else (120, 160, channels)
            u8 = rng.randint(0, 256, shape).astype(np.uint8)
            f32 = (rng.rand(*shape) * 4 - 2).astype(np.float32)
            for dsize, modes in (((97, 71), (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA)),
                                 ((320, 240), (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA)),
                                 ((80, 60), (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA)),
                                 ((160, 120), (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA))):
                for interp in modes:
                    self._check(u8, dsize, interp)
                    self._check(f32, dsize, interp)


# cvtColor compares with this build's own CPU entry, not the resize portable
# oracle. Integer conversions and float copies compare exactly; finite float
# arithmetic uses the documented per-pixel input-channel scale bound.
_CVT_CODES = [
    (cv.COLOR_BGR2BGRA, 3, 4), (cv.COLOR_BGRA2BGR, 4, 3),
    (cv.COLOR_BGR2RGBA, 3, 4), (cv.COLOR_RGBA2BGR, 4, 3),
    (cv.COLOR_BGR2RGB, 3, 3), (cv.COLOR_BGRA2RGBA, 4, 4),
    (cv.COLOR_BGR2GRAY, 3, 1), (cv.COLOR_RGB2GRAY, 3, 1),
    (cv.COLOR_BGRA2GRAY, 4, 1), (cv.COLOR_RGBA2GRAY, 4, 1),
    (cv.COLOR_BGR2YCrCb, 3, 3), (cv.COLOR_RGB2YCrCb, 3, 3),
    (cv.COLOR_YCrCb2BGR, 3, 3), (cv.COLOR_YCrCb2RGB, 3, 3),
    (cv.COLOR_BGR2HSV, 3, 3), (cv.COLOR_RGB2HSV, 3, 3),
    (cv.COLOR_BGR2HSV_FULL, 3, 3), (cv.COLOR_RGB2HSV_FULL, 3, 3),
    (cv.COLOR_BGR2YUV, 3, 3), (cv.COLOR_RGB2YUV, 3, 3),
    (cv.COLOR_YUV2BGR, 3, 3), (cv.COLOR_YUV2RGB, 3, 3),
]


_CVT_F32_COPY = [
    (cv.COLOR_BGR2BGRA, 3, 4), (cv.COLOR_BGRA2BGR, 4, 3),
    (cv.COLOR_BGR2RGBA, 3, 4), (cv.COLOR_RGBA2BGR, 4, 3),
    (cv.COLOR_BGR2RGB, 3, 3), (cv.COLOR_BGRA2RGBA, 4, 4),
    (cv.COLOR_GRAY2BGR, 1, 3), (cv.COLOR_GRAY2BGRA, 1, 4),
]
_CVT_F32_ARITHMETIC = [
    (cv.COLOR_BGR2GRAY, 3, 1), (cv.COLOR_RGB2GRAY, 3, 1),
    (cv.COLOR_BGRA2GRAY, 4, 1), (cv.COLOR_RGBA2GRAY, 4, 1),
    (cv.COLOR_BGR2YCrCb, 3, 3), (cv.COLOR_RGB2YCrCb, 3, 3),
    (cv.COLOR_YCrCb2BGR, 3, 3), (cv.COLOR_YCrCb2RGB, 3, 3),
    (cv.COLOR_BGR2YUV, 3, 3), (cv.COLOR_RGB2YUV, 3, 3),
    (cv.COLOR_YUV2BGR, 3, 3), (cv.COLOR_YUV2RGB, 3, 3),
]


def _cvt_f32_allow(src, ref):
    scale = np.max(np.abs(src.astype(np.float64)), axis=2) if src.ndim == 3 else np.abs(src.astype(np.float64))
    if ref.ndim == 3:
        scale = scale[:, :, None]
    return (8.0 * 2.0 ** -24) * np.maximum(np.maximum(1.0, np.abs(ref.astype(np.float64))), scale)


class triton_cvtcolor_test(_TritonBase):
    """The cvtColor binding, against this build's own integer CPU path."""

    def test_matches_cpu_exactly(self):
        rng = np.random.RandomState(3)
        for code, scn, dcn in _CVT_CODES:
            for h, w in ((1, 1), (7, 13), (61, 97), (480, 640)):
                img = rng.randint(0, 256, (h, w, scn)).astype(np.uint8)
                got = cv.triton.cvtColor(cv.cuda_GpuMat(img), code).download()
                ref = cv.cvtColor(img, code)
                # Integer arithmetic on both sides: exact, not "within a tolerance".
                self.assertEqual((h, w, dcn) if dcn > 1 else (h, w), got.shape)
                self.assertEqual(got.dtype, np.uint8)
                self.assertEqual(int(np.abs(got.astype(np.int16) -
                                            ref.astype(np.int16)).max()), 0,
                                 msg="code " + str(code))

    def test_float_bound_does_not_borrow_another_pixels_scale(self):
        src = np.array([[[1.0, 1.0, 1.0], [65535.0, 65535.0, 65535.0]]], np.float32)
        ref = np.array([[1.0, 65535.0]], np.float32)
        wrong = ref.astype(np.float64)
        wrong[0, 0] += 1e-3
        self.assertGreater(abs(wrong[0, 0] - ref[0, 0]), _cvt_f32_allow(src, ref)[0, 0])
        # The former whole-image bound would incorrectly accept this small-pixel error.
        self.assertLess(abs(wrong[0, 0] - ref[0, 0]), (8.0 * 2.0 ** -24) * 65535.0)

    def test_all_float32_codes_with_per_pixel_scale(self):
        rng = np.random.RandomState(19)
        self.assertEqual(20, len({code for code, _, _ in _CVT_F32_COPY + _CVT_F32_ARITHMETIC}))
        for cases, exact in ((_CVT_F32_COPY, True), (_CVT_F32_ARITHMETIC, False)):
            for code, scn, dcn in cases:
                for w in (1, 7, 16, 17, 257):
                    shape = (3, w) if scn == 1 else (3, w, scn)
                    img = rng.uniform(-2.0, 2.0, shape).astype(np.float32)
                    img[:, ::3] *= np.float32(32767.5)
                    stream = cv.cuda_Stream()
                    src_gpu = cv.cuda_GpuMat(img)
                    got_gpu = cv.triton.cvtColor(src_gpu, code, stream=stream)
                    stream.waitForCompletion()
                    got = got_gpu.download()
                    ref = cv.cvtColor(img, code)
                    self.assertEqual(ref.shape, got.shape)
                    self.assertEqual(np.float32, got.dtype)
                    self.assertTrue(np.isfinite(got).all())
                    if exact:
                        np.testing.assert_array_equal(got, ref)
                    else:
                        self.assertTrue(np.all(np.abs(got.astype(np.float64) - ref) <= _cvt_f32_allow(img, ref)),
                                        msg="float32 code " + str(code))

    def test_malformed_video_and_legal_odd_extract(self):
        for code in (cv.COLOR_YUV2BGR_UYVY, cv.COLOR_YUV2RGB_UYVY,
                     cv.COLOR_YUV2BGR_YUY2, cv.COLOR_YUV2RGB_YUY2,
                     cv.COLOR_YUV2BGR_YVYU, cv.COLOR_YUV2RGB_YVYU):
            with self.assertRaises(cv.error):
                cv.triton.cvtColor(cv.cuda_GpuMat(np.zeros((1, 3, 2), np.uint8)), code)
        for shape in ((24, 31), (23, 32)):
            with self.assertRaises(cv.error):
                cv.triton.cvtColor(cv.cuda_GpuMat(np.zeros(shape, np.uint8)), cv.COLOR_YUV2BGR_NV12)
        img = np.arange(6, dtype=np.uint8).reshape(1, 3, 2)
        got = cv.triton.cvtColor(cv.cuda_GpuMat(img), cv.COLOR_YUV2GRAY_UYVY).download()
        np.testing.assert_array_equal(got, cv.cvtColor(img, cv.COLOR_YUV2GRAY_UYVY))

    def test_rejected_codes_raise(self):
        img = np.zeros((32, 32, 3), np.uint8)
        # The float32 conversions: refused by name rather than served under a tolerance.
        for code in (cv.COLOR_HSV2BGR, cv.COLOR_BGR2HLS):
            with self.assertRaises(cv.error):
                cv.triton.cvtColor(cv.cuda_GpuMat(img), code)
        # CV_32F is served only where the module has a stated contract: the bit-copy codes and
        # the three arithmetic families (colour->gray, YCrCb, YUV). Everything else at 32F is
        # refused. An earlier version of this test refused 32F BGR2GRAY outright, which stopped
        # being true once that family got its measured bound (contract-32f.md).
        f32 = np.random.RandomState(5).uniform(0.0, 1.0, (32, 32, 3)).astype(np.float32)
        for code in (cv.COLOR_BGR2XYZ, cv.COLOR_BGR2HSV, cv.COLOR_BGR2Lab):
            with self.assertRaises(cv.error):
                cv.triton.cvtColor(cv.cuda_GpuMat(f32), code)
        # ...and the arithmetic ones really are served, within the documented bound rather than
        # exactly: the CPU has several 32F answers (fused body W>=16, fused tail W<16, unfused
        # build), so equality is not the criterion here.
        got = cv.triton.cvtColor(cv.cuda_GpuMat(f32), cv.COLOR_BGR2GRAY).download()
        want = cv.cvtColor(f32, cv.COLOR_BGR2GRAY)
        self.assertEqual(got.dtype, np.float32)
        k = 8.0 * 2.0 ** -24
        allow = k * np.maximum(np.maximum(1.0, np.abs(want.astype(np.float64))),
                               np.abs(f32.astype(np.float64)).max(axis=2))
        self.assertLessEqual(float((np.abs(got.astype(np.float64) -
                                           want.astype(np.float64)) / allow).max()), 1.0)
        # cvtColor's YUV extraction wrapper is 8U-only even though extractChannel
        # alone accepts 32F. Preserve the CPU entry's refusal rather than extending it.
        f32_extract = np.ones((3, 7, 2), np.float32)
        for code in (cv.COLOR_YUV2GRAY_UYVY, cv.COLOR_YUV2GRAY_YUY2):
            with self.assertRaises(cv.error):
                cv.cvtColor(f32_extract, code)
            with self.assertRaises(cv.error):
                cv.triton.cvtColor(cv.cuda_GpuMat(f32_extract), code)
        # A source whose channel count does not match the code.
        with self.assertRaises(cv.error):
            cv.triton.cvtColor(cv.cuda_GpuMat(np.zeros((32, 32), np.uint8)),
                               cv.COLOR_BGR2GRAY)


# warpAffine (contract 0.4.3): bindings are checked on any usable build;
# numeric tests use the selected linked CPU reference. NN/LINEAR/AREA use
# the padded vector path; CUBIC uses the natural-width call. Fused arithmetic
# is calibrated from CPU outputs, not hardware feature flags. The unchanged
# per-element criteria are uint8 exact, uint16 <= 1 level, and float32
# min(5e-5 * max(abs(ref), 1.0), 1e-4).
_WA_F32_REL = 5.0e-5
_WA_F32_FLOOR = 1.0

_WA_GEOMS = [
    ("identity",  [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    ("subpixel",  [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]),
    ("rot_scale", [[0.9, -0.43589, 12.0], [0.43589, 0.9, -7.0]]),
    ("shear",     [[1.0, 0.35, 0.0], [0.0, 1.0, 0.0]]),
    ("negative",  [[-1.0, 0.0, 60.0], [0.0, -1.0, 40.0]]),
]


def _wa_within(got, ref):
    """The contract's per-element bound. Returns (elements outside, max normalised error)."""
    if ref.dtype in (np.uint8, np.uint16):
        d = np.abs(got.astype(np.int64) - ref.astype(np.int64))
        return int((d > (0 if ref.dtype == np.uint8 else 1)).sum()), float(d.max(initial=0))
    a, b = got.astype(np.float64), ref.astype(np.float64)
    d = np.abs(a - b)
    allowed = np.minimum(_WA_F32_REL * np.maximum(np.abs(b), _WA_F32_FLOOR), 1e-4)
    return int((~(d <= allowed)).sum()), float(np.max(d / allowed, initial=0.0))


class triton_warpaffine_binding_test(_TritonBase):
    """The binding works -- shape, dtype, channels, matrix types, output reuse, refusals."""

    def test_returns_requested_shape_dtype_and_channels(self):
        rng = np.random.RandomState(0)
        M = np.array(_WA_GEOMS[1][1], np.float64)
        for channels in (1, 3, 4):
            shape = (37, 53) if channels == 1 else (37, 53, channels)
            for img in (rng.randint(0, 256, shape).astype(np.uint8),
                        rng.randint(0, 65536, shape).astype(np.uint16),
                        (rng.rand(*shape) * 4 - 2).astype(np.float32)):
                for dsize in ((53, 37), (80, 60)):
                    got = cv.triton.warpAffine(cv.cuda_GpuMat(img), M, dsize).download()
                    expect = ((dsize[1], dsize[0]) if channels == 1
                              else (dsize[1], dsize[0], channels))
                    self.assertEqual(got.shape, expect)
                    self.assertEqual(got.dtype, img.dtype)

    def test_matrix_accepts_float32_and_float64(self):
        # cv::warpAffine takes either; the module narrows to double internally, so both must give
        # the same result. A binding that forwarded the Mat blindly would handle one of them.
        rng = np.random.RandomState(1)
        img = rng.randint(0, 256, (37, 53, 3)).astype(np.uint8)
        g = cv.cuda_GpuMat(img)
        ref = None
        for dt in (np.float32, np.float64):
            M = np.array(_WA_GEOMS[2][1], dt)
            got = cv.triton.warpAffine(g, M, (53, 37)).download()
            if ref is None:
                ref = got
            else:
                self.assertEqual(int(np.abs(got.astype(np.int16) -
                                            ref.astype(np.int16)).max()), 0)

    def test_border_value_is_a_scalar_per_channel(self):
        # A C3 border of (10, 20, 30) fills three channels; a single 77 fills channel 0 only,
        # exactly as cv::warpAffine expands Scalar(v) to (v, 0, 0, 0).
        img = np.zeros((9, 7, 3), np.uint8)
        M = np.array([[1.0, 0.0, -1000.0], [0.0, 1.0, -1000.0]])    # everything outside
        for bv, want in (((10, 20, 30), (10, 20, 30)), (77, (77, 0, 0))):
            got = cv.triton.warpAffine(cv.cuda_GpuMat(img), M, (7, 9),
                                       borderMode=cv.BORDER_CONSTANT, borderValue=bv).download()
            self.assertEqual(tuple(int(v) for v in got[0, 0]), want)

    def test_float32_border_keeps_fractional_precision(self):
        img = np.zeros((9, 7), np.float32)
        M = np.array([[1.0, 0.0, -1000.0], [0.0, 1.0, -1000.0]])
        for bv in (0.5, -1.75, 3.25):
            got = cv.triton.warpAffine(cv.cuda_GpuMat(img), M, (7, 9),
                                       borderMode=cv.BORDER_CONSTANT,
                                       borderValue=bv).download()
            self.assertAlmostEqual(float(got[0, 0]), bv, places=6)

    def test_destination_is_reused_when_supplied(self):
        rng = np.random.RandomState(2)
        img = rng.randint(0, 256, (37, 53, 3)).astype(np.uint8)
        M = np.array(_WA_GEOMS[2][1], np.float64)
        dst = cv.cuda_GpuMat((37, 53), cv.CV_8UC3)
        cv.triton.warpAffine(cv.cuda_GpuMat(img), M, (53, 37), dst=dst)
        first = dst.download()
        cv.triton.warpAffine(cv.cuda_GpuMat(img), M, (53, 37), dst=dst)
        self.assertEqual(int(np.abs(first.astype(np.int16) -
                                    dst.download().astype(np.int16)).max()), 0)

    def test_rejected_inputs_raise(self):
        u8 = np.zeros((32, 32, 3), np.uint8)
        M = np.eye(3, dtype=np.float64)[:2]
        # INTER_CUBIC became served in 0.4.0; INTER_AREA followed the CPU's own rewrite to
        # INTER_LINEAR (cv::warpAffine maps AREA to LINEAR before doing anything else, and the two
        # produce identical output). Only LANCZOS4 is refused now.
        with self.assertRaises(cv.error):
            cv.triton.warpAffine(cv.cuda_GpuMat(u8), M, (32, 32), flags=cv.INTER_LANCZOS4)
        # AREA is served AND equals LINEAR, checked rather than merely not raising.
        area = cv.triton.warpAffine(cv.cuda_GpuMat(u8), M, (32, 32),
                                    flags=cv.INTER_AREA).download()
        lin = cv.triton.warpAffine(cv.cuda_GpuMat(u8), M, (32, 32),
                                   flags=cv.INTER_LINEAR).download()
        self.assertEqual(int(np.abs(area.astype(np.int16) - lin.astype(np.int16)).max()), 0)
        # CV_16U is served; CV_16S is not.
        got = cv.triton.warpAffine(cv.cuda_GpuMat(np.zeros((32, 32, 3), np.uint16)),
                                   M, (32, 32)).download()
        self.assertEqual(got.dtype, np.uint16)
        with self.assertRaises(cv.error):
            cv.triton.warpAffine(cv.cuda_GpuMat(np.zeros((32, 32, 3), np.int16)), M, (32, 32))
        with self.assertRaises(cv.error):
            cv.triton.warpAffine(cv.cuda_GpuMat(np.zeros((32, 32, 2), np.uint8)), M, (32, 32))
        with self.assertRaises(cv.error):
            cv.triton.warpAffine(cv.cuda_GpuMat(u8), np.zeros((3, 3), np.float64), (32, 32))


class triton_warpaffine_numeric_test(_TritonBase):
    """The numbers match the selected CPU reference. Skips on other builds."""

    def setUp(self):
        super(triton_warpaffine_numeric_test, self).setUp()
        if not _is_reference_build():
            self.skipTest("cv.warpAffine of this build is not the reference configuration; "
                          "the C++ suite checks this build's selected CPU paths separately")

    def _check(self, img, M, dsize, flags, border, bv=0):
        g = cv.cuda_GpuMat(img)
        got = cv.triton.warpAffine(g, M, dsize, flags=flags, borderMode=border,
                                   borderValue=bv).download()
        ref = cv.warpAffine(img, M, dsize, flags=flags, borderMode=border, borderValue=bv)
        self.assertEqual(got.shape, ref.shape)
        nbad, worst = _wa_within(got, ref)
        # Against a NATURAL-width CPU call the two can differ at critical coordinates: under
        # NEAREST by an adjacent pixel's value, under LINEAR by one level. This is the
        # compatibility cost the contract states, so the test asserts the LINEAR bound and only
        # checks NEAREST on geometries where the CPU accepts the shared region -- using the
        # vector-path reference, which pads and crops, exactly as the C++ suite does.
        if (flags & cv.INTER_MAX) in (cv.INTER_NEAREST, cv.INTER_LINEAR):
            # The invoker-family paths have the body/tail split, so the reference is the
            # padded vector path. CUBIC does not: it keeps the natural-width ref above.
            wide = ((dsize[0] + 128 + 127) // 128) * 128
            ref_v = cv.warpAffine(img, M, (wide, dsize[1]), flags=flags,
                                  borderMode=border, borderValue=bv)[:, :dsize[0]]
            nbad, worst = _wa_within(got, ref_v)
        self.assertEqual(nbad, 0, msg="%d elements outside the contract, worst %.3f"
                                     % (nbad, worst))

    def test_matches_cpu(self):
        rng = np.random.RandomState(4)
        for name, m in _WA_GEOMS:
            M = np.array(m, np.float64)
            for channels in (1, 3, 4):
                shape = (37, 53) if channels == 1 else (37, 53, channels)
                for img in (rng.randint(0, 256, shape).astype(np.uint8),
                            rng.randint(0, 65536, shape).astype(np.uint16),
                            (rng.rand(*shape) * 4 - 2).astype(np.float32)):
                    for flags, border in ((cv.INTER_LINEAR, cv.BORDER_REPLICATE),
                                          (cv.INTER_NEAREST, cv.BORDER_CONSTANT),
                                          (cv.INTER_LINEAR, cv.BORDER_REFLECT_101),
                                          (cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                                           cv.BORDER_CONSTANT),
                                          (cv.INTER_CUBIC, cv.BORDER_CONSTANT),
                                          (cv.INTER_CUBIC, cv.BORDER_REFLECT_101)):
                        self._check(img, M, (53, 37), flags, border, 7)

    def test_matches_cpu_larger_destination(self):
        rng = np.random.RandomState(5)
        img = rng.randint(0, 256, (37, 53, 3)).astype(np.uint8)
        for name, m in _WA_GEOMS:
            self._check(img, np.array(m, np.float64), (120, 90),
                        cv.INTER_LINEAR, cv.BORDER_CONSTANT, 3)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
