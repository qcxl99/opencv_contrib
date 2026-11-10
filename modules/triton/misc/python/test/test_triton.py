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

_F32_ATOL = {cv.INTER_NEAREST: 0.0, cv.INTER_LINEAR: 4.948e-07, cv.INTER_AREA: 8.419e-07}


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
        for interp in (cv.INTER_LANCZOS4, cv.INTER_CUBIC):
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
        got = cv.triton.resize(cv.cuda_GpuMat(np_mat), dsize, interpolation=interp).download()
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
            for dsize, modes in (((97, 71), (cv.INTER_NEAREST, cv.INTER_LINEAR)),
                                 ((320, 240), (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_AREA)),
                                 ((80, 60), (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_AREA)),
                                 ((160, 120), (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_AREA))):
                for interp in modes:
                    self._check(u8, dsize, interp)
                    self._check(f32, dsize, interp)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
