#!/usr/bin/env python
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import cv2 as cv
import numpy as np

import test_triton as common

from tests_common import NewOpenCVTests, unittest


class triton_hip_device_test(NewOpenCVTests):
    """Owned HIP storage, Python conversions and asynchronous device operations."""

    def setUp(self):
        super(triton_hip_device_test, self).setUp()
        if not hasattr(cv, "triton") or cv.triton.backendName() != "hip-jit":
            self.skipTest("requires the HIP Triton module")
        if not cv.triton.haveBackend():
            self.skipTest("no usable HIP device")
        # Missing bindings on a usable backend are failures, not skipped tests.
        self.assertTrue(hasattr(cv.triton, "HipImage"))
        self.assertTrue(hasattr(cv.triton, "HipStream"))

    @staticmethod
    def _image(host):
        result = cv.triton.HipImage()
        result.upload(host)
        return result

    @staticmethod
    def _nearest(host, size):
        width, height = size
        y = np.arange(height, dtype=np.int64) * host.shape[0] // height
        x = np.arange(width, dtype=np.int64) * host.shape[1] // width
        return host[y[:, None], x[None, :]]

    def test_upload_download_metadata_and_strided_host(self):
        depths = ((np.uint8, cv.CV_8U), (np.uint16, cv.CV_16U),
                  (np.float32, cv.CV_32F))
        for dtype, depth in depths:
            for channels in (1, 2, 3, 4):
                with self.subTest(dtype=dtype, channels=channels):
                    shape = (11, 17) if channels == 1 else (11, 17, channels)
                    values = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
                    # Exercise both a positive-stride ROI and a negative-stride view.
                    host = values[1:10:2, 2:16:2][::-1]
                    image = cv.triton.HipImage()
                    self.assertTrue(image.empty())
                    self.assertEqual(image.step(), 0)
                    image.upload(host)
                    self.assertFalse(image.empty())
                    self.assertEqual(image.size(), (host.shape[1], host.shape[0]))
                    self.assertEqual(image.device(), 0)
                    self.assertEqual(image.depth(), depth)
                    self.assertEqual(image.channels(), channels)
                    self.assertEqual(image.type(), getattr(cv, {cv.CV_8U: "CV_8UC",
                                      cv.CV_16U: "CV_16UC", cv.CV_32F: "CV_32FC"}[depth] + str(channels)))
                    self.assertGreaterEqual(image.step(), host.shape[1] * channels * host.itemsize)
                    np.testing.assert_array_equal(image.download(), host)
                    # OutputArray also accepts a caller-provided NumPy destination.
                    output = np.empty_like(host)
                    image.download(output)
                    np.testing.assert_array_equal(output, host)

    def test_roi_alias_upload_and_parent_lifetime(self):
        host = np.arange(13 * 19 * 3, dtype=np.uint16).reshape(13, 19, 3)
        parent = self._image(host)
        roi = parent.roi((3, 2, 7, 5))
        self.assertEqual(roi.step(), parent.step())
        replacement = np.full((5, 7, 3), 1234, np.uint16)
        roi.upload(replacement)
        expected = host.copy()
        expected[2:7, 3:10] = replacement
        np.testing.assert_array_equal(parent.download(), expected)
        del parent
        gc.collect()
        np.testing.assert_array_equal(roi.download(), replacement)

    def test_resize_and_explicit_destination_reuse(self):
        host = np.arange(19 * 29 * 3, dtype=np.uint8).reshape(19, 29, 3)
        src = self._image(host)
        destination = self._image(np.zeros((11, 17, 3), np.uint8))
        returned = cv.triton.resize(src, (17, 11), dst=destination,
                                   interpolation=cv.INTER_NEAREST)
        expected = self._nearest(host, (17, 11))
        np.testing.assert_array_equal(returned.download(), expected)
        np.testing.assert_array_equal(destination.download(), expected)
        # A mutation through the result must also be visible through the supplied dst.
        returned.roi((0, 0, 1, 1)).upload(np.full((1, 1, 3), 83, np.uint8))
        self.assertTrue(np.all(destination.download()[0, 0] == 83))
        constant = self._image(np.full((12, 18), 47, np.float32))
        scaled = cv.triton.resize(constant, (0, 0), fx=0.5, fy=0.5,
                                 interpolation=cv.INTER_LINEAR)
        np.testing.assert_array_equal(scaled.download(), np.full((6, 9), 47, np.float32))

    def test_packed_colour_and_device_pipeline(self):
        rng = np.random.RandomState(41)
        host = rng.randint(0, 256, (23, 31, 3)).astype(np.uint8)
        src = self._image(host)
        packed = cv.triton.cvtColor(src, cv.COLOR_BGR2BGR565)
        np.testing.assert_array_equal(packed.download(), cv.cvtColor(host, cv.COLOR_BGR2BGR565))
        unpacked = cv.triton.cvtColor(packed, cv.COLOR_BGR5652BGR)
        np.testing.assert_array_equal(unpacked.download(),
                                      cv.cvtColor(cv.cvtColor(host, cv.COLOR_BGR2BGR565),
                                                  cv.COLOR_BGR5652BGR))
        for dtype in (np.uint16, np.float32):
            values = host[:, :, 0].astype(dtype)
            converted = cv.triton.cvtColor(self._image(values), cv.COLOR_GRAY2BGR)
            np.testing.assert_array_equal(converted.download(), np.repeat(values[:, :, None], 3, axis=2))
        # No upload/download between the three operators.
        resized = cv.triton.resize(src, (17, 13), interpolation=cv.INTER_NEAREST)
        gray = cv.triton.cvtColor(resized, cv.COLOR_BGR2GRAY)
        matrix = np.array([[1., 0., 2.], [0., 1., -1.]], np.float64)
        output = cv.triton.warpAffine(gray, matrix, (17, 13), flags=cv.INTER_NEAREST,
                                     borderValue=9)
        reference = cv.warpAffine(cv.cvtColor(self._nearest(host, (17, 13)), cv.COLOR_BGR2GRAY),
                                  matrix, (17, 13), flags=cv.INTER_NEAREST, borderValue=9)
        np.testing.assert_array_equal(output.download(), reference)

    def test_transparent_preserves_initialized_destination(self):
        host = np.arange(9 * 13, dtype=np.uint16).reshape(9, 13)
        src = self._image(host)
        initial = np.full(host.shape, 987, host.dtype)
        dst = self._image(initial)
        matrix = np.array([[1., 0., 3.], [0., 1., 0.]], np.float32)
        result = cv.triton.warpAffine(src, matrix, (13, 9), dst=dst,
                                     flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP,
                                     borderMode=cv.BORDER_TRANSPARENT)
        expected = initial.copy()
        expected[:, :10] = host[:, 3:]
        np.testing.assert_array_equal(result.download(), expected)
        for destination in (None, cv.triton.HipImage(),
                            self._image(np.full((2, 3), 777, np.uint16))):
            with self.subTest(destination=destination):
                with self.assertRaises(cv.error):
                    cv.triton.warpAffine(src, matrix, (13, 9), dst=destination,
                                         borderMode=cv.BORDER_TRANSPARENT)
        np.testing.assert_array_equal(dst.download(), expected)

    def test_cross_stream_dependencies_and_pending_read(self):
        host = np.arange(73 * 109 * 3, dtype=np.uint8).reshape(73, 109, 3)
        src = self._image(host)
        producer, consumer = cv.triton.HipStream(), cv.triton.HipStream()
        self.assertEqual(producer.device(), 0)
        self.assertIsInstance(producer.queryIfComplete(), bool)
        small = cv.triton.resize(src, (61, 47), interpolation=cv.INTER_NEAREST,
                                stream=producer)
        # No waitForCompletion: dependencies are inserted by image storage.
        gray = cv.triton.cvtColor(small, cv.COLOR_BGR2GRAY, stream=consumer)
        # A host write after a queued read must preserve the read's original input.
        small.upload(np.zeros((47, 61, 3), np.uint8))
        expected = cv.cvtColor(self._nearest(host, (61, 47)), cv.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(gray.download(), expected)
        consumer.waitFor(producer)
        consumer.waitForCompletion()
        self.assertTrue(consumer.queryIfComplete())
        # Drop all upstream owners while the surviving output remains usable.
        final = cv.triton.resize(gray, (29, 23), interpolation=cv.INTER_NEAREST,
                                stream=producer)
        del src, small, gray, producer, consumer
        gc.collect()
        np.testing.assert_array_equal(final.download(), self._nearest(expected, (29, 23)))

    def test_pending_roi_read_survives_parent_release(self):
        host = np.arange(31 * 43 * 3, dtype=np.uint8).reshape(31, 43, 3)
        parent = self._image(host)
        roi = parent.roi((7, 5, 23, 17))
        stream = cv.triton.HipStream()
        result = cv.triton.resize(roi, (13, 11), interpolation=cv.INTER_NEAREST,
                                 stream=stream)
        del roi, parent, stream
        gc.collect()
        np.testing.assert_array_equal(result.download(),
                                      self._nearest(host[5:22, 7:30], (13, 11)))

    def test_overlap_refusal_and_disjoint_row_views(self):
        host = np.arange(17 * 36, dtype=np.uint8).reshape(17, 36)
        parent = self._image(host)
        src = parent.roi((0, 0, 12, 17))
        overlap = parent.roi((6, 0, 12, 17))
        before = parent.download()
        for destination in (src, overlap):
            with self.assertRaises(cv.error):
                cv.triton.resize(src, (12, 17), dst=destination,
                                 interpolation=cv.INTER_NEAREST)
        np.testing.assert_array_equal(parent.download(), before)
        # Bounding address intervals overlap, but active bytes in each row do not.
        disjoint = parent.roi((24, 0, 12, 17))
        cv.triton.resize(src, (12, 17), dst=disjoint, interpolation=cv.INTER_NEAREST)
        expected = host.copy()
        expected[:, 24:36] = host[:, :12]
        np.testing.assert_array_equal(parent.download(), expected)

    def test_invalid_inputs_preserve_existing_destination(self):
        src = self._image(np.full((9, 13, 3), 17, np.uint8))
        initial = np.full((5, 7, 3), 91, np.uint8)
        dst = self._image(initial)
        invalid_calls = (
            lambda: cv.triton.resize(src, (7, 5), dst=dst,
                                     interpolation=cv.INTER_LANCZOS4),
            lambda: cv.triton.resize(src, (0, 0), dst=dst, fx=-1, fy=1),
            lambda: cv.triton.cvtColor(src, -1, dst=dst),
            lambda: cv.triton.warpAffine(src, np.zeros((3, 3)), (7, 5), dst=dst),
            lambda: dst.upload(np.zeros((5, 7), np.float64)),
            lambda: src.roi((-1, 0, 4, 4)),
            lambda: cv.triton.resize(cv.triton.HipImage(), (7, 5), dst=dst),
        )
        for index, call in enumerate(invalid_calls):
            with self.subTest(case=index):
                with self.assertRaises(cv.error):
                    call()
                np.testing.assert_array_equal(dst.download(), initial)
        for constructor in (cv.triton.HipImage, cv.triton.HipStream):
            # OpenCV's generated tp_init can wrap a constructor's cv.error in
            # SystemError; verify the original error rather than losing it.
            with self.assertRaises((cv.error, SystemError)) as raised:
                constructor(device=-1)
            error = raised.exception
            if isinstance(error, SystemError):
                self.assertIsInstance(error.__cause__, cv.error)
                error = error.__cause__
            self.assertEqual(error.code, -5)

    def test_defaults_and_pitched_outputs_for_each_operator(self):
        host = np.arange(13 * 19 * 3, dtype=np.uint8).reshape(13, 19, 3)
        src_parent = self._image(host)
        src = src_parent.roi((2, 1, 13, 9))
        matrix = np.array([[1., 0., 1.], [0., 1., -1.]], np.float64)
        operations = (
            (3, lambda **kw: cv.triton.resize(src, (7, 5), **kw),
             dict(fx=0, fy=0, interpolation=cv.INTER_LINEAR, stream=None)),
            (1, lambda **kw: cv.triton.cvtColor(src, cv.COLOR_BGR2GRAY, **kw),
             dict(stream=None)),
            (3, lambda **kw: cv.triton.warpAffine(src, matrix, (7, 5), **kw),
             dict(flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT,
                  borderValue=(0, 0, 0, 0), stream=None)),
        )
        for channels, operation, defaults in operations:
            with self.subTest(channels=channels, defaults=defaults):
                shape = (9, 15) if channels == 1 else (9, 15, channels)
                parent_host = np.full(shape, 183, np.uint8)
                parent = self._image(parent_host)
                # cvtColor preserves source size; the other calls resize to 7x5.
                size = (13, 9) if channels == 1 else (7, 5)
                if channels == 1:
                    parent_host = np.full((13, 21), 183, np.uint8)
                    parent = self._image(parent_host)
                destination = parent.roi((2, 1, size[0], size[1]))
                expected = operation().download()
                explicit = operation(dst=None, **defaults).download()
                np.testing.assert_array_equal(explicit, expected)
                returned = operation(dst=destination, **defaults)
                np.testing.assert_array_equal(returned.download(), expected)
                parent_host[1:1 + size[1], 2:2 + size[0]] = expected
                np.testing.assert_array_equal(parent.download(), parent_host)
                np.testing.assert_array_equal(src_parent.download(), host)

    def test_cross_device_refusals_preserve_output(self):
        try:
            other = cv.triton.HipImage(device=1)
        except (cv.error, SystemError) as error:
            cause = error.__cause__ if isinstance(error, SystemError) else error
            if isinstance(cause, cv.error) and cause.code == -5:
                self.skipTest("requires two visible HIP devices")
            raise
        stream = cv.triton.HipStream(device=1)
        other.upload(np.full((9, 13, 3), 17, np.uint8))
        host = np.full((9, 13, 3), 83, np.uint8)
        src = self._image(host)
        dst = self._image(host)
        matrix = np.array([[1., 0., 0.], [0., 1., 0.]], np.float64)
        operations = (
            lambda image, **kw: cv.triton.resize(image, (13, 9), **kw),
            lambda image, **kw: cv.triton.cvtColor(image, cv.COLOR_BGR2RGB, **kw),
            lambda image, **kw: cv.triton.warpAffine(image, matrix, (13, 9), **kw),
        )
        for operation in operations:
            for image, kwargs in ((other, dict(dst=dst)),
                                  (src, dict(dst=dst, stream=stream))):
                with self.assertRaises(cv.error):
                    operation(image, **kwargs)
                np.testing.assert_array_equal(dst.download(), host)
        with self.assertRaises(cv.error):
            cv.triton.HipStream().waitFor(stream)

    def test_shared_storage_from_python_threads(self):
        host = np.arange(37 * 59 * 3, dtype=np.uint8).reshape(37, 59, 3)
        parent = self._image(host)
        barrier = Barrier(2)

        def consume(size):
            stream = cv.triton.HipStream()
            roi = parent.roi((3, 2, 47, 29))
            barrier.wait(timeout=30)
            result = cv.triton.resize(roi, size, interpolation=cv.INTER_NEAREST,
                                     stream=stream)
            return result, size

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(consume, size) for size in ((23, 17), (19, 13))]
            outputs = [future.result(timeout=120) for future in futures]
        del parent
        gc.collect()
        for image, size in outputs:
            np.testing.assert_array_equal(image.download(),
                                          self._nearest(host[2:31, 3:50], size))


class _HipEntry:
    # Reuse the CUDA tests' cases, references and criteria at the public HIP entry.
    # Only device upload and stream construction differ between the bindings.
    @staticmethod
    def _upload(host):
        result = cv.triton.HipImage()
        result.upload(host)
        return result

    @staticmethod
    def _stream():
        return cv.triton.HipStream()

    def setUp(self):
        NewOpenCVTests.setUp(self)
        if not hasattr(cv, "triton") or cv.triton.backendName() != "hip-jit":
            self.skipTest("requires the HIP Triton module")
        if not cv.triton.haveBackend():
            self.skipTest("no usable HIP device")
        self.assertTrue(hasattr(cv.triton, "HipImage"))
        self.assertTrue(hasattr(cv.triton, "HipStream"))


class triton_hip_resize_binding_test(_HipEntry, common.triton_binding_test):
    pass


class triton_hip_resize_numeric_test(_HipEntry, common.triton_numeric_test):
    def setUp(self):
        super(triton_hip_resize_numeric_test, self).setUp()
        if not common._is_reference_build():
            self.skipTest("linked resize CPU entry is not the reference configuration; "
                          "the C++ suite uses the portable oracle")


class triton_hip_cvtcolor_test(_HipEntry, common.triton_cvtcolor_test):
    pass


class triton_hip_warpaffine_binding_test(_HipEntry, common.triton_warpaffine_binding_test):
    pass


class triton_hip_warpaffine_numeric_test(_HipEntry, common.triton_warpaffine_numeric_test):
    pass


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
