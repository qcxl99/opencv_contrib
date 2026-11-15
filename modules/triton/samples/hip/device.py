#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

"""Run resize, colour conversion and an affine warp on HIP device images."""

import argparse

import cv2 as cv
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0,
                        help="visible HIP device index (default: 0)")
    args = parser.parse_args()
    if cv.triton.backendName() != "hip-jit":
        parser.error("this sample requires OpenCV's Triton HIP backend")

    image = (np.arange(96 * 128 * 3, dtype=np.uint32) % 256).astype(np.uint8)
    image = image.reshape(96, 128, 3)
    src = cv.triton.HipImage(device=args.device)
    src.upload(image)
    producer = cv.triton.HipStream(device=args.device)
    consumer = cv.triton.HipStream(device=args.device)

    small = cv.triton.resize(src, (64, 48), interpolation=cv.INTER_AREA,
                             stream=producer)
    # Device-image dependencies order the two streams automatically.
    gray = cv.triton.cvtColor(small, cv.COLOR_BGR2GRAY, stream=consumer)
    matrix = np.array([[1, 0, 3], [0, 1, -2]], dtype=np.float64)
    dst = cv.triton.HipImage(device=args.device)
    warped = cv.triton.warpAffine(gray, matrix, (64, 48), dst=dst,
                                 flags=cv.INTER_NEAREST,
                                 borderMode=cv.BORDER_REPLICATE, stream=consumer)
    result = warped.download()  # Blocking NumPy convenience transfer.

    reference = cv.resize(image, (64, 48), interpolation=cv.INTER_AREA)
    reference = cv.cvtColor(reference, cv.COLOR_BGR2GRAY)
    reference = cv.warpAffine(reference, matrix, (64, 48),
                              flags=cv.INTER_NEAREST,
                              borderMode=cv.BORDER_REPLICATE)
    np.testing.assert_array_equal(result, reference)
    print("HIP device {}: {} {}, CPU comparison passed".format(
        src.device(), result.shape, result.dtype))


if __name__ == "__main__":
    main()
