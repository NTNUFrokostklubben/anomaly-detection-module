import numpy as np
import core.water_detector as wd
from numba import prange, njit


@njit(parallel=True, cache=True)
def calculate_average_color_block(img_arr: np.ndarray[tuple[int, int, int]],  increment: int) -> np.ndarray[tuple[float]]:
    """
    Calculate the average color in a block of size "increment" squared, and then return a list of values.
    :param img_arr: the image to process. Expects (Band, H, W)
    :param increment: the amount of jumps to do and the shape of the block.
    :return:
    """

    _, y_shape,x_shape  = img_arr.shape
    y_blocks = (y_shape + increment - 1) // increment
    x_blocks = (x_shape + increment - 1) // increment
    num_blocks = y_blocks*x_blocks
    color_values = np.zeros((num_blocks, 3))

    for by in prange(y_blocks):
        for bx in prange(x_blocks):
            y_start = by * increment
            x_start = bx * increment
            y_end = min(y_start + increment, y_shape)
            x_end = min(x_start + increment, x_shape)
            r, g, b = wd.block_mean_rgb(img_arr, y_start, y_end, x_start, x_end, None)
            color_values[by * x_blocks + bx, 0] = r
            color_values[by * x_blocks + bx, 1] = g
            color_values[by * x_blocks + bx, 2] = b

    return color_values


def detect_artifact_consistency(images: list, increment: int) -> np.ndarray:
    """
    Detects artifact blocks by measuring color consistency across multiple images.
    A block that stays the same color across all images likely contains an artifact.

    :param images: list of image arrays, each of shape (Band, H, W).
    :param increment: block size in pixels, total pixels is increment squared.
    :return: 1D array of per-block consistency scores. Low values indicate likely artifacts.
    """
    block_means = np.stack([calculate_average_color_block(img, increment) for img in images])
    return (block_means.max(axis=0) - block_means.min(axis=0)).sum(axis=1) / 3.0


