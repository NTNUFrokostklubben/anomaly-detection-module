import numpy as np
import core.water_detector as wd
from numba import prange, njit


@njit(parallel=True, cache=True)
def calculate_average_color_block(img_arr: np.ndarray[tuple[int, int, int]],  increment: int):
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
    color_values =np.zeros(num_blocks)

    for by in prange(y_blocks):
        for bx in prange(x_blocks):
            y_start = by * increment
            x_start = bx * increment
            y_end = min(y_start + increment, y_shape)
            x_end = min(x_start + increment, x_shape)
            color_values[bx*by] =  wd.block_mean_rgb(img_arr, y_start, y_end, x_start, x_end)




    return 0


def detect_artifact_naive(img_arr: np.ndarray[tuple[int, int, int]], increment: int):
    """
    A naive approach to detecting image artifacts.
    :param img_arr: the first image, for comparing. Expects (Band, H, W)
    :param next_img_arr: The second image to compare the first image to. (Band, H, W)
    :param increment: the amount of jumps to do.
    :return:
    """

    if img_arr.shape != next_img_arr.shape:
        return 0
