from datetime import datetime
from typing import Any

import numpy as np
from numpy import dtype, ndarray
from osgeo import gdal
from skimage import morphology
import matplotlib.pyplot as plt


def load_geotiff(path) -> tuple[ndarray[tuple[Any, ...], dtype[Any]], gdal.Dataset]:
    """
    Load geotiff image into memory. Temporary function
    :param path: path to the tiff image
    :return: the image as array in shape(bands, H, W) and the gdal dataset.
    """
    ds = gdal.OpenEx(path)
    data = ds.ReadAsArray()  # shape: (bands, H, W)
    print("fin read")
    return data, ds

def smoothing(mask: np.ndarray, increment, sensitivity):

    """
    Smoothes out large bodies of water.
    :param mask: the mask to smooth
    :param increment: the size of the smoothing block
    :param sensitivity: how sensitive the smoothing should be
    :return: the smoothed mask
    """
    idy, idx = mask.shape
    for y in range(0, idy, increment):
        for x in range(0, idx, increment):
            block:np.ndarray = mask[ y:y+increment, x:x+increment]
            if block.mean() < sensitivity:
                mask[ y:y + increment, x:x + increment].fill(False)

    return mask


def create_water_mask(data: np.ndarray, increment) -> ndarray[tuple[bool]]:
    """
    This function creates a water mask using a jumping block algorithm.

    :param data: ndarray -  the image to create a water mask on
    :param increment: int - size of the block squared, not total pixels.
    :return: The water mask.
    """
    shape = data.shape
    y_shape = shape[1]
    x_shape = shape[2]
    x_jump = increment
    y_jump = increment

    mask = np.zeros_like(data[0], dtype=bool)
    previous = False
    idx = 0
    idy = 0
    cont = True
    rollover = False
    last_line = False

    while cont:
        if idy + increment >= y_shape:
            y_jump = y_shape-idy
            last_line = True

        if rollover:
            idx = 0
            if last_line:
                break
            x_jump = increment
            idy += y_jump
            rollover = False

        if idx+increment >= x_shape:
            x_jump = x_shape-idx
            rollover = True

        block_slice: np.ndarray = data[0:3, idy:idy + y_jump, idx:idx + x_jump]
        red_mean, green_mean, blue_mean = (block_slice[i].mean() for i in range(3))
        blue_ratio = blue_mean / (red_mean + green_mean + blue_mean + 1e-6)
        blue_dominant = blue_mean > max(red_mean, green_mean)

        if blue_dominant and 33 < blue_mean < 78 and blue_ratio > 0.36:
            if previous:
                mask[idy:idy+y_jump, idx:idx+x_jump] = True
                idx += x_jump
            else:
                i, last_block  = __block_slicer(block_slice, x_jump, y_jump)
                idx += i
                previous = last_block
        else:
            idx += x_jump
    return mask


def __block_slicer(block_slice, x_jump, y_jump):

    """
    Slices the block when previous is false but this block is true. I.E if the previous block was not water but this
    block is water, go back and find the coastline and start iterating from the coastline.
    :param block_slice: ndarray - the sliced block to slice further, block size is determined by increment
    :param x_jump: the amount x jumps / x offset
    :param y_jump: the amount y jumps / y offset
    :return: the offset for x, truth for this block being water.
    """

    for i in range(0, x_jump):
        red_line = block_slice[0][0:y_jump, i].mean()
        green_line = block_slice[1][0:y_jump, i].mean()
        blue_line = block_slice[2][0:y_jump, i].mean()
        if blue_line > red_line and blue_line > green_line:
           return i, True
    return None


def clean_water_mask(mask_array, max_size=500000) -> ndarray[tuple[bool]]:
    """
    Remove shadow splotches from a water mask ndarray.
    :param mask_array: ndarray as type bool or binary
    :param max_size: int - maximum size of the splotches to remove in pixels, default 500,000

    Returns:
        cleaned: ndarray (bool) - True where water, False elsewhere
    """

    cleaned = morphology.remove_small_objects(mask_array, max_size=max_size)
    return cleaned


# --- Main ---
def main():

    """
    Function for testing water mask creation
    :return:
    """
    gdal.DontUseExceptions()
    #data, _ = load_geotiff(r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\HX-14365_073_042_14863-pink.tif")
    data, _ = load_geotiff(
        r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\HX-14365_073_001_14822.tif")
    mask = create_water_mask(data, 30)

    mask = clean_water_mask(mask)
    r = np.where(mask, data[0], 255)
    g = np.where(mask, data[1], 255)
    b = np.where(mask, data[2], 255)
    img = np.dstack((r, g, b))
    rows, cols = np.nonzero(mask)
    cropped = img[ rows.min():rows.max() + 1, cols.min():cols.max() + 1, : ]

    plt.imshow(cropped)
    plt.show()

main()