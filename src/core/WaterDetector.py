import colorsys
import math
import os
from datetime import datetime
import tifffile as ti
from pathlib import Path
from typing import Any
from numba import njit, prange, cuda
from os import listdir
from os.path import isfile, join

import numpy as np

from numpy import dtype, ndarray
from osgeo import gdal
from skimage import morphology
from scipy import ndimage
import matplotlib.pyplot as plt
import geopandas as gpd



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



@njit(parallel=True,cache=True)
def create_water_mask_hsl_numba(data, increment):
    """
        This function creates a water mask using a jumping block algorith. uses HSL to find water instead of RGB.
        Because of Numba optimization, it is not possible to generalize this function or even reduce the complexity.
        :param data: ndarray -  the image to create a water mask on
        :param increment: int - size of the block squared, not total pixels.
        :return: The water mask.
    """
    y_shape = data.shape[1]
    x_shape = data.shape[2]
    mask = np.zeros((y_shape, x_shape), dtype=np.bool_)

    y_blocks = (y_shape + increment - 1) // increment
    x_blocks = (x_shape + increment - 1) // increment

    for by in prange(y_blocks):  # parallel over rows, each row has its own `previous`
        previous = False
        for bx in range(x_blocks):
            y_start = by * increment
            x_start = bx * increment
            y_end = min(y_start + increment, y_shape)
            x_end = min(x_start + increment, x_shape)

            r_sum = 0.0
            g_sum = 0.0
            b_sum = 0.0
            count = (y_end - y_start) * (x_end - x_start)

            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    r_sum += data[0, y, x]
                    g_sum += data[1, y, x]
                    b_sum += data[2, y, x]

            r_mean = (r_sum / count) / 255.0
            g_mean = (g_sum / count) / 255.0
            b_mean = (b_sum / count) / 255.0

            max_c = max(r_mean, g_mean, b_mean)
            min_c = min(r_mean, g_mean, b_mean)
            delta = max_c - min_c

            h = 0.0
            if delta > 0:
                if max_c == r_mean:
                    h = (60 * ((g_mean - b_mean) / delta)) % 360
                elif max_c == g_mean:
                    h = 60 * ((b_mean - r_mean) / delta) + 120
                else:
                    h = 60 * ((r_mean - g_mean) / delta) + 240

            is_water = 170.0 < h < 290.0

            if is_water:
                if previous:
                    for y in range(y_start, y_end):
                        for x in range(x_start, x_end):
                            mask[y, x] = True
                else:
                    # coastline: scan columns to find where water starts
                    coast_x = x_end  # default to no water found
                    for i in range(x_start, x_end):
                        r_col = 0.0
                        g_col = 0.0
                        b_col = 0.0
                        for y in range(y_start, y_end):
                            r_col += data[0, y, i]
                            g_col += data[1, y, i]
                            b_col += data[2, y, i]
                        col_count = y_end - y_start
                        if b_col > r_col and b_col > g_col:
                            coast_x = i
                            break
                    for y in range(y_start, y_end):
                        for x in range(coast_x, x_end):
                            mask[y, x] = True
                previous = True
            else:
                previous = False

    return mask

@cuda.jit
def _hsl_compute_blocks_kernel(data: np.ndarray, mask, increment):
    bx, by = cuda.grid(2)
    y_shape = data.shape[1]
    x_shape = data.shape[2]

    y_blocks = (y_shape + increment - 1) // increment
    x_blocks = (x_shape + increment - 1) // increment

    if by >= y_blocks or bx >= x_blocks:
        return

    y_start = by * increment
    x_start = bx * increment
    y_end = min(y_start + increment, y_shape)
    x_end = min(x_start + increment, x_shape)

    r_sum = 0.0
    g_sum = 0.0
    b_sum = 0.0
    count = (y_end - y_start) * (x_end - x_start)

    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            r_sum += data[0, y, x]
            g_sum += data[1, y, x]
            b_sum += data[2, y, x]

    r_mean = (r_sum / count) / 255.0
    g_mean = (g_sum / count) / 255.0
    b_mean = (b_sum / count) / 255.0

    max_c = max(r_mean, g_mean, b_mean)
    min_c = min(r_mean, g_mean, b_mean)
    delta = max_c - min_c

    h = 0.0
    if delta > 0:
        if max_c == r_mean:
            h = (60.0 * ((g_mean - b_mean) / delta)) % 360.0
        elif max_c == g_mean:
            h = 60.0 * ((b_mean - r_mean) / delta) + 120.0
        else:
            h = 60.0 * ((r_mean - g_mean) / delta) + 240.0

    if 170.0 < h < 290.0:
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                mask[y, x] = True


def create_water_mask_hsl_cuda(data, increment):
    y_shape = data.shape[1]
    x_shape = data.shape[2]
    y_blocks = (y_shape + increment - 1) // increment
    x_blocks = (x_shape + increment - 1) // increment

    data_gpu = cuda.to_device(data)
    mask_gpu = cuda.to_device(np.zeros((y_shape, x_shape), dtype=np.bool_))

    threads_2d = (16, 16)
    blocks_2d = (math.ceil(x_blocks / 16), math.ceil(y_blocks / 16))
    _hsl_compute_blocks_kernel[blocks_2d, threads_2d](data_gpu, mask_gpu, increment)

    return mask_gpu.copy_to_host()

@njit(parallel=True, cache=True)
def create_water_mask_rgb_numba(data, increment):
    """
        This function creates a water mask using a jumping block algorith. uses rgb to find water instead of hsl.
        Because of Numba optimization, it is not possible to generalize this function.
        :param data: ndarray -  the image to create a water mask on
        :param increment: int - size of the block squared, not total pixels.
        :return: The water mask.
    """
    y_shape = data.shape[1]
    x_shape = data.shape[2]
    mask = np.zeros((y_shape, x_shape), dtype=np.bool_)

    y_blocks = (y_shape + increment - 1) // increment
    x_blocks = (x_shape + increment - 1) // increment

    for by in prange(y_blocks):
        previous = False
        for bx in range(x_blocks):
            y_start = by * increment
            x_start = bx * increment
            y_end = min(y_start + increment, y_shape)
            x_end = min(x_start + increment, x_shape)

            r_sum = 0.0
            g_sum = 0.0
            b_sum = 0.0
            count = (y_end - y_start) * (x_end - x_start)

            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    r_sum += data[0, y, x]
                    g_sum += data[1, y, x]
                    b_sum += data[2, y, x]

            r_mean = r_sum / count
            g_mean = g_sum / count
            b_mean = b_sum / count

            blue_ratio = b_mean / (r_mean + g_mean + b_mean + 1e-6)
            blue_dominant = b_mean > r_mean and b_mean > g_mean

            is_deep_water = blue_dominant and 28.0 < b_mean < 78.0 and blue_ratio > 0.36 and r_mean < 30
            chroma_approx = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)

            is_turquoise = (
                    r_mean < 50.0 and  # red suppressed
                    g_mean > 60.0 and  # green actually present
                    b_mean > 55.0 and  # blue actually present
                    r_mean < g_mean and
                    chroma_approx > 25.0  # not a washed out grey
            )

            is_water = is_deep_water or is_turquoise

            if is_water:
                if previous:
                    for y in range(y_start, y_end):
                        for x in range(x_start, x_end):
                            mask[y, x] = True
                else:
                    coast_x = x_end
                    for i in range(x_start, x_end):
                        r_col = 0.0
                        g_col = 0.0
                        b_col = 0.0
                        for y in range(y_start, y_end):
                            r_col += data[0, y, i]
                            g_col += data[1, y, i]
                            b_col += data[2, y, i]
                        if b_col > r_col and b_col > g_col:
                            coast_x = i
                            break
                    for y in range(y_start, y_end):
                        for x in range(coast_x, x_end):
                            mask[y, x] = True
                previous = True
            else:
                previous = False

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
    return 1, True

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




def detect_holes(mask):
    """
    Detects holes in masks. Optimized for large images like tif.
    :param mask: the mask to detect holes in.
    """
    max_size = 300000
    filled_mask = ndimage.binary_fill_holes(mask)
    holes = filled_mask ^ mask
    cleaned = morphology.remove_small_objects(holes, max_size=max_size)
    labeled_holes, num_holes = ndimage.label(cleaned)
    print(num_holes)
    plt.imshow(cleaned)
    plt.show()




def run_all_images(folder, increment):
    mypath = folder
    # mypath = r"D:\HX-14365_NordmøreGSD10\RGB"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and Path(f).suffix==".tif" ]
    dataset = []
    for idx in range(len(onlyfiles)):
        data, _ = load_geotiff(path=mypath + "\\" + onlyfiles[idx])
        # dataset.insert(idx, gdal.Open(mypath + "\\" + onlyfiles[idx]))

        start = datetime.now()
        hsl_mask = clean_water_mask(create_water_mask_hsl_cuda(data, increment))
        end = datetime.now()
        time = end - start
        print("mask creation time cuda:" + str(time))

        start = datetime.now()
        rgb_mask = clean_water_mask(create_water_mask_rgb_numba(data, increment))
        end = datetime.now()
        time = end-start
        print("mask creation time cpu:" + str(time))

        discrepancy = hsl_mask & ~rgb_mask  # pixels HSL sees but RGB doesn't
        hsl_pixels = np.sum(hsl_mask)
        rgb_pixels = np.sum(rgb_mask)
        if hsl_pixels != 0 and rgb_pixels != 0:
            discrepancy_ratio = np.sum(discrepancy) / hsl_pixels
        elif (hsl_pixels == 0 and rgb_pixels != 0) or (rgb_pixels == 0 and hsl_pixels != 0):
            discrepancy_ratio = 1
        else:
            print("error on image" + onlyfiles[idx])

        print("image " + onlyfiles[idx])
        print(discrepancy_ratio)
        print(discrepancy_ratio > 0.5)



def hsl_rgb_comparison(data, increment):
    hsl_mask = clean_water_mask(create_water_mask_hsl_numba(data, increment))
    rgb_mask = clean_water_mask(create_water_mask_rgb_numba(data, increment))
    rows_hsl, cols_hsl = np.nonzero(hsl_mask)
    rows_rgb, cols_rgb = np.nonzero(rgb_mask)

    if rows_hsl.size > rows_rgb.size:
        crop_rows = rows_hsl
    else:
        crop_rows = rows_rgb.size
    if cols_rgb.size > cols_hsl.size:
        crop_cols = cols_rgb
    else:
        crop_cols = cols_hsl

    cropped_hsl = hsl_mask[crop_rows.min():crop_rows.max() + 1, crop_cols.min():crop_cols.max() + 1]
    cropped_rgb = rgb_mask[crop_rows.min():crop_rows.max() + 1, crop_cols.min():crop_cols.max() + 1]
    discrepancy = cropped_hsl & ~cropped_rgb  # pixels HSL sees but RGB doesn't
    hsl_pixels = np.sum(cropped_hsl)
    discrepancy_ratio = np.sum(discrepancy) / hsl_pixels
    if hsl_pixels == 0:
        return False
    print(discrepancy_ratio)
    print(discrepancy_ratio > 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cropped_hsl, cmap='gray')
    axes[0].set_title('hsl')
    axes[1].imshow(cropped_rgb, cmap='gray')
    axes[1].set_title(f'rgb')
    plt.tight_layout()
    plt.show()
    return 0
# --- Main ---

def find_water_polygon(gpkg_path: str):
    gdf = gpd.read_file(gpkg_path, layer="polygons")

    gdf.columns = [col.encode("latin-1").decode("utf-8") for col in gdf.columns]
    """print(gdf.columns.tolist())
    print(gdf.dtypes)
    print(gdf.head(2))"""

    print(gdf["objekttypenavn"].unique())
    print(gdf["REGULERT"].unique())
    print(f"Total polygons: {len(gdf)}")

def main():
    print
    """
    Function for testing water mask creation
    :return:
    """
    """os.environ['NUMBA_CACHE_DIR'] = 'C:/Users/name/numba_cache'
    dummy = np.zeros((3, 10, 10), dtype=np.uint8)
    start = datetime.now()
    create_water_mask_hsl_numba(dummy, 5)
    create_water_mask_hsl_numba(dummy, 5)
    print("compile time:", datetime.now() - start)"""


    gdal.DontUseExceptions()
    #path = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\misc\Vann_22.gpkg"

    import imagecodecs
    print(imagecodecs.jpeg8_version())
    start = datetime.now()
    arr = ti.imread(r"C:\Users\Augus\Documents\Skule\bachelor\testing-images\HX-14365_073_011_14832.tif", maxworkers=8)
    print("read time:", datetime.now() - start)
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))

    #find_water_polygon(path)

    #folder = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images"
    #folder = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\anomaly_images\Romsdal-2022-HX13173"
    #data, _ = load_geotiff(r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\HX-14365_073_047_14868.tif")




   # data, _ = load_geotiff(
       # r"C:\Users\Augus\Documents\Skule\bachelor\testing-images\HX-14365_073_011_14832.tif")
    #print("read data:", datetime.now() - start)
    #data, _ = load_geotiff(
    #    r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\anomaly_images\Romsdal-2022-HX13173\HX-13173_112_002_5547.tif")
    increment = 30
    start = datetime.now()
    mask_cuda = create_water_mask_hsl_cuda(arr, increment)
    print("cuda time:", datetime.now() - start)

    start = datetime.now()
    mask_hsl= create_water_mask_hsl_numba(arr, increment)
    print("cached CPU time:", datetime.now() - start)


    #run_all_images(folder, increment)

    #hsl_rgb_comparison(data, increment )



    """r = np.where(hsl_mask, data[0], 255)
    g = np.where(hsl_mask, data[1], 255)
    b = np.where(hsl_mask, data[2], 255)
    img = np.dstack((r, g, b))
    rows, cols = np.nonzero(hsl_mask)
    cropped = img[ rows.min():rows.max() + 1, cols.min():cols.max() + 1, : ]
    
    plt.imshow(cropped)
    plt.show()"""



