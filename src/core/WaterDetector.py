import colorsys
from concurrent.futures import  ProcessPoolExecutor
from datetime import datetime
from typing import Any
from numba import njit, prange
from os import listdir
from os.path import isfile, join

import numpy as np
from fontTools.misc.timeTools import timestampNow
from numpy import dtype, ndarray
from osgeo import gdal
from skimage import morphology
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2


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


def create_water_mask_rgb(data: np.ndarray, increment) -> ndarray[tuple[bool]]:
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
    return clean_water_mask(mask)

def create_water_mask_hsl(data: np.ndarray, increment) -> ndarray[tuple[bool]]:
        """
        This function creates a water mask using a jumping block algorithm uses HSL to find water instead of RGB.

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
                y_jump = y_shape - idy
                last_line = True

            if rollover:
                idx = 0
                if last_line:
                    break
                x_jump = increment
                idy += y_jump
                rollover = False

            if idx + increment >= x_shape:
                x_jump = x_shape - idx
                rollover = True

            block_slice: np.ndarray = data[0:3, idy:idy + y_jump, idx:idx + x_jump]
            red_mean, green_mean, blue_mean = (block_slice[i].mean() for i in range(3))

            r_norm = red_mean / 255.0
            g_norm = green_mean / 255.0
            b_norm = blue_mean / 255.0
            h_norm, l_norm, s_norm = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
            h_degrees = h_norm * 360
            l_percent = l_norm * 100
            s_percent = s_norm * 100

            if 170 < h_degrees < 260:
                if previous:
                    mask[idy:idy + y_jump, idx:idx + x_jump] = True
                    idx += x_jump
                else:
                    i, last_block = __block_slicer(block_slice, x_jump, y_jump)
                    idx += i
                    previous = last_block
            else:
                idx += x_jump
        return clean_water_mask(mask)


def create_water_mask_hsl_vectorized(data: np.ndarray, increment: int) -> np.ndarray:
    r = data[0] / 255.0
    g = data[1] / 255.0
    b = data[2] / 255.0

    # Pad so dimensions are divisible by increment
    pad_y = (increment - data.shape[1] % increment) % increment
    pad_x = (increment - data.shape[2] % increment) % increment
    r = np.pad(r, ((0, pad_y), (0, pad_x)))
    g = np.pad(g, ((0, pad_y), (0, pad_x)))
    b = np.pad(b, ((0, pad_y), (0, pad_x)))

    # Reshape into blocks and take mean of each block
    h, w = r.shape
    r_blocks = r.reshape(h // increment, increment, w // increment, increment).mean(axis=(1, 3))
    g_blocks = g.reshape(h // increment, increment, w // increment, increment).mean(axis=(1, 3))
    b_blocks = b.reshape(h // increment, increment, w // increment, increment).mean(axis=(1, 3))

    # Vectorized RGB -> HLS on block means
    max_c = np.maximum(np.maximum(r_blocks, g_blocks), b_blocks)
    min_c = np.minimum(np.minimum(r_blocks, g_blocks), b_blocks)
    delta = max_c - min_c

    h_channel = np.zeros_like(r_blocks)
    mask_delta = delta > 0

    m = mask_delta & (max_c == r_blocks)
    h_channel[m] = (60 * ((g_blocks[m] - b_blocks[m]) / delta[m])) % 360

    m = mask_delta & (max_c == g_blocks)
    h_channel[m] = 60 * ((b_blocks[m] - r_blocks[m]) / delta[m]) + 120

    m = mask_delta & (max_c == b_blocks)
    h_channel[m] = 60 * ((r_blocks[m] - g_blocks[m]) / delta[m]) + 240

    # Water detection on block means
    block_water = (h_channel > 170) & (h_channel < 260)

    # Expand blocks back to original image size
    mask = block_water.repeat(increment, axis=0).repeat(increment, axis=1)[:data.shape[1], :data.shape[2]]
    return clean_water_mask(mask)

@njit(parallel=True,cache=True)
def create_water_mask_hsl_numba(data, increment):
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

            if is_water and delta > 0.06:
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


@njit(parallel=True, cache=True)
def create_water_mask_rgb_numba(data, increment):
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
            red_suppressed = r_mean < (b_mean + g_mean) * 0.5
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


def detect_holes(water_mask: ndarray[tuple[bool]], min_water_area= 5000000) -> ndarray:
    labeled_water, num_water = ndimage.label(water_mask)
    min_island_area = 2000000
    print(labeled_water.dtype)  # int32
    print(num_water)
    all_holes = np.zeros_like(water_mask, dtype=bool)

    for i in range(1, num_water + 1):
        region = labeled_water == i
        if region.sum() < min_water_area:  # skip small water objects
            continue

        # Only fill holes within this specific region
        filled = ndimage.binary_fill_holes(region)
        holes = filled & ~region

        labeled_holes, num_islands = ndimage.label(holes)
        for j in range(1, num_islands + 1):
            if (labeled_holes == j).sum() < min_island_area:
                holes[labeled_holes == j] = False

        all_holes |= holes

        _, ax = plt.subplots()
        ax.imshow(region, cmap='Blues')
        ax.imshow(np.ma.masked_where(~holes, holes), cmap='Reds')
        plt.show()
    accepted = holes.sum() > 0  # or count them:
    _, num_accepted = ndimage.label(holes)

    print(f"Water body {i}: {num_accepted} island(s) detected")


def detect_holes2(mask):
    max_size = 300000
    filled_mask = ndimage.binary_fill_holes(mask)
    holes = filled_mask ^ mask
    cleaned = morphology.remove_small_objects(holes, max_size=max_size)
    labeled_holes, num_holes = ndimage.label(cleaned)
    print(num_holes)
    plt.imshow(cleaned)
    plt.show()


def edge_detect(data, low = 50, high = 150):
    img = np.dstack([data[0], data[1], data[2]])

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray.astype(np.uint8), low, high)

    # img = np.dstack([data[0], data[1], data[2]])
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # _, rgb_water_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title(f'Canny (low={low}, high={high})')
    plt.tight_layout()
    plt.show()
    """
    return edges

def run_all_images(folder, increment):
    mypath = folder
    # mypath = r"D:\HX-14365_NordmøreGSD10\RGB"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    dataset = []
    for idx in range(len(onlyfiles)):
        data, _ = load_geotiff(path=mypath + "\\" + onlyfiles[idx])
        # dataset.insert(idx, gdal.Open(mypath + "\\" + onlyfiles[idx]))
        hsl_mask = clean_water_mask(create_water_mask_hsl_numba(data, increment))
        rgb_mask = clean_water_mask(create_water_mask_rgb_numba(data, increment))

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
def main():

    """
    Function for testing water mask creation
    :return:
    """
    gdal.DontUseExceptions()
    folder = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images"
    #data, _ = load_geotiff(r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\HX-14365_073_047_14868.tif")
    data, _ = load_geotiff(
        r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\HX-14365_073_015_14836.tif")
    #data, _ = load_geotiff(
     #   r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\anomaly_images\Romsdal-2022-HX13173\HX-13173_112_002_5547.tif")
    increment = 30

    #run_all_images(folder, increment)

    hsl_rgb_comparison(data, increment )


    """ print("numba function time HSL and RGB")
    before = datetime.now()
    # mask_rgb = create_water_mask_rgb(data, increment)
    mask_numba_hsl = create_water_mask_hsl_numba(data, increment)
    mask_numba_rgb = create_water_mask_rgb_numba(data, increment)
    clean_numba_mask_hsl=clean_water_mask(mask_numba_hsl)
    clean_numba_mask_rgb=clean_water_mask(mask_numba_rgb)
    after = datetime.now()
    print(after - before)"""

    """
    print("naive function time")
    before = datetime.now()
    # mask_rgb = create_water_mask_rgb(data, increment)
    mask_hsl = create_water_mask_hsl(data, increment)
    after = datetime.now()
    print(after - before)

    print("vectorized function time")
    before = datetime.now()
    #with ProcessPoolExecutor() as executor:
       # t_hsl = executor.submit(create_water_mask_hsl_vectorized, data, increment)
        ##t_rgb = executor.submit(create_water_mask_rgb, data, increment)
    mask_vec = create_water_mask_hsl_vectorized(data, increment)
    after = datetime.now()
    print(after-before)
    """


    #mask = smoothing(mask, 150, 0.4)
    #rows, cols = np.nonzero(mask)
    #mask = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    #detect_holes2(mask)

    """r = np.where(hsl_mask, data[0], 255)
    g = np.where(hsl_mask, data[1], 255)
    b = np.where(hsl_mask, data[2], 255)
    img = np.dstack((r, g, b))
    rows, cols = np.nonzero(hsl_mask)
    cropped = img[ rows.min():rows.max() + 1, cols.min():cols.max() + 1, : ]
    
    plt.imshow(cropped)
    plt.show()"""



