from datetime import datetime
from typing import Any

import numpy as np
from numpy import dtype, ndarray
from osgeo import gdal
import cv2
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def load_geotiff(path) -> tuple[ndarray[tuple[Any, ...], dtype[Any]], gdal.Dataset]:
    ds = gdal.OpenEx(path)
    data = ds.ReadAsArray()  # shape: (bands, H, W)
    print("fin read")
    return data, ds

def create_water_mask(data):
    print("starting mask creation")
    if data.ndim == 2:
        raise ValueError("Single band image, cannot create water mask")

    r = data[0].astype(np.float32)
    g = data[1].astype(np.float32)
    b = data[2].astype(np.float32)

    blue_dominant = ((b > r) & (b > g)).astype(np.uint8)

    # Local variance using OpenCV box filter
    b_sq = cv2.boxFilter(b ** 2, -1, (15, 15))
    b_mean = cv2.boxFilter(b, -1, (15, 15))
    local_var = b_sq - b_mean ** 2

    smooth = (local_var < np.percentile(local_var, 40)).astype(np.uint8)
    mask = cv2.bitwise_and(blue_dominant, smooth)
    print("getting there")
    # Morphological cleanup with OpenCV
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove small blobs using connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_size = 500
    # Vectorized: find all labels that meet the size threshold at once
    large_labels = np.nonzero(stats[1:, cv2.CC_STAT_AREA] >= min_size)[0] + 1  # +1 to skip background
    clean_mask = np.isin(labels, large_labels).astype(np.uint8)
    return clean_mask.astype(bool)



def create_water_mask_2(data):
    print("starting mask creation")
    if data.ndim == 2:
        raise ValueError("Single band image, cannot create water mask")

    # assume data shape is (bands, H, W)
    r = data[0].astype(np.float32)
    g = data[1].astype(np.float32)
    b = data[2].astype(np.float32)

    # blue-dominant seed (same as before)
    blue_dominant = ((b > r) & (b > g)).astype(np.uint8)

    # Local variance using OpenCV box filter (same as before)
    ksize = (15, 15)
    b_sq = cv2.boxFilter(b ** 2, -1, ksize)
    b_mean = cv2.boxFilter(b, -1, ksize)
    local_var = b_sq - b_mean ** 2
    smooth = (local_var < np.percentile(local_var, 40)).astype(np.uint8)

    seeds = cv2.bitwise_and(blue_dominant, smooth)

    # --- New: detect bright/white specular highlights that should belong to water ---
    # Convert to a simple intensity and saturation proxy without full HSV conversion:
    # intensity: mean of channels; saturation proxy: (max-min)/max (range of channels)
    max_ch = np.maximum(np.maximum(r, g), b)
    min_ch = np.minimum(np.minimum(r, g), b)
    intensity = (r + g + b) / 3.0
    # avoid divide-by-zero
    sat_proxy = np.where(max_ch > 0, (max_ch - min_ch) / max_ch, 0.0)

    # thresholds (tuneable):
    # - bright if intensity is high relative to the image (use percentile)
    # - low saturation if color channels are close (white/gray)
    bright_thresh = np.percentile(intensity, 90)  # top 10% considered bright; adjust if needed
    sat_thresh = 0.25  # lower means more "white/neutral"

    bright = (intensity >= bright_thresh).astype(np.uint8)
    low_sat = (sat_proxy <= sat_thresh).astype(np.uint8)

    white_pixels = cv2.bitwise_and(bright, low_sat)

    # Optionally, only accept white pixels that are near existing seeds (so we don't include bright non-water objects)
    # Create a small dilation of seeds and intersect with white pixels
    dilate_radius = 15  # tune this depending on image resolution
    if dilate_radius > 0:
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1))
        seeds_dilated = cv2.dilate(seeds, dil_kernel)
    else:
        seeds_dilated = seeds

    white_near_seeds = cv2.bitwise_and(white_pixels, seeds_dilated)

    # Combine seeds with these white pixels
    combined = cv2.bitwise_or(seeds, white_near_seeds)

    print("getting there")
    # Morphological cleanup with OpenCV (close to fill holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # Remove small blobs using connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_size = 500
    # Vectorized: find all labels that meet the size threshold at once
    large_labels = np.nonzero(stats[1:, cv2.CC_STAT_AREA] >= min_size)[0] + 1  # +1 to skip background
    clean_mask = np.isin(labels, large_labels).astype(np.uint8)
    return clean_mask.astype(bool)

def smoothing(mask: np.ndarray, increment, sensitivity):
    idy, idx = mask.shape
    for y in range(0, idy, increment):
        for x in range(0, idx, increment):
            block:np.ndarray = mask[ y:y+increment, x:x+increment]
            if block.mean() > sensitivity:
                mask[ y:y + increment, x:x + increment].fill(True)

    return mask

def jumping_block_water_detect(data: np.ndarray, increment):
    shape = data.shape
    yShape = shape[1]
    xShape = shape[2]

    xJump = increment
    yJump = increment
    r = data[0].astype(np.float32)
    g = data[1].astype(np.float32)
    b = data[2].astype(np.float32)
    mask = np.zeros_like(data[0], dtype=bool)
    previous = False
    blockStart: np.ndarray = data[0:3, 0: xJump, 0:yJump]
    rMean = blockStart[0].mean()
    gMean = blockStart[1].mean()
    bMean = blockStart[2].mean()
    if bMean > gMean and bMean > rMean:
        previous = True
        mask[ 0:xJump, 0:yJump].fill(True)
    idx = xJump
    idy = 0
    cont = True
    coast = False
    rollover = False
    lastLine = False

    while cont:
        #if idx >= 0 and idy >13400:
           # breakpoint()
            #print(mask[7020:7050, 0:30].all())
        if idy + increment >= yShape:
            yJump = yShape-idy
            lastLine = True
        if rollover:
            idx = 0
            if lastLine:
                break
            xJump = increment
            idy += yJump
            rollover = False

        if idx+increment >= xShape:
            xJump = xShape-idx
            rollover = True

        blockSlice: np.ndarray = data[0:3, idy:idy + yJump, idx:idx + xJump]
        redMean = blockSlice[0].mean()
        greenMean = blockSlice[1].mean()
        blueMean = blockSlice[2].mean()
        if blueMean > greenMean and blueMean > redMean:
            if previous:
                mask[idy:idy+yJump, idx:idx+xJump] = True
                idx += xJump
            else: #Add angling to the pixel line instead of a straight line, might improve coastline look
                for i in range(0, xJump):
                    rLine = blockSlice[0][0:yJump , i].mean()
                    gLine = blockSlice[1][0:yJump , i].mean()
                    bLine = blockSlice[2][0:yJump , i].mean()
                    if bLine > rLine and bLine > gLine:
                        idx += i
                        previous = True
                        break
        else:
            idx += xJump

    return mask




# --- Main ---
def main():
    gdal.DontUseExceptions()
    data, ds = load_geotiff(r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\HX-14365_073_042_14863.tif")
    #water_mask = create_water_mask_2(data)
    #smooth_mask = smoothing(water_mask, 200, 0.25)
    before = datetime.now()
    mask = jumping_block_water_detect(data, 30)
    after = datetime.now()
    delta = after - before
    print(delta)
    dataset: gdal.Dataset = ds
    #plt.imshow(mask, cmap='gray_r', interpolation='nearest')
    #plt.show()
"""
    bandRed = dataset.GetRasterBand(1).ReadAsArray()
    bandGreen = dataset.GetRasterBand(2).ReadAsArray()
    bandBlue = dataset.GetRasterBand(3).ReadAsArray()
    r = np.where(mask, bandRed, 255)
    g = np.where(mask, bandGreen, 255)
    b = np.where(mask, bandBlue, 255)
    img = np.dstack((r, g, b))
    #img = np.transpose(data, (1, 2, 0))
    plt.imshow(img)
    plt.show()
"""


