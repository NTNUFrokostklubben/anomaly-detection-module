from datetime import datetime
from typing import Any

import numpy as np
from numpy import dtype, ndarray
from osgeo import gdal
import cv2
from skimage import morphology
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



"""
def smoothing(mask: np.ndarray, increment, sensitivity):
    idy, idx = mask.shape
    for y in range(0, idy, increment):
        for x in range(0, idx, increment):
            block:np.ndarray = mask[ y:y+increment, x:x+increment]
            if block.mean() > sensitivity:
                mask[ y:y + increment, x:x + increment].fill(True)

    return mask
"""

def smoothing(mask: np.ndarray, increment, sensitivity):
    idy, idx = mask.shape
    for y in range(0, idy, increment):
        for x in range(0, idx, increment):
            block:np.ndarray = mask[ y:y+increment, x:x+increment]
            if block.mean() < sensitivity:
                mask[ y:y + increment, x:x + increment].fill(False)

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
    rollover = False
    lastline = False

    while cont:
        #if idx >= 19700 and idy >10900:
           # breakpoint()
            #print(mask[7020:7050, 0:30].all())
        if idy + increment >= yShape:
            yJump = yShape-idy
            lastline = True
        if rollover:
            idx = 0
            if lastline:
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
        brightness = (redMean + greenMean + blueMean) / 3
        saturation = max(redMean, greenMean, blueMean) - min(redMean, greenMean, blueMean)
        blueRatio = blueMean / (redMean + greenMean + blueMean + 1e-6)
        #(greenMean - blueMean < 10 and blueMean > redMean and greenMean > redMean and greenMean > 60 and blueMean > 60)
        #(greenMean - blueMean < 5 and blueMean > redMean)

        if ((blueMean > greenMean and blueMean > redMean and blueMean > 40 and blueMean < 70 and blueRatio > 0.36)):
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


def clean_water_mask(mask_array, min_size=500):
    """
    Remove shadow splotches from a water mask ndarray.

    Parameters:
        mask_array: ndarray - water mask (dark/low values = water)
        min_size: int - minimum component size to keep (default 500)

    Returns:
        cleaned: ndarray (bool) - True where water, False elsewhere
    """
    # Threshold to binary (assumes water = dark pixels, threshold at 50)

    # Remove all connected components smaller than min_size
    cleaned = morphology.remove_small_objects(mask_array, max_size=150000)


    return cleaned


# --- Main ---
def main():
    gdal.DontUseExceptions()
    data, ds = load_geotiff(r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\HX-14365_073_001_14822.tif")
    mask = jumping_block_water_detect(data, 30)

    mask = clean_water_mask(mask)
    r = np.where(mask, data[0], 255)
    g = np.where(mask, data[1], 255)
    b = np.where(mask, data[2], 255)
    img = np.dstack((r, g, b))
    rows, cols = np.where(mask)
    cropped = img[ rows.min():rows.max() + 1, cols.min():cols.max() + 1, : ]

    plt.imshow(cropped)
    plt.show()

    #plt.imshow(mask, cmap='gray_r', interpolation='nearest')
    #plt.show()

    #bandRed = dataset.GetRasterBand(1).ReadAsArray()
    #bandGreen = dataset.GetRasterBand(2).ReadAsArray()
    #bandBlue = dataset.GetRasterBand(3).ReadAsArray()
    #img = np.transpose(data, (1, 2, 0))


