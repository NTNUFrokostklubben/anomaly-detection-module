import time
import numpy as np
from pathlib import Path
from osgeo import gdal
from utils.find_overlap import get_overlap_pixel_images
from skimage.color import rgb2hsv
from skimage.io import imread

def color_average(ds) -> float:
    """ Calculate the average color value of a GDAL dataset by reading each band separately and averaging their values.
    
    Args:
        ds (gdal.Dataset): GDAL dataset to calculate the average color value for
        
    Returns:
        float: Average color value across all bands, rounded to 5 decimal places    
    """
    arr = ds.ReadAsArray()
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    avg = arr.mean()
    return round(float(avg), 5)


def color_average_overlap(ds, bounds, scale=0):
    """
    Calculate average brightness (V) for a cropped overlap region using downsampled read.
    """

    min_x, max_x, min_y, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y

    buf_x = max(1, width // (2**scale))
    buf_y = max(1, height // (2**scale))

    arr = ds.ReadAsArray(
        xoff=min_x,
        yoff=min_y,
        xsize=width,
        ysize=height,
        buf_xsize=buf_x,
        buf_ysize=buf_y,
    )

    return round(float(arr.mean()), 5)

def overlap_color_difference(ds1, ds2, bounds1, bounds2):
    """
    Compute color difference between overlapping image regions
    
    Args:
        ds1 (gdal.Dataset): first image dataset
        ds2 (gdal.Dataset): second image dataset
        bounds1 (tuple): bounds for the overlapping region in the first image (min_x, max_x, min_y, max_y)
        bounds2 (tuple): bounds for the overlapping region in the second image (min_x, max_x, min_y, max_y)
    Returns:
        tuple: (avg1, avg2, diff) where avg1 and avg2 are the average brightness values, and diff is the absolute difference between them    
    """

    avg1 = color_average_overlap(ds1, bounds1)
    avg2 = color_average_overlap(ds2, bounds2)

    diff = abs(avg1 - avg2)

    return avg1, avg2, diff

def timer(func, *args, **kwargs):
    """Timer function for checking colour difference

    Args:
        func (any): function to time
        *args (any): arguments to pass to the function

    Returns: 
        tuple[Any, float] : result of the function and time taken in seconds
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start, 

def check_difference_two_images(gdf, img1_num, strip1, img1, img2_num, strip2, img2):
    
    bounds1, bounds2 = get_overlap_pixel_images(gdf, img1_num, strip1, img2_num, strip2)

    if bounds1 is None:
        print("No overlap")
        return

    ds1 = gdal.Open(img1)
    ds2 = gdal.Open(img2)

    (result, t) = timer(
        overlap_color_difference,
        ds1,
        ds2,
        bounds1,
        bounds2
    )

    avg1, avg2, diff = result
    
    return avg1, avg2, diff, t
    
