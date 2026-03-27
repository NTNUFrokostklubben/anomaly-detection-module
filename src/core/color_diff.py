import time
import numpy as np
import geopandas as gpd
from src.utils.find_overlap import get_overlap_pixel_images
import math


def set_confidence_level(diff: float) -> float:
    """Calculate the confidence level of the overlapping region
    f(x)=1-e^(-e^(k(x-t)))
    Args:
        diff: difference between the average rgb values of the overlapping regions

    Returns:
        Confidence level of the overlapping region
    """
    norm_diff = abs(diff / 255.0)
    k, t = 120, 0.055
    f_floor = 1 - math.exp(-math.exp(k * (0 - t)))
    f_raw = 1 - math.exp(-math.exp(k * (norm_diff - t)))
    result = (f_raw - f_floor) / (1 - f_floor)

    return float("{:.3f}".format(result))

def color_average_overlap(img_arr: np.ndarray, bounds:tuple[float, float, float, float]) -> float:
    """Calculate the average colour value of a GDAL dataset for a specific overlapping region defined by bounds.
    Args:
        img_arr (np.ndarray): Array representation of the image
        bounds (tuple): Bounds for the overlapping region in pixel coordinates, as tuples (min_x, max_x, min_y, max_y)
    Returns:
        float: Average colour value for the overlapping region, rounded to 5 decimal places
    """    
    min_x, max_x, min_y, max_y = bounds
    
    img_arr = img_arr[:, min_y:max_y, min_x:max_x]

    mean_arr = img_arr.mean()
    return round(float(mean_arr), 5)


def overlap_color_difference(img_arr1: np.ndarray,
                             img_arr2: np.ndarray,
                             bounds1: tuple[float, float, float, float],
                             bounds2: tuple[float, float, float, float]
                             ) -> tuple[float, float, float]:
    """
    Compute colour difference between overlapping image regions
    
    Args:
        img_arr1 (np.ndarray): first image array
        img_arr2 (np.ndarray): second image array
        bounds1 (tuple): bounds for the overlapping region in the first image (min_x, max_x, min_y, max_y)
        bounds2 (tuple): bounds for the overlapping region in the second image (min_x, max_x, min_y, max_y)
    Returns:
        tuple: (avg1, avg2, diff) where avg1 and avg2 are the average brightness values, and diff is the absolute difference between them    
    """
    
    avg1 = color_average_overlap(img_arr1, bounds1)
    avg2 = color_average_overlap(img_arr2, bounds2)
    diff = abs(avg1 - avg2)

    return avg1, avg2, diff


def check_difference_two_images(gdf: gpd.GeoDataFrame,
                                img1_num: int,
                                strip1:int,
                                arr1: np.ndarray,
                                img2_num: int,
                                strip2:int,
                                arr2: np.ndarray) -> tuple[float, float, float, float, float]:
    """Compare two images, timing both array creation and overlap calculation.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing two images to compare
        img1_num (int): First image number
        strip1 (int): First strip number
        arr1 (np.ndarray): First image array
        img2_num (int): Second image number
        strip2 (int): Second strip number
        arr2 (np.ndarray): Second image array

    Returns:
        avg1 and avg2 are the average brightness values, diff is the absolute difference between them,
        and time is the total time taken in seconds

    """
    bounds1, bounds2 = get_overlap_pixel_images(gdf, img1_num, strip1, img2_num, strip2)
    if bounds1 is None:
        return

    # Wrap array retrieval + overlap calculation in the timer
    start = time.perf_counter()
    result = overlap_color_difference(arr1, arr2, bounds1, bounds2)
    end = time.perf_counter()

    avg1, avg2, diff = result

    confidence_level = set_confidence_level(diff)

    return avg1, avg2, diff, end - start, confidence_level