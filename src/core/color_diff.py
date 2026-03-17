import time
import numpy as np
from pathlib import Path
import geopandas as gpd
from utils.find_overlap import get_overlap_pixel_images

def color_average_overlap(ds_arr: np.ndarray, bounds:tuple[float, float, float, float]) -> float:
    """Calculate the average colour value of a GDAL dataset for a specific overlapping region defined by bounds.
    Args:
        ds_arr (np.ndarray): Array representation of the image dataset
        bounds (tuple): Bounds for the overlapping region in pixel coordinates, as tuples (min_x, max_x, min_y, max_y)
    Returns:
        float: Average colour value for the overlapping region, rounded to 5 decimal places
    """    
    min_x, max_x, min_y, max_y = bounds
    
    ds_arr = ds_arr[:, min_y:max_y, min_x:max_x]

    mean_arr = ds_arr.mean()
    return round(float(mean_arr), 5)


def overlap_color_difference(ds_arr1: np.ndarray, 
                             ds_arr2: np.ndarray, 
                             bounds1: tuple[float, float, float, float], 
                             bounds2: tuple[float, float, float, float]
                             ) -> tuple[float, float, float]:
    """
    Compute color difference between overlapping image regions
    
    Args:
        ds_arr1 (np.ndarray): first image dataset
        ds_arr2 (np.ndarray): second image dataset
        bounds1 (tuple): bounds for the overlapping region in the first image (min_x, max_x, min_y, max_y)
        bounds2 (tuple): bounds for the overlapping region in the second image (min_x, max_x, min_y, max_y)
    Returns:
        tuple: (avg1, avg2, diff) where avg1 and avg2 are the average brightness values, and diff is the absolute difference between them    
    """
    
    avg1 = color_average_overlap(ds_arr1, bounds1)
    avg2 = color_average_overlap(ds_arr2, bounds2)
    diff = abs(avg1 - avg2)

    return avg1, avg2, diff

# def timer(func, *args, **kwargs) -> tuple[tuple[float, float, float], float]:
#     """Timer function for checking colour difference
#
#     Args:
#         func (any): function to time
#         *args (any): arguments to pass to the function
#
#     Returns:
#         tuple[tuple[float, float, float], float] : result of the function and time taken in seconds
#     """
#     start = time.perf_counter()
#     result = func(*args, **kwargs)
#     end = time.perf_counter()
#     return result, end - start


def check_difference_two_images(gdf: gpd.GeoDataFrame,
                                img1_num: int,
                                strip1:int,
                                arr1: np.ndarray,
                                img2_num: int,
                                strip2:int,
                                arr2: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compare two images, timing both array creation and overlap calculation.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing two images to compare
        img1_num (int): First image number
        strip1 (int): First strip number
        img1_path (Path): First image path
        img2_num (int): Second image number
        strip2 (int): Second strip number
        img2_path (Path): Second image path

    Returns:

    """
    bounds1, bounds2 = get_overlap_pixel_images(gdf, img1_num, strip1, img2_num, strip2)
    if bounds1 is None:
        return

    # Wrap array retrieval + overlap calculation in the timer
    start = time.perf_counter()
    result = overlap_color_difference(arr1, arr2, bounds1, bounds2)
    end = time.perf_counter()

    avg1, avg2, diff = result

    return avg1, avg2, diff, end - start