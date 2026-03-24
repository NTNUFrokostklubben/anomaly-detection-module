import time
import numpy as np
import geopandas as gpd
from src.utils.find_overlap import get_overlap_pixel_images
import math


def set_confidence_level(diff: float) -> dict:
    norm_diff = abs(diff / 255.0)

    # F1: Shifted exponential
    k1, t1 = 40, 0.025
    f1 = 1 - math.exp(-k1 * (norm_diff - t1)) if norm_diff > t1 else 0.0

    # F2: Power-scaled exponential
    k2, p2 = 20000, 3
    f2 = 1 - math.exp(-k2 * (norm_diff ** p2))

    # F3: Logistic — zero-floored
    k3, t3 = 120, 0.045
    f3_floor = 1 / (1 + math.exp(-k3 * (0 - t3)))
    f3_raw = 1 / (1 + math.exp(-k3 * (norm_diff - t3)))
    f3 = (f3_raw - f3_floor) / (1 - f3_floor)

    # F4: Gompertz — zero-floored
    k4, t4 = 60, 0.05
    f4_floor = 1 - math.exp(-math.exp(k4 * (0 - t4)))
    f4_raw = 1 - math.exp(-math.exp(k4 * (norm_diff - t4)))
    f4 = (f4_raw - f4_floor) / (1 - f4_floor)

    # F5: Hill function
    t5, p5 = 0.05, 3
    ratio = (norm_diff / t5) ** p5
    f5 = ratio / (1 + ratio)

    results = {
        "norm_diff": norm_diff,
        "F1_shifted_exp": f1,
        "F2_power_exp": f2,
        "F3_logistic": f3,
        "F4_gompertz": f4,
        "F5_hill": f5,
    }

    print(f"\n--- Confidence analysis ---")
    print(f"Raw diff:         {diff}")
    print(f"Normalized diff:  {norm_diff:.6f}")
    print(f"{'Function':<22} {'Score':>8}")
    print("-" * 32)
    for key, value in results.items():
        if key != "norm_diff":
            print(f"{key:<22} {value:>8.6f}")

    return results

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
    Compute colour difference between overlapping image regions
    
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