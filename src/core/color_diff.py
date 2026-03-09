import time
import numpy as np
from pathlib import Path
from osgeo import gdal
from utils.find_overlap import get_overlap_pixel_images


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

def get_crop_extent(ds, bounds):
    min_x, max_x, min_y, max_y = bounds

    gt = ds.GetGeoTransform()

    px_w = gt[1]
    px_h = gt[5]

    origin_x = gt[0]
    origin_y = gt[3]

    world_min_x = origin_x + min_x * px_w
    world_max_x = origin_x + max_x * px_w

    world_min_y = origin_y + max_y * px_h
    world_max_y = origin_y + min_y * px_h

    return (world_min_x, world_min_y, world_max_x, world_max_y)

def color_average_overlap(ds, bounds):
    """
    Calculate average color for a cropped overlap region

    Args:
        ds (gdal.Dataset): image dataset
        bounds (tuple): (min_x, max_x, min_y, max_y)

    Returns:
        float: average color value
    """

    min_x, max_x, min_y, max_y = bounds

    arr = ds.ReadAsArray()
    arr = arr[:, min_y:max_y, min_x:max_x]

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    return round(float(arr.mean()), 5)

def overlap_color_difference(ds1, ds2, bounds1, bounds2):
    """
    Compute color difference between overlapping image regions
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

def check_difference_two_images(gpkg, img1_num, strip1, img1, img2_num, strip2, img2):
    
    bounds1, bounds2 = get_overlap_pixel_images(gpkg, img1_num, strip1, img2_num, strip2)

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
    

def main():
    data_path = Path(__file__).parent.parent.parent / "tests" / "testdata" / "RGB_testing"

    img1 = data_path / "HX-14365_001_005_00005.tif"
    img2 = data_path / "HX-14365_001_006_00006.tif"
    
    #add a check for fetching paths

    gpkg = Path(__file__).parent.parent.parent / "tests" / "testdata" / "test_file_short.gpkg"
    # TODO: make sure the program only loads the gpkg file in once
    avg1, avg2, difference, time = check_difference_two_images(gpkg, 5, 1, img1, 6, 1, img2)


    print(f"Image1 avg: {avg1}")
    print(f"Image2 avg: {avg2}")
    print(f"Difference: {difference}")
    print(f"Time: {time:.6f}s")

# if __name__ == "__main__":
#     main()