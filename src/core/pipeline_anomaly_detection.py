from core.color_diff import check_difference_two_images
from pathlib import Path
import time
from controller.image_cache_controller import  load_image_array
import geopandas as gpd
import numpy as np

def start_water_detection_analysis():
    """
    Start water detection analysis
    """
    # Todo connect with the finished result of the water detection analysis
    return

def start_color_difference_analysis(gdf: gpd.GeoDataFrame, i:int, arr1: np.ndarray, arr2: np.ndarray ):
    """
    Start colour difference analysis

    Args:
        gdf (gpd.GeoDataFrame): The geodataframe to analyse
        i (int): The index of the first image to analyse
        arr1 (np.ndarray): The array of the first image to analyse
        arr2 (np.ndarray): The array of the second image to analyse
    """
    avg1, avg2, diff, t = check_difference_two_images(
        gdf,
        int(gdf.iloc[i]["bildenummer"]),
        int(gdf.iloc[i]["stripenummer"]),
        arr1,
        int(gdf.iloc[i + 1]["bildenummer"]),
        int(gdf.iloc[i + 1]["stripenummer"]),
        arr2,
    )

    print("----------- Color Difference -------------")
    print(f"Comparing image {gdf.iloc[i]['bildenummer']} and image {gdf.iloc[i + 1]['bildenummer']}")
    print(f"Image {gdf.iloc[i]['bildenummer']} avg: {avg1}")
    print(f"Image {gdf.iloc[i + 1]['bildenummer']} avg: {avg2}")
    print(f"Difference: {diff}")
    print(f"Time analysis: {t:.6f}s\n")

def start_anomaly_analysis(gdf, image_folder_path: Path):
    """
    Start anomaly analysis
    Args:
        gdf (gpd.GeoDataFrame): The geodataframe to analyse
        image_folder_path (Path): The folder path of the images to analyse
    """
    image_count = len(gdf)

    t0 = time.perf_counter()
    for i in range(image_count - 1):

        img1_path = image_folder_path / gdf.iloc[i]["bildefilRGB"]
        img2_path = image_folder_path / gdf.iloc[i + 1]["bildefilRGB"]

        if not img1_path.exists() or not img2_path.exists():
            continue

        arr1, arr2, t_load = load_image_array(img1_path, img2_path)

        print("------------------------------------------")
        print(f"Comparing image {gdf.iloc[i]['bildenummer']} and image {gdf.iloc[i + 1]['bildenummer']}")
        print(f"Loading images to arr : {t_load:.6f}s \n")

        start_color_difference_analysis(gdf, i, arr1, arr2)
        start_water_detection_analysis()
        print("\n")


    print("Overall time:", time.perf_counter() - t0)
    print(f"Found {image_count} images in the GeoPackage.")
