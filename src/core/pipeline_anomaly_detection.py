from datetime import datetime

from core import crop_arrays_binary
from core.color_diff import check_difference_two_images
from pathlib import Path
import time
from controller.image_cache_controller import load_two_image_arrays
import geopandas as gpd
import numpy as np
import core.water_detector as wd
from core.line_detector import detect_glare
from entity.image.Image import Image
import core.artifact_detector as ad


def start_glare_detection_analysis(arr: np.ndarray, img_path: Path):
    """
    Run glare detection on a preloaded image array.

    Args:
        arr: (C, H, W) array already loaded in cache
        img_path: used for output file naming
    """

    print("----------- Glare Detection -------------")
    glare = detect_glare(arr, img_path)
    for ln in glare:
        print(f"  {ln['type']}  centre={ln['centre']}  width={ln['width_px']}px  score={ln['peak_score']:.3f}")


def start_artifact_detection_analysis(image, increment):
    before = datetime.now()
    values = ad.detect_artifact_consistency([image], increment)
    after = datetime.now()
    t = (after - before).total_seconds()
    if values is not None:
        print("----------- Artifact Analysis -------------")

        print(f"Analysing artifacts in image{image.img_id}")
        print(f"Artifact candidates: {np.sort(values.flatten())[:20]}")
        print(f"Time analysis: {t:.6f}s\n")


def start_water_detection_analysis(image: Image, sosig_df: gpd.GeoDataFrame, water_gdf: gpd.GeoDataFrame):
    """
    Start water detection analysis
    """
    increment = 30
    before = datetime.now()
    polygon_mask = wd.create_water_polygon_mask(water_gdf, sosig_df, image.img_id, image.dataset)
    polygon_mask = wd.clean_water_mask(polygon_mask)
    hsl_mask = wd.create_water_mask_hsl(image.img_arr, increment, polygon_mask)
    polygon_mask, hsl_mask = crop_arrays_binary(polygon_mask, hsl_mask)
    disagreement_ratio = wd.find_disagreement_ratio(polygon_mask, hsl_mask)
    after = datetime.now()
    t = (after - before).total_seconds()
    print("----------- Water  mask difference -------------")

    print(f"Analysing water mask in image{image.img_id}")
    print(f"Disagreement ratio between masks: {disagreement_ratio}")
    print(f"Time analysis: {t:.6f}s\n")


def start_color_difference_analysis(gdf: gpd.GeoDataFrame, i: int, arr1: np.ndarray, arr2: np.ndarray):
    """
    Start colour difference analysis

    Args:
        gdf (gpd.GeoDataFrame): The geodataframe to analyse
        i (int): The index of the first image to analyse
        arr1 (np.ndarray): The array of the first image to analyse
        arr2 (np.ndarray): The array of the second image to analyse
    """
    avg1, avg2, diff, t, confidence_level = check_difference_two_images(
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
    print(f"Difference normalised: {diff / 255}")
    print(f"Confidence level: {confidence_level}")
    print(f"Time analysis: {t:.6f}s\n")


def start_anomaly_analysis(sosi_gdf: gpd.GeoDataFrame, image_folder_path: Path, *, water_gdf: gpd.GeoDataFrame = None):
    """
    Start anomaly analysis
    Args:
        sosi_gdf: The geodataframe to analyse
        image_folder_path (Path): The folder path of the images to analyse
        water_gdf: The water contour GeoDataFrame for water masking.
    """
    image_count = len(sosi_gdf)

    t0 = time.perf_counter()
    for i in range(image_count - 1):

        img1_path = image_folder_path / sosi_gdf.iloc[i]["bildefilRGB"]
        img2_path = image_folder_path / sosi_gdf.iloc[i + 1]["bildefilRGB"]

        if not img1_path.exists() or not img2_path.exists():
            continue
        image1: Image = Image.from_filename(sosi_gdf.iloc[i]["bildefilRGB"])
        arr1, ds1, arr2, _, t_load = load_two_image_arrays(img1_path, img2_path)
        image1.img_arr, image1.dataset = arr1, ds1

        print("------------------------------------------")
        print(f"Comparing image {sosi_gdf.iloc[i]['bildenummer']} and image {sosi_gdf.iloc[i + 1]['bildenummer']}")
        print(f"Loading images to arr : {t_load:.6f}s \n")

        if water_gdf is not None:
            start_water_detection_analysis(image1, sosi_gdf, water_gdf)

        start_artifact_detection_analysis(image1, 100)
        start_color_difference_analysis(sosi_gdf, i, arr1, arr2)
        start_glare_detection_analysis(arr1, img1_path)

        print("\n")

    print("Overall time:", time.perf_counter() - t0)
    print(f"Found {image_count} images in the GeoPackage.")
