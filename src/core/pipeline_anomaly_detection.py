import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.color_diff import check_difference_two_images
from pathlib import Path
import time
from controller.image_cache_controller import load_two_image_arrays, ImageCache
import geopandas as gpd
import numpy as np
import core.water_detector as wd
from core.line_artefact_detector import detect_line_artefact
from entity.image.Image import Image
import core.artifact_detector as ad
from utils.db_connector import DbConnector, AnalysisType
from services.config_parser.ConfigHandler import Config

logger = logging.getLogger("analysis.pipeline")

def start_line_artefact_detection_analysis(arr: np.ndarray, img_path: Path, log: bool):
    """
    Run line artefact detection on a preloaded image array.

    Args:
        arr: (C, H, W) array already loaded in cache
        img_path: used for output file naming
        log: whether to print or log the results of the analysis
    """
    try:
        db = DbConnector()
        t_0 = time.monotonic()
        all_lines, confidence_result = detect_line_artefact(arr)
        db.add_analysis(img_path.name, AnalysisType.ARTIFACT_LINE,confidence_result)
        if log:
            for ln in all_lines:
                logger.info("Line artefact found:  %s  centre=%s  width=%spx  score=%s",ln['type'],ln['centre'], ln['width_px'], ln['peak_score'],
                            extra={"analysis": "line_artefact", "img_id": img_path.name})
        else:
            for ln in all_lines:
                print(f"  {ln['type']}  centre={ln['centre']}  width={ln['width_px']}px  score={ln['peak_score']:.3f}")
            t_1 = time.monotonic()
            print(f"Total time for line artefact detection: {(t_1 - t_0):.2f}s")
    except Exception as e:
        logger.error("Line artefact detection failed, excpt_msg:%s",e,  extra={"analysis": "line_artefact", "img_id": img_path.name})


def start_artifact_detection_analysis(image, increment, log: bool):
    """
    Start artefact detection analysis on a single image, using the line artefact data from the database to compare against.
    :param image: the image to analyse, must contain img_arr and img_id
    :param increment: the size of the block of pixels to compare.
    :param log: whether to log or not
    """

    try:
        before = time.monotonic()
        values = ad.detect_artifact_consistency([image], increment)
        after = time.monotonic()
        t = after - before
        db = DbConnector()

        if values is not None and not log:
            print("----------- Artifact Analysis -------------")
            print(f"Analysing artifacts in image{image.img_id}")
            print(f"Artifact candidates: {np.sort(values.flatten())[:20]}")
            print(f"Time analysis: {t:.6f}s\n")
            db.add_analysis(image.img_id, AnalysisType.ARTIFACT, ad.artifact_confidence(np.min(values.flatten())))
        elif values is not None and log:
            logger.info("Block artifact confidence score: %s", ad.artifact_confidence(np.min(values.flatten())), extra={"analysis": "artifact", "img_id": image.img_id})
            db.add_analysis(image.img_id, AnalysisType.ARTIFACT, ad.artifact_confidence(np.min(values.flatten())))
        else:
            logger.info("Not enough data for artifact analysis, skipping.", extra={"analysis": "artifact", "img_id": image.img_id})
    except Exception as e:
        logger.error("Artefact detection failed, excpt_msg:%s",e,  extra={"analysis": "artifact", "img_id": image.img_id})

def start_water_detection_analysis(image: Image, sosig_df: gpd.GeoDataFrame, water_gdf: gpd.GeoDataFrame,  log: bool):
    """
        Start water detection analysis on a single image, by comparing a polygon mask based on the water contours
     to a mask created from the image itself using HSL values.
    :param image:  The image to create a water mask on.
    :param sosig_df:  the sosi GeoDataFrame with polygon and metadata information for the image.
    :param water_gdf:  the water polygon GeoDataFrame to create the polygon mask from.
    :param log:  bool for whether to log or not.
    """
    try:
        config = Config()
        increment = int(config.get("pipeline", "water_mask_increment"))
        t_0 = time.monotonic()
        polygon_mask = wd.create_water_polygon_mask(water_gdf, sosig_df, image.img_id, image.metadata)
        #print(f"  polygon_mask:   {(time.monotonic() - t_0):.2f}s")
        t = time.monotonic()
        hsl_mask = wd.create_water_mask_hsl(image.img_arr, increment, polygon_mask)
        #print(f"  hsl_mask:       {(time.monotonic() - t):.2f}s")
        disagreement_ratio = wd.find_disagreement_ratio(polygon_mask, hsl_mask)

        confidence_level = wd.dissimilarity_confidence(disagreement_ratio)
        t_1 = time.monotonic()
        db = DbConnector()
        db.add_analysis(image.img_id, AnalysisType.WATER_MASK, confidence_level)
        if log:
            logger.info("Water mask disagreement ratio: %s, confidence level: %s", disagreement_ratio, confidence_level, extra={"analysis": "water_mask", "img_id": image.img_id})
        else:
            print("----------- Water  mask difference -------------")
            print(f"Analysing water mask in image{image.img_id}")
            print(f"Disagreement ratio between masks: {disagreement_ratio}")
            print(f"Time analysis: {t_1 - t_0:.6f}s\n")
    except Exception as e:
        logger.error("Water detection failed, excpt_msg:%s",e,  extra={"analysis": "water_mask", "img_id": image.img_id})


def start_color_difference_analysis(gdf: gpd.GeoDataFrame, i: int, arr1: np.ndarray, arr2: np.ndarray, image_id: str, log: bool):
    """
    Start colour difference analysis

    Args:
        gdf (gpd.GeoDataFrame): The geodataframe to analyse
        i (int): The index of the first image to analyse
        arr1 (np.ndarray): The array of the first image to analyse
        arr2 (np.ndarray): The array of the second image to analyse
        image_id (str): The image id to add the analysis result to the database
    """
    try:
        avg1, avg2, diff, confidence_level = check_difference_two_images(
            gdf,
            int(gdf.iloc[i]["bildenummer"]),
            int(gdf.iloc[i]["stripenummer"]),
            arr1,
            int(gdf.iloc[i + 1]["bildenummer"]),
            int(gdf.iloc[i + 1]["stripenummer"]),
            arr2,
        )
        db = DbConnector()
        db.add_analysis(image_id, AnalysisType.COLOR_AVERAGE, confidence_level)
        if log:
            logger.info("Color difference confidence level: %s", confidence_level, extra={"analysis": "color_difference", "img_id": image_id})
    except Exception as e:
        logger.error("Color difference analysis failed, excpt_msg:%s",e,  extra={"analysis": "color_difference", "img_id": image_id})


def start_anomaly_analysis(sosi_gdf: gpd.GeoDataFrame, image_folder_path: Path, *, water_gdf: gpd.GeoDataFrame = None,
                           on_image_complete=None, stop_analysis_event=None):
    """
    Start anomaly analysis
    Args:
        sosi_gdf: The geodataframe to analyse
        image_folder_path (Path): The folder path of the images to analyse
        water_gdf: The water contour GeoDataFrame for water masking.
        on_image_complete: Callback function for when image is done analyzing
        stop_analysis_event: Event called over grpc to stop processing the analysis and send back current dataset.
    """
    try:
        image_count = len(sosi_gdf)

        t0 = time.perf_counter()
        db = DbConnector()
        anomaly_sets = []
        config = Config()
        log = True
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(image_count - 1):
                # Stops analysis if the stop_analysis_event has been triggered. This is cross-thread
                if stop_analysis_event is not None and stop_analysis_event.is_set():
                    break
                t_0 = time.monotonic()
                img1_path = image_folder_path / sosi_gdf.iloc[i]["bildefilRGB"]
                img2_path = image_folder_path / sosi_gdf.iloc[i + 1]["bildefilRGB"]

                if not img1_path.exists() or not img2_path.exists():
                    continue

                image1: Image = Image.from_filename(sosi_gdf.iloc[i]["bildefilRGB"])
                arr1, rm1, arr2, _, t_load = load_two_image_arrays(img1_path, img2_path)
                image1.img_arr, image1.metadata = arr1, rm1

                # print("------------------------------------------")
                # print(f"Comparing image {sosi_gdf.iloc[i]['bildenummer']} and image {sosi_gdf.iloc[i + 1]['bildenummer']}")
                #print(f"Loading images to arr : {t_load:.6f}s \n")

                futures = {
                    executor.submit(start_artifact_detection_analysis, image1,
                                    int(config.get("pipeline", "artifact_block_increment")), log): "artifact",
                    executor.submit(start_line_artefact_detection_analysis, arr1, img1_path, log): "line_artifact",
                }
                if sosi_gdf.iloc[i]["stripenummer"] == sosi_gdf.iloc[i + 1]["stripenummer"]:
                     futures[ executor.submit(start_color_difference_analysis, sosi_gdf, i, arr1, arr2, image1.img_id, log)] = "color"

                if water_gdf is not None:
                    futures[executor.submit(start_water_detection_analysis, image1, sosi_gdf, water_gdf, log)] = "water"

                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Analysis '{name}' failed: {e}")

                image1.max_confidence = db.get_max_confidence_img(image1.img_id)
                print(f"Max confidence level for image {image1.img_id}: {image1.max_confidence}")

                print("\n")
                image1.img_arr = None
                image1.metadata = None
                anomaly_sets.append(image1)
                if on_image_complete:
                    on_image_complete()
                t_1 = time.monotonic()
                print(f"Total time for analyses on image {image1.img_id}: {(t_1 - t_0):.2f}s")

        print("Overall time:", time.perf_counter() - t0)
        print(f"Found {image_count} images in the GeoPackage.")
        del arr1
        del arr2
        cache = ImageCache()
        cache.clear()
        gc.collect()
        return anomaly_sets
    except Exception as e:
        logger.error("Anomaly analysis failed, excpt_msg:%s",e,  extra={"analysis": "overall_anomaly_analysis", "img_id": "multiple"})
        return []
