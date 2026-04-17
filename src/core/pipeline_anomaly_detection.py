from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.shared_memory import SharedMemory

from core.color_diff import check_difference_two_images
from pathlib import Path
import time
from controller.image_cache_controller import load_two_image_arrays
import geopandas as gpd
import numpy as np
import core.water_detector as wd
from core.line_artifact_detector import detect_glare
from entity.image.Image import Image
import core.artifact_detector as ad
from utils.db_connector import DbConnector, AnalysisType

@dataclass
class _ShmSpec:
    """Descriptor passed to workers instead of a numpy array — zero-copy via shared memory."""
    name: str
    shape: tuple
    dtype: str

    def attach(self) -> tuple[np.ndarray, SharedMemory]:
        """Attach to the shared block and return a numpy view plus the handle to close later."""
        shm = SharedMemory(name=self.name)
        return np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf), shm


def _make_shm(arr: np.ndarray) -> tuple[_ShmSpec, SharedMemory]:
    """Copy arr into a new shared-memory block; returns the descriptor and the owner handle."""
    shm = SharedMemory(create=True, size=arr.nbytes)
    np.copyto(np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf), arr)
    return _ShmSpec(shm.name, arr.shape, str(arr.dtype)), shm


# ---------------------------------------------------------------------------
# Per-analysis worker wrappers — module-level so pickle can find them
# ---------------------------------------------------------------------------

def _artifact_worker(image: Image, increment: int, spec: _ShmSpec):
    arr, shm = spec.attach()
    image.img_arr = arr
    try:
        start_artifact_detection_analysis(image, increment)
    finally:
        shm.close()


def _water_worker(image: Image, sosig_df: gpd.GeoDataFrame, water_gdf: gpd.GeoDataFrame, spec: _ShmSpec):
    arr, shm = spec.attach()
    image.img_arr = arr
    try:
        start_water_detection_analysis(image, sosig_df, water_gdf)
    finally:
        shm.close()


def _color_worker(gdf: gpd.GeoDataFrame, i: int, spec1: _ShmSpec, spec2: _ShmSpec, image_id: str):
    arr1, shm1 = spec1.attach()
    arr2, shm2 = spec2.attach()
    try:
        start_color_difference_analysis(gdf, i, arr1, arr2, image_id)
    finally:
        shm1.close()
        shm2.close()


def _glare_worker(spec: _ShmSpec, img_path: Path):
    arr, shm = spec.attach()
    try:
        start_glare_detection_analysis(arr, img_path)
    finally:
        shm.close()


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
    db = DbConnector()
    if values is not None:
        print("----------- Artifact Analysis -------------")

        print(f"Analysing artifacts in image{image.img_id}")
        print(f"Artifact candidates: {np.sort(values.flatten())[:20]}")
        print(f"Time analysis: {t:.6f}s\n")
        db.add_analysis(image.img_id, AnalysisType.ARTIFACT, ad.artifact_confidence(np.min(values.flatten())))


def start_water_detection_analysis(image: Image, sosig_df: gpd.GeoDataFrame, water_gdf: gpd.GeoDataFrame):
    """
    Start water detection analysis
    """
    increment = 30
    t_0 = time.monotonic()
    polygon_mask = wd.create_water_polygon_mask(water_gdf, sosig_df, image.img_id, image.metadata)
    print(f"  polygon_mask:   {(time.monotonic() - t_0):.2f}s")
    t = time.monotonic()
    hsl_mask = wd.create_water_mask_hsl(image.img_arr, increment, polygon_mask)
    print(f"  hsl_mask:       {(time.monotonic() - t):.2f}s")
    t = time.monotonic()
    disagreement_ratio = wd.find_disagreement_ratio(polygon_mask, hsl_mask)

    confidence_level = wd.dissimilarity_confidence(disagreement_ratio)
    t_1 = time.monotonic()
    db = DbConnector()
    db.add_analysis(image.img_id, AnalysisType.WATER_MASK, confidence_level)
    print("----------- Water  mask difference -------------")

    print(f"Analysing water mask in image{image.img_id}")
    print(f"Disagreement ratio between masks: {disagreement_ratio}")
    print(f"Time analysis: {t_1 - t_0:.6f}s\n")


def start_color_difference_analysis(gdf: gpd.GeoDataFrame, i: int, arr1: np.ndarray, arr2: np.ndarray, image_id: str):
    """
    Start colour difference analysis

    Args:
        gdf (gpd.GeoDataFrame): The geodataframe to analyse
        i (int): The index of the first image to analyse
        arr1 (np.ndarray): The array of the first image to analyse
        arr2 (np.ndarray): The array of the second image to analyse
        image (Image): The image object to add the analysis result to the database
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
    db = DbConnector()
    db.add_analysis(image_id, AnalysisType.COLOR_AVERAGE, confidence_level)

    print("----------- Color Difference -------------")
    print(f"Comparing image {gdf.iloc[i]['bildenummer']} and image {gdf.iloc[i + 1]['bildenummer']}")
    print(f"Image {gdf.iloc[i]['bildenummer']} avg: {avg1}")
    print(f"Image {gdf.iloc[i + 1]['bildenummer']} avg: {avg2}")
    print(f"Difference: {diff}")
    print(f"Difference normalised: {diff / 255}")
    print(f"Confidence level: {confidence_level}")
    print(f"Time analysis: {t:.6f}s\n")

def start_anomaly_analysis(sosi_gdf: gpd.GeoDataFrame, image_folder_path: Path, * ,water_gdf: gpd.GeoDataFrame = None):
    """
    Start anomaly analysis
    Args:
        sosi_gdf: The geodataframe to analyse
        image_folder_path (Path): The folder path of the images to analyse
        water_gdf: The water contour GeoDataFrame for water masking.
    """
    image_count = len(sosi_gdf)

    t0 = time.perf_counter()
    db = DbConnector()
    anomaly_sets = []
    with ProcessPoolExecutor() as executor:
        for i in range(image_count - 1):
            t_0 = time.monotonic()
            img1_path = image_folder_path / sosi_gdf.iloc[i]["bildefilRGB"]
            img2_path = image_folder_path / sosi_gdf.iloc[i + 1]["bildefilRGB"]

            if (not img1_path.exists() or not img2_path.exists()) or (sosi_gdf.iloc[i]["stripenummer"] != sosi_gdf.iloc[i + 1]["stripenummer"]):
                continue
            image1: Image = Image.from_filename(sosi_gdf.iloc[i]["bildefilRGB"])
            arr1, rm1, arr2, _, t_load = load_two_image_arrays(img1_path, img2_path)
            image1.metadata = rm1  # img_arr intentionally left None — workers get it via shared memory

            print("------------------------------------------")
            print(f"Comparing image {sosi_gdf.iloc[i]['bildenummer']} and image {sosi_gdf.iloc[i + 1]['bildenummer']}")
            print(f"Loading images to arr : {t_load:.6f}s \n")

            spec1, shm1 = _make_shm(arr1)
            spec2, shm2 = _make_shm(arr2)
            try:
                futures = {
                    executor.submit(_artifact_worker, image1, 50, spec1): "artifact",
                    executor.submit(_color_worker, sosi_gdf, i, spec1, spec2, image1.img_id): "color",
                    executor.submit(_glare_worker, spec1, img1_path): "glare",
                }
                if water_gdf is not None:
                    futures[executor.submit(_water_worker, image1, sosi_gdf, water_gdf, spec1)] = "water"

                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Analysis '{name}' failed: {e}")
            finally:
                shm1.close()
                shm1.unlink()
                shm2.close()
                shm2.unlink()

            image1.max_confidence = db.get_max_confidence_img(image1.img_id)
            print(f"Max confidence level for image {image1.img_id}: {image1.max_confidence}")

            print("\n")
            anomaly_sets.append(image1)
            t_1 = time.monotonic()
            print(f"Total time for analyses on image {image1.img_id}: {(t_1 - t_0):.2f}s")

    print("Overall time:", time.perf_counter() - t0)
    print(f"Found {image_count} images in the GeoPackage.")
    return anomaly_sets
