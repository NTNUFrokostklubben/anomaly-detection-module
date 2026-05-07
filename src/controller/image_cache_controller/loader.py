import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy import dtype, ndarray
from entity.image.RasterMeta import RasterMeta

from .image_cache import ImageCache


def load_two_image_arrays(
        img1_path: Path | str,
        img2_path: Path | str,
) -> tuple[np.ndarray, RasterMeta, np.ndarray, RasterMeta]:
    """
    Load two images using the singleton cache.

    Args:
        img1_path (Path): Path to the first image.
        img2_path (Path): Path to the second image.

    Returns:
        Array of the first and second image, as well as the time taken to load the images.
    """

    cache = ImageCache()

    #t0 = time.perf_counter()
    arr1, rm1 = cache.get(img1_path)
    arr2, rm2 = cache.get(img2_path)

    #t_load = time.perf_counter() - t0

    return arr1, rm1, arr2, rm2

def load_image_array(img_path: str | Path, level: int = 0, series: int = 0) -> tuple[
    ndarray[tuple[int, int, int], dtype[Any]], RasterMeta]:
    """
    Load one image array
    :param img_path: the path to the image
    :param level: the desired level, default is 0
    :param series: the desired series, default is 0. Don't touch if you don't know what it is
    :return: the image array.
    """
    cache = ImageCache()
    return cache.get(img_path, level, series)
