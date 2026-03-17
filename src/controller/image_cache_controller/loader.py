import time
from pathlib import Path
import numpy as np

from .image_cache import ImageCache


def load_image_array(
        img1_path: Path,
        img2_path: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load two images using the singleton cache.

    Args:
        img1_path (Path): Path to the first image.
        img2_path (Path): Path to the second image.

    Returns:
        Array of the first and second image, as well as the time taken to load the images.
    """

    cache = ImageCache()

    t0 = time.perf_counter()

    arr1 = cache.get(img1_path)
    arr2 = cache.get(img2_path)

    t_load = time.perf_counter() - t0

    return arr1, arr2, t_load
