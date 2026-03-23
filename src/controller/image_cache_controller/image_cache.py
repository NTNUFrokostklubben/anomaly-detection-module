from osgeo import gdal
import numpy as np
from pathlib import Path
from utils.io_tools import read_tiff_fast


class ImageCache:
    """
    Singleton image cache that keeps up to 2 images.
    Oldest image is removed when capacity is exceeded.
    """
    _instance = None

    def __new__(cls, max_size: int = 2):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_size: int = 2):
        if self._initialized:
            return

        self.max_size = max_size
        self._cache = {}
        self._order = []  # track insertion order
        self._initialized = True

    def get(self, img_path: Path) -> np.ndarray:
        """
        Get an image from the cache.
        Args:
            img_path (Path): Path to the image.

        Returns:
            Array of the image.
        """
        img_path = img_path.resolve()

        # When img_path is in the cache
        if img_path in self._cache:
            return self._cache[img_path]

        # When img_path is NOT in the cache
        ds = gdal.Open(str(img_path))
        if ds is None:
            raise ValueError(f"Could not open image: {img_path}")
        arr = read_tiff_fast(img_path)

        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]

        # remove oldest
        if len(self._order) >= self.max_size:
            oldest = self._order.pop(0)
            del self._cache[oldest]

        self._cache[img_path] = arr
        self._order.append(img_path)

        return arr, ds

    def clear(self):
        """Clear cache completely."""
        self._cache.clear()
