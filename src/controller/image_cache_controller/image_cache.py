from osgeo import gdal
import numpy as np
from pathlib import Path
from collections import OrderedDict


class ImageCache:

    def __init__(self, max_size: int = 2):
        self.max_size = max_size
        self._cache: OrderedDict[Path, np.ndarray] = OrderedDict()


    def get(self, img_path: Path) -> np.ndarray:
        """
        Get image as numpy array, using cache if available.

        Args:
            img_path (Path): Path to image

        Returns:
            np.ndarray: Image array
        """

        # Cache hit
        if img_path in self._cache:
            # Move to end (marks as most recently used)
            self._cache.move_to_end(img_path)
            return self._cache[img_path]

        # Cache miss → load image
        ds = gdal.Open(str(img_path))
        if ds is None:
            raise ValueError(f"Could not open image: {img_path}")

        arr = ds.ReadAsArray()

        # Ensure 3D shape (C, H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]

        # Enforce cache size
        if len(self._cache) >= self.max_size:
            # popitem(last=False) removes oldest entry
            self._cache.popitem(last=False)

        # Store in cache
        self._cache[img_path] = arr

        return arr

    def clear(self):
        """Clear cache completely."""
        self._cache.clear()