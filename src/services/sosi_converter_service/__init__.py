"""
GDAL infrastructure wrapper.
"""

from .converter import convert_sosi_to_gpkg

__all__ = [
    "convert_sosi_to_gpkg",
]