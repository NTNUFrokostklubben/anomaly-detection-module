"""
GDAL infrastructure wrapper.

Provides a clean API for converting SOSI files
using the bundled GDAL/ogr2ogr distribution.
"""

from .converter import convert_sosi_to_gpkg

__all__ = [
    "convert_sosi_to_gpkg",
]