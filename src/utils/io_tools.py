from pathlib import Path
import geopandas as gpd
import numpy as np
from osgeo import gdal
import tifffile as tf

from services.sosi_converter_service import convert_sosi_to_gpkg


def convert_sosi_get_gdf(input_file: Path) -> gpd.GeoDataFrame:
    """
    Converts a sosi file to gpkg and returns GeoDataFrame data.

    Args:
        input_file: Path to sosi file

    Returns:
        GeoDataFrame
    """
    output_path = input_file.with_suffix(".gpkg")
    convert_sosi_to_gpkg(str(input_file), str(output_path))
    return get_gdf_content(output_path)

def get_gdf_content(gpkg_path: Path) -> gpd.GeoDataFrame:
    """Load a GeoPackage file and return its content as a GeoDataFrame.
    
    Args:
        gpkg_path (Path): Path to the GeoPackage file
    
    Returns:
        gpd.GeoDataFrame: The content of the GeoPackage file as a GeoDataFrame    
    """
    gdf = gpd.read_file(gpkg_path, layer="polygons", encoding="ISO-8859-1")
    return gdf

def load_tiff_dataset(path: str | Path) ->  gdal.Dataset:
    """
    Load tiff dataset

    :param path: path to the tiff image
    :return: the gdal dataset.
    """
    ds = gdal.OpenEx(path)
    if ds is None:
        raise ValueError(f"Could not open image: {path}")
    return ds

def read_tiff_fast(path, *, series: int = None, level: int = None) -> np.ndarray[tuple[int, int, int]]:
    """
    Fast reading of large tiff image using tifffile with turbojpeg. No metadata included, for that use
    `load_tiff_dataset`. Transposes images to be (Bands, H,W) from (H,W, Bands) since code base already uses that
     format. Also slices away any extra bands outside RGB, since some image manipulation software adds alpha channel band.
    :param level: The level of the image, higher number is lower resolution, 0 is full size.
    :param series: Related images in the same file, only use this if you know what you are doing
    :param path: path to the tiff image.
    :return: the image as array in shape(bands, H, W).
    """

    if series is not None:
        if level is not None:
            img = tf.imread(path, maxworkers=8, series=series, level=level)
        else:
            img = tf.imread(path, maxworkers=8, series=series)
    elif level is not None:
        img = tf.imread(path, maxworkers=8, level=level)
    else:
        img = tf.imread(path, maxworkers=8)

    return np.transpose(img[:, :, :3], (2, 0, 1))

def count_images_in_folder(path) -> int:
    """
    Counts all GeoTIFF images present for a path

    Args:
        path: Canonical path to folder containing images

    Returns:
        Number of GeoTIFF images
    """
    return len(list(Path(path).glob("*.tif")))