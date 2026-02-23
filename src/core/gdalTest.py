from datetime import datetime

from osgeo import gdal
from osgeo.gdal import Dataset


def main():
    gdal.DontUseExceptions()
    dataset: Dataset = gdal.Open(r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\HX-14365_073_001_14822.tif", gdal.GA_ReadOnly)
    if dataset is None:
        print("Could not open image")
        return
    print("Image opened successfully")
    raster = dataset.GetRasterBand()

