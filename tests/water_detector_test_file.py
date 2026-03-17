
import core.water_detector as wd

from os import listdir
from os.path import isfile, join
from pathlib import Path
import geopandas as gp
from datetime import datetime
import tifffile as tf
import numpy as np
from osgeo import gdal
import geopandas as gpd

def run_all_images():
    increment = 30
    contour_path = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\misc\Vann_22.gpkg"
    sosi_path = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\misc\HX-14365_Vertikalbilde.gpkg"
    folder = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images"
    mypath = folder
    gdf: gp.GeoDataFrame = gpd.read_file(contour_path, layer="polygons")
    sosidf: gp.GeoDataFrame = gpd.read_file(sosi_path, layer="polygons")


    only_files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and Path(f).suffix==".tif" ]
    for idx in range(len(only_files)):
        ds   = wd.load_geotiff_dataset(path=mypath + "\\" + only_files[idx])
        img_data = tf.imread(mypath + "\\" + only_files[idx], maxworkers=8)
        img_data = np.ascontiguousarray(img_data.transpose(2, 0, 1))


        start = datetime.now()
        polygon_mask = wd.clean_water_mask(wd.create_water_polygon_mask(gdf, sosidf, only_files[idx], ds), increment)
        end = datetime.now()
        time = end - start
        print("mask creation time polygon:" + str(time))

        rows, cols = np.nonzero(polygon_mask)
        polygon_mask = polygon_mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        img_data = img_data[:, rows.min():rows.max() + 1, cols.min():cols.max() + 1]

        start = datetime.now()
        hsl_mask = wd.create_water_mask_hsl(img_data, increment, polygon_mask)
        hsl_mask = wd.clean_water_mask(hsl_mask, increment)
        end = datetime.now()
        time = end - start
        print("mask creation time cuda:" + str(time))

        disagreement = polygon_mask.astype(np.bool_) ^ hsl_mask.astype(np.bool_)
        disagreement_count = np.sum(disagreement)
        disagreement_ratio = disagreement_count / polygon_mask.size


        print("image " + only_files[idx])
        print(disagreement_ratio)



def main():
    """
    Function for testing water mask creation
    :return:
    """

    gdal.DontUseExceptions()
    contour_path = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\misc\Vann_22.gpkg"
    #sosi_path = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\misc\HX-14365_Vertikalbilde.gpkg"
    sosi_path = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\anomaly_images\Romsdal-2022-HX13173\HX-13173_Vertikalbilde.gpkg"
    #img_name = "HX-14365_073_014_14835.tif"
    #img_name = "HX-14365_073_047_14868.tif"
    #img_name = "HX-14365_073_001_14822.tif"
    img_name = "HX-13173_112_005_5550.tif"
    #folder = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images"
    #wd.run_all_images(folder, contour_path, sosi_path, 30)
    gdf: gp.GeoDataFrame = gpd.read_file(contour_path, layer="polygons")
    sosidf: gp.GeoDataFrame = gpd.read_file(sosi_path, layer="polygons")

    img_arr = tf.imread(
       r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\anomaly_images\Romsdal-2022-HX13173\\" + img_name,
        maxworkers=8)
    #img_arr = tf.imread(
    #      r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\\" + img_name,
     #    maxworkers=8)


    img_arr = np.ascontiguousarray(img_arr.transpose(2, 0, 1))
    ds = wd.load_geotiff_dataset(
        r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\anomaly_images\Romsdal-2022-HX13173\\"+ img_name)
    # ds = load_geotiff_dataset(
    # r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\\"+ img_name)
    # img_arr = np.ascontiguousarray(img_arr.transpose(2, 0, 1))

    before = datetime.now()
    polygon_mask = wd.create_water_polygon_mask(gdf, sosidf, img_name, ds)
    polygon_mask = wd.clean_water_mask(polygon_mask)

    rows, cols = np.nonzero(polygon_mask)
    polygon_mask = polygon_mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    img_arr = img_arr[:, rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    hsl_mask = wd.create_water_mask_hsl(img_arr, 30, polygon_mask)

    disagreement = polygon_mask.astype(np.bool_) ^ hsl_mask.astype(np.bool_)
    disagreement_count = np.sum(disagreement)
    disagreement_ratio = disagreement_count / polygon_mask.size

    after = datetime.now()
    print(after - before)
    print(disagreement_ratio)


    #masked_img = img_arr * mask[np.newaxis, ...]
    #masked_img = np.ascontiguousarray(masked_img.transpose(1, 2, 0))
    #plt.imshow(masked_img)
    #plt.show()

""" rows, cols = np.nonzero(polygon_mask)
    polygon_mask = polygon_mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    cropped_img = img_arr[:, rows.min():rows.max() + 1, cols.min():cols.max() + 1]"""

