import math
from datetime import datetime
from typing import List

from numpy import ndarray
from osgeo import gdal
from osgeo.gdal import Dataset, Band
import numpy as np
from os import listdir
from os.path import isfile, join
np.set_printoptions(suppress=True, precision=3) #Removes awful numpy number formatting


def main():
    """
    The main function for this file, probably needs to be renamed later. Mainly for testing
    :return: null
    """
    gdal.DontUseExceptions()
    #mypath = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images"
    mypath = r"D:\HX-14365_NordmøreGSD10\RGB"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    dataset: List[Dataset] = []
    for idx in range(len(onlyfiles)):
        dataset.insert(idx, gdal.Open(mypath +"\\" +onlyfiles[idx]))
    vals = np.zeros(len(dataset))
    print(len(dataset))
    vals[0] = color_average(dataset[0])
    vals[1] = color_average(dataset[1])
    vals[2]  = color_average(dataset[2])
    vals[3]  = color_average(dataset[3])
    vals[4]  = color_average(dataset[4])

    for idx in range(5, len(dataset)):
        vals[idx] = color_average(dataset[idx])
        print(vals[idx-4], vals[idx-3], vals[idx-2], vals[idx-1], vals[idx])
        print(np.diff(vals[idx-5:idx]))

    delta_arr = np.diff(vals)
    print("\n")
    print(vals)
    print("\n")
    print(delta_arr)



def color_average(ds: Dataset) -> float:
    """
    Calculate the average color in an image. Very slow for now but no other functions so far beats this one
    :param ds: Dataset from GDAL
    :return: the value representing the average color for this dataset
    """
    rasterArr: List[Band] = []
    bandArr: List[ndarray] = []
    sumara = 0
    for idx in range(ds.RasterCount):
        rasterArr.insert(idx, ds.GetRasterBand(idx + 1))
        bandArr.insert(idx, rasterArr[idx].ReadAsArray())
    shape = bandArr[0].shape
    factor = shape[0]* shape[1]

    for idx in range(len(bandArr)):
        sumara += bandArr[idx].sum()/factor
    return float(f"{sumara/len(bandArr):.2f}")



def color_average_array(bandArr: List[ndarray]) -> float:
    """
    Calculate the average color in an image. Very slow for now but no other functions so far beats this one
    :param bandArr: Array containing the bands from a dataset, must be from gdal.Dataset.ReadAsArray()
    :return: the value representing the average color for the dataset's bands
    """
    shape = bandArr[0].shape
    factor = shape[0] * shape[1]
    sumara = 0
    for idx in range(len(bandArr)):
        sumara += bandArr[idx].sum() / factor
    return float(f"{sumara / len(bandArr):.2f}")