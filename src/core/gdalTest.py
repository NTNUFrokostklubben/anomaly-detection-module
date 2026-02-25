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
    gdal.DontUseExceptions()
    mypath = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(onlyfiles)
    dataset: List[Dataset] = []
    for idx in range(len(onlyfiles)):
        dataset.insert(idx, gdal.Open(mypath +"\\" +onlyfiles[idx]))


    if dataset is None:
        print("Could not open image")
        return
    print("Image opened successfully")

    vals = np.zeros(len(dataset))
    print(len(dataset))
    vals[0] = color_average2(dataset[0])
    vals[1] = color_average2(dataset[1])
    vals[2]  = color_average2(dataset[2])
    vals[3]  = color_average2(dataset[3])
    vals[5]  = color_average2(dataset[4])

    for idx in range(5, len(dataset)):
        vals[idx] = color_average2(dataset[idx])
        print(vals[idx-4], vals[idx-3], vals[idx-2], vals[idx-1], vals[idx])

    delta_arr = np.diff(vals)
    print("\n")
    print(vals)
    print("\n")
    print(delta_arr)

# for idx in range(dataset.RasterCount):
#    rasterArr.insert(idx, dataset.GetRasterBand(idx + 1))
#   bandArr.insert(idx, rasterArr[idx].ReadAsArray())
#  rasterArr2.insert(idx, dataset2.GetRasterBand(idx + 1))
# bandArr2.insert(idx, rasterArr2[idx].ReadAsArray())


def color_average(ds: Dataset):
    rasterArr: List[Band] = []
    # bandArr: List[ndarray] = []

    sumara = 0
    for idx in range(ds.RasterCount):
        rasterArr.insert(idx, ds.GetRasterBand(idx + 1))

    xsize = rasterArr[0].XSize
    ysize = rasterArr[0].YSize
    chunk = 256

    shape = xsize * ysize
    sumara
    for idx in range(len(rasterArr)):
        for y in range(0, ysize, chunk):
            rows = min(chunk, ysize - y)  # clamps the last tile to remaining rows
            for x in range(0, xsize, chunk):
                cols = min(chunk, xsize - x)
                arr: ndarray = rasterArr[idx].ReadAsArray(x, y, cols, rows)
                sumara += arr.sum(dtype=np.float64)

    return sumara/(shape*3)


def color_average2(ds: Dataset):
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
    return f"{sumara/len(bandArr):.2f}"