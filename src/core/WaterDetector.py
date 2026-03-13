
import math

import geopandas
import imagecodecs
from datetime import datetime
import cv2
import rasterio
import tifffile as tf
from numba import njit, prange, cuda
import numpy as np
from scipy.spatial.distance import cdist
from shapely.affinity import scale, rotate
from numpy import ndarray
from osgeo import gdal
from shapely.affinity import affine_transform,scale, translate
from skimage import morphology
from scipy import ndimage
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.ops import unary_union
from affine import Affine as af
from rasterio.features import rasterize, geometry_mask
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import transform as shapely_transform


def load_geotiff_dataset(path) ->  gdal.Dataset:
    """
    Load geotiff image into memory. Temporary function
    :param path: path to the tiff image
    :return: the image as array in shape(bands, H, W) and the gdal dataset.
    """
    ds = gdal.OpenEx(path)
    #data = ds.ReadAsArray()  # shape: (bands, H, W)
    print("fin read")
    return ds


def __find_image_row(gdf, img_name):
    """
    Temporary function until utils are pushed to develop
    :param gdf:
    :param img_num:
    :param strip_num:
    :return:
    """
    matches = gdf[gdf["bildefilRGB"] == img_name]

    if matches.empty:
        raise ValueError(f"Image with name {img_name} not found")

    return matches.iloc[0]


@njit(parallel=True,cache=True)
def create_water_mask_hsl_numba(data, increment):
    """
        This function creates a water mask using a jumping block algorith. uses HSL to find water instead of RGB.
        Because of Numba optimization, it is not possible to generalize this function or even reduce the complexity.
        :param data: ndarray -  the image to create a water mask on
        :param increment: int - size of the block squared, not total pixels.
        :return: The water mask.
    """
    y_shape = data.shape[1]
    x_shape = data.shape[2]
    mask = np.zeros((y_shape, x_shape), dtype=np.bool_)

    y_blocks = (y_shape + increment - 1) // increment
    x_blocks = (x_shape + increment - 1) // increment

    for by in prange(y_blocks):  # parallel over rows, each row has its own `previous`
        previous = False
        for bx in range(x_blocks):
            y_start = by * increment
            x_start = bx * increment
            y_end = min(y_start + increment, y_shape)
            x_end = min(x_start + increment, x_shape)

            r_sum = 0.0
            g_sum = 0.0
            b_sum = 0.0
            count = (y_end - y_start) * (x_end - x_start)

            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    r_sum += data[0, y, x]
                    g_sum += data[1, y, x]
                    b_sum += data[2, y, x]

            r_mean = (r_sum / count) / 255.0
            g_mean = (g_sum / count) / 255.0
            b_mean = (b_sum / count) / 255.0

            max_c = max(r_mean, g_mean, b_mean)
            min_c = min(r_mean, g_mean, b_mean)
            delta = max_c - min_c

            h = 0.0
            if delta > 0:
                if max_c == r_mean:
                    h = (60 * ((g_mean - b_mean) / delta)) % 360
                elif max_c == g_mean:
                    h = 60 * ((b_mean - r_mean) / delta) + 120
                else:
                    h = 60 * ((r_mean - g_mean) / delta) + 240

            is_water = 170.0 < h < 290.0

            if is_water:
                if previous:
                    for y in range(y_start, y_end):
                        for x in range(x_start, x_end):
                            mask[y, x] = True
                else:
                    # coastline: scan columns to find where water starts
                    coast_x = x_end  # default to no water found
                    for i in range(x_start, x_end):
                        r_col = 0.0
                        g_col = 0.0
                        b_col = 0.0
                        for y in range(y_start, y_end):
                            r_col += data[0, y, i]
                            g_col += data[1, y, i]
                            b_col += data[2, y, i]
                        col_count = y_end - y_start
                        if b_col > r_col and b_col > g_col:
                            coast_x = i
                            break
                    for y in range(y_start, y_end):
                        for x in range(coast_x, x_end):
                            mask[y, x] = True
                previous = True
            else:
                previous = False

    return mask

@cuda.jit
def _hsl_compute_blocks_kernel(data: np.ndarray, mask, increment):
    bx, by = cuda.grid(2)
    y_shape = data.shape[1]
    x_shape = data.shape[2]

    y_blocks = (y_shape + increment - 1) // increment
    x_blocks = (x_shape + increment - 1) // increment

    if by >= y_blocks or bx >= x_blocks:
        return

    y_start = by * increment
    x_start = bx * increment
    y_end = min(y_start + increment, y_shape)
    x_end = min(x_start + increment, x_shape)

    r_sum = 0.0
    g_sum = 0.0
    b_sum = 0.0
    count = (y_end - y_start) * (x_end - x_start)

    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            r_sum += data[0, y, x]
            g_sum += data[1, y, x]
            b_sum += data[2, y, x]

    r_mean = (r_sum / count) / 255.0
    g_mean = (g_sum / count) / 255.0
    b_mean = (b_sum / count) / 255.0

    max_c = max(r_mean, g_mean, b_mean)
    min_c = min(r_mean, g_mean, b_mean)
    delta = max_c - min_c

    h = 0.0
    if delta > 0:
        if max_c == r_mean:
            h = (60.0 * ((g_mean - b_mean) / delta)) % 360.0
        elif max_c == g_mean:
            h = 60.0 * ((b_mean - r_mean) / delta) + 120.0
        else:
            h = 60.0 * ((r_mean - g_mean) / delta) + 240.0

    if 170.0 < h < 290.0:
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                mask[y, x] = True


def create_water_mask_hsl_cuda(data, increment):
    y_shape = data.shape[1]
    x_shape = data.shape[2]
    y_blocks = (y_shape + increment - 1) // increment
    x_blocks = (x_shape + increment - 1) // increment

    data_gpu = cuda.to_device(data)
    mask_gpu = cuda.to_device(np.zeros((y_shape, x_shape), dtype=np.bool_))

    threads_2d = (16, 16)
    blocks_2d = (math.ceil(x_blocks / 16), math.ceil(y_blocks / 16))
    _hsl_compute_blocks_kernel[blocks_2d, threads_2d](data_gpu, mask_gpu, increment)

    return mask_gpu.copy_to_host()


def clean_water_mask(mask_array, max_size=1000000) -> ndarray[tuple[bool]]:
    """
    Remove shadow splotches from a water mask ndarray.
    :param mask_array: ndarray as type bool or binary
    :param max_size: int - maximum size of the splotches to remove in pixels, default 500,000

    Returns:
        cleaned: ndarray (bool) - True where water, False elsewhere
    """

    cleaned = morphology.remove_small_objects(mask_array, max_size=max_size)
    return cleaned




def detect_holes(mask):
    """
    Detects holes in masks. Optimized for large images like tif.
    :param mask: the mask to detect holes in.
    """
    max_size = 300000
    filled_mask = ndimage.binary_fill_holes(mask)
    holes = filled_mask ^ mask
    cleaned = morphology.remove_small_objects(holes, max_size=max_size)
    labeled_holes, num_holes = ndimage.label(cleaned)
    print(num_holes)
    plt.imshow(cleaned)
    plt.show()



def find_water_polygon(gpkg_path: str, sosi_path: str, img_name: str, ds: gdal.Dataset, img_data: np.ndarray):
    """

    :param img_data:
    :param ds:
    :param gpkg_path:
    :param sosi_path:
    :param img_name:
    :return:
    """
    gdf: geopandas.GeoDataFrame = gpd.read_file(gpkg_path, layer="polygons")
    sosidf: geopandas.GeoDataFrame = gpd.read_file(sosi_path, layer="polygons")
    raster_crs = ds.GetProjection()
    if not raster_crs:
        raise RuntimeError("Raster has no CRS; can't align vectors.")
    gdf = gdf.to_crs(raster_crs)
    sosidf = sosidf.to_crs(raster_crs)

    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    print(gt)

    row = __find_image_row(sosidf,img_name)
    affine = af.from_gdal(*gt)
    overlap = gdf['geometry'].intersects(row['geometry'])


    merged = unary_union(gdf[overlap]['geometry'])

    mask = geometry_mask(
        [merged],
        transform=affine,
        invert=True,
        out_shape=(height, width)
    )


    masked_img = img_data * mask[np.newaxis, ...]
    masked_img = np.ascontiguousarray(masked_img.transpose(1, 2, 0))
    plt.imshow(masked_img)
    plt.show()

def find_water_polygon_mask(gpkg_path, sosi_path, img_name, ds, img_data):


    gdf = gpd.read_file(gpkg_path, layer="polygons")
    sosidf = gpd.read_file(sosi_path, layer="polygons")

    with rasterio.open(ds.GetDescription()) as src:
        raster_crs = src.crs
        width = src.width
        height = src.height

    gdf = gdf.to_crs(raster_crs)
    sosidf = sosidf.to_crs(raster_crs)

    row = __find_image_row(sosidf, img_name)

    gt = ds.GetGeoTransform()
    x0, dx, rot_x, y0, rot_y, dy = gt

    def pixel_to_geo(px, py, gt):
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        return x, y

    # Build image footprint from GeoTransform corners
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    geo_corners = [pixel_to_geo(px, py, gt) for px, py in corners]
    image_footprint = ShapelyPolygon(geo_corners)

    # Get overlapping water polygons clipped to image footprint
    overlap = gdf['geometry'].intersects(image_footprint)
    merged = unary_union(gdf[overlap]['geometry'])
    merged_clipped = merged.intersection(image_footprint)

    # Invert the geotransform to get geo->pixel matrix
    det = dx * dy - rot_x * rot_y
    a    =  dy  / det
    b    = -rot_x / det
    d    = -rot_y / det
    e    =  dx  / det
    xoff = -(a * x0 + b * y0)
    yoff = -(d * x0 + e * y0)

    # SOSI polygon
    sosi_geom = row['geometry']
    if sosi_geom.geom_type == 'MultiPolygon':
        sosi_geom = max(sosi_geom.geoms, key=lambda g: g.area)

    sosi_coords = np.array(sosi_geom.exterior.coords[:-1], dtype=np.float32)
    if len(sosi_coords) != 4:
        hull = sosi_geom.convex_hull
        hull_coords = np.array(hull.exterior.coords[:-1], dtype=np.float32)
        top    = hull_coords[np.argmax(hull_coords[:, 1])]
        bottom = hull_coords[np.argmin(hull_coords[:, 1])]
        left   = hull_coords[np.argmin(hull_coords[:, 0])]
        right  = hull_coords[np.argmax(hull_coords[:, 0])]
        sosi_coords = np.array([top, right, bottom, left], dtype=np.float32)

    # Match SOSI corners to GT corners using pairwise distances
    from scipy.spatial.distance import cdist
    geo_corners_arr = np.array(geo_corners, dtype=np.float32)
    dists = cdist(sosi_coords, geo_corners_arr)
    matches = np.argmin(dists, axis=1)

    print("Corner matching:")
    for i, j in enumerate(matches):
        sx, sy = sosi_coords[i]
        gx, gy = geo_corners[j]
        print(f"  SOSI[{i}] ({sx:.1f},{sy:.1f}) -> GT[{j}] ({gx:.1f},{gy:.1f})  dist={dists[i,j]:.1f}m")

    sosi_matched = sosi_coords  # unchanged order
    gt_matched = np.array([
        geo_corners[1],  # for SOSI[0]
        geo_corners[2],  # for SOSI[1]
        geo_corners[3],  # for SOSI[2]
        geo_corners[0],  # for SOSI[3]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(sosi_matched, gt_matched)

    # Normalize coordinates to 0-1 range to avoid floating point precision issues
    all_pts = np.vstack([sosi_matched, gt_matched])
    min_x, min_y = all_pts[:, 0].min(), all_pts[:, 1].min()
    max_x, max_y = all_pts[:, 0].max(), all_pts[:, 1].max()
    scale_n = max(max_x - min_x, max_y - min_y)

    def normalize(pts):
        return np.array([[(p[0] - min_x) / scale_n, (p[1] - min_y) / scale_n] for p in pts], dtype=np.float32)

    def denormalize(pts):
        return np.array([[p[0] * scale_n + min_x, p[1] * scale_n + min_y] for p in pts], dtype=np.float32)

    sosi_norm = normalize(sosi_matched)
    gt_norm = normalize(gt_matched)


    gt_center = np.array(geo_corners, dtype=np.float32).mean(axis=0)
    dists_corners = [dists[i, matches[i]] for i in range(4)]
    median_dist = np.median(dists_corners)


    gt_inset = gt_matched.copy()
    for i in range(4):
        dist = dists_corners[i]
        corner_inset = dist * 0.001
        gt_inset[i] = gt_matched[i] + (gt_center - gt_matched[i]) * corner_inset
        print(f"  Corner {i}: dist={dist:.1f}m, inset={corner_inset:.4f}")


    M_norm = cv2.getPerspectiveTransform(sosi_norm, normalize(gt_inset))

    print("Verification (normalized):")
    for i, pt in enumerate(sosi_norm):
        result = cv2.perspectiveTransform(np.array([[[pt[0], pt[1]]]], dtype=np.float32), M_norm)[0][0]
        gx, gy = gt_norm[i]
        print(f"  ({pt[0]:.4f},{pt[1]:.4f}) -> ({result[0]:.4f},{result[1]:.4f}) should be ({gx:.4f},{gy:.4f})")

    def perspective_correct(x_coords, y_coords, z_coords=None):
        pts = np.array([[x, y] for x, y in zip(x_coords, y_coords)], dtype=np.float32)
        pts_norm = (pts - np.array([min_x, min_y])) / scale_n
        corrected_norm = cv2.perspectiveTransform(pts_norm.reshape(-1, 1, 2), M_norm).reshape(-1, 2)
        corrected = corrected_norm * scale_n + np.array([min_x, min_y])
        return (corrected[:, 0], corrected[:, 1])

    merged_corrected = shapely_transform(perspective_correct, merged_clipped)
    merged_px = affine_transform(merged_corrected, [a, b, d, e, xoff, yoff])
    print("Pixel bounds:", merged_px.bounds)

    identity = af(1, 0, 0, 0, 1, 0)
    mask = rasterize(
        [(merged_px, 1)],
        out_shape=(height, width),
        transform=identity,
        fill=0,
        dtype=np.uint8
    )

    masked_img = img_data * mask[np.newaxis, ...]
    masked_img = np.ascontiguousarray(masked_img.transpose(1, 2, 0))
    plt.imshow(masked_img)
    plt.show()

def refine_mask_hsl(img_data, polygon_mask, increment=30):
    """
    Refines a polygon-based water mask using HSL-based water detection.
    Only keeps pixels where both the polygon mask and HSL detection agree.
    :param img_data: ndarray - image in shape (bands, H, W)
    :param polygon_mask: ndarray - binary mask from polygon, shape (H, W)
    :param increment: int - block size for HSL algorithm
    :return: refined mask shape (H, W)
    """
    # Run HSL water detection on the full image
    hsl_mask = create_water_mask_hsl_numba(img_data, increment)

    # Only keep pixels where both masks agree
    refined = (polygon_mask.astype(np.bool_)) & hsl_mask

    return refined.astype(np.uint8)
def main():
    """
    Function for testing water mask creation
    :return:
    """

    gdal.DontUseExceptions()
    path_gpkq = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\misc\Vann_22.gpkg"
    path_sosi = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\misc\HX-14365_Vertikalbilde.gpkg"
    #img_name = "HX-14365_073_014_14835.tif"
    img_name = "HX-14365_073_047_14868.tif"
    #img_name = "HX-14365_073_001_14822.tif"
    img_arr = tf.imread(
        r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\\" + img_name,
        maxworkers=8)
    print(img_arr.shape)
    img_arr = np.ascontiguousarray(img_arr.transpose(2, 0, 1))
    print(img_arr.shape)
    ds = load_geotiff_dataset(
        r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\\"+ img_name)

    before = datetime.now()
    mask = find_water_polygon_mask(path_gpkq, path_sosi, img_name, ds, img_arr)
    """ masked_img = img_arr * mask[np.newaxis, ...]
    masked_img = np.ascontiguousarray(masked_img.transpose(1, 2, 0))
    plt.imshow(masked_img)
    plt.show()
    mask = refine_mask_hsl(img_arr, mask, increment=30)
    after = datetime.now()
    print(after - before)
    mask = clean_water_mask(mask.astype(bool))
    rows, cols = np.nonzero(mask)
    masked_img = img_arr * mask[np.newaxis, ...]
    masked_img = np.ascontiguousarray(masked_img.transpose(1, 2, 0))
    masked_img = masked_img[rows.min():rows.max() + 1, cols.min():cols.max() + 1, :]
    plt.imshow(masked_img)
    plt.show()
    """

    #folder = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images"
    #folder = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\anomaly_images\Romsdal-2022-HX13173"
    #data, _ = load_geotiff(r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\HX-14365_073_047_14868.tif")

    """r = np.where(hsl_mask, data[0], 255)
    g = np.where(hsl_mask, data[1], 255)
    b = np.where(hsl_mask, data[2], 255)
    img = np.dstack((r, g, b))
    rows, cols = np.nonzero(hsl_mask)
    cropped = img[ rows.min():rows.max() + 1, cols.min():cols.max() + 1, : ]
    
    plt.imshow(cropped)
    plt.show()"""



