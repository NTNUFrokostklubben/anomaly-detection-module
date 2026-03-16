
import math
import geopandas as gp
from datetime import datetime
import cv2
import tifffile as tf
from numba import njit, prange, cuda
import numpy as np
from numpy import ndarray
from osgeo import gdal
from skimage import morphology
from scipy import ndimage
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.ops import unary_union
from affine import Affine as af, Affine
from rasterio.features import geometry_mask
from shapely.geometry import Polygon, MultiPolygon



def load_geotiff_dataset(path: str) ->  gdal.Dataset:
    """
    Load geotiff image into memory. Temporary function
    :param path: path to the tiff image
    :return: the image as array in shape(bands, H, W) and the gdal dataset.
    """
    ds = gdal.OpenEx(path)
    print("fin read")
    return ds


def __find_image_row(gdf: gp.GeoDataFrame , img_name: str):
    """
    Temporary function until utils are pushed to develop
    :param gdf: geo dataframe that contains rows for images
    :param img_name: the image name for the image.
    :return: the row that matches the image name.
    """
    matches = gdf[gdf["bildefilRGB"] == img_name]

    if matches.empty:
        raise ValueError(f"Image with name {img_name} not found")

    return matches.iloc[0]


@njit(parallel=True,cache=True)
def create_water_mask_hsl_numba(data: np.ndarray, increment: int) -> np.ndarray[tuple[int, int]]:
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
def _hsl_compute_blocks_kernel(data: np.ndarray, mask: np.ndarray[tuple[int, int]], increment: int):
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


def create_water_mask_hsl_cuda(data: ndarray[tuple[int, int, int]], increment: int) -> ndarray[tuple[bool]]:
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


def clean_water_mask(mask_array: ndarray[tuple[bool]], max_size=1000000) -> ndarray[tuple[bool]]:
    """
    Remove shadow splotches from a water mask ndarray.
    :param mask_array: ndarray as type bool or binary
    :param max_size: int - maximum size of the splotches to remove in pixels, default 500,000

    Returns:
        cleaned: ndarray (bool) - True where water, False elsewhere
    """

    cleaned = morphology.remove_small_objects(mask_array, max_size=max_size)
    return cleaned




def detect_holes(mask: ndarray[tuple[bool]]) -> ndarray[tuple[bool]]:
    """
    Detects holes in masks. Optimized for large images like tif.
    :param mask: the mask to detect holes in.
    """
    max_size = 300000
    filled_mask = ndimage.binary_fill_holes(mask)
    holes = filled_mask ^ mask
    cleaned = morphology.remove_small_objects(holes, max_size=max_size)
    labeled_holes, _ = ndimage.label(cleaned)
    return labeled_holes




def find_water_polygon_mask(gpkg_path: str, sosi_path: str, img_name: str, ds: gdal.Dataset) -> np.ndarray:
    """
    Builds a water mask for the given image by aligning water contours from a GeoPackage
    to the raster extent, correcting for Y-axis flip in the SOSI boundary polygon.

    :param gpkg_path: Path to GeoPackage containing water contour polygons
    :param sosi_path: Path to SOSI file containing image boundary polygons
    :param img_name: Image name used to look up the corresponding SOSI boundary row
    :param ds: GDAL dataset of the raster image
    :return: Boolean mask array of shape (height, width), True where water is present
    """
    gdf: gp.GeoDataFrame = gpd.read_file(gpkg_path, layer="polygons")
    sosidf: gp.GeoDataFrame = gpd.read_file(sosi_path, layer="polygons")

    raster_crs = ds.GetProjection()
    if not raster_crs:
        raise RuntimeError("Raster has no CRS; can't align vectors.")

    gdf = gdf.to_crs(raster_crs)
    sosidf = sosidf.to_crs(raster_crs)

    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    affine = af.from_gdal(*gt)
    inv_affine = ~affine

    row = __find_image_row(sosidf, img_name)
    overlap = gdf['geometry'].intersects(row['geometry'])

    sosi_corners_flat = [
        pt
        for ring in [list(geom.exterior.coords)[:-1] for geom in row['geometry'].geoms]
        for pt in ring
    ]
    sosi_px = [(geo_to_pixel(x, y, inv_affine), (x, y)) for x, y in sosi_corners_flat]

    image_corners_px = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    src_pts = np.float32([
        min(sosi_px, key=lambda p: np.linalg.norm(np.array(p[0]) - ic))[0]
        for ic in image_corners_px
    ])

    sosi_ul, sosi_ur, sosi_lr, sosi_ll = src_pts
    r_ul = np.array([0, 0])
    r_ur = np.array([width, 0])
    r_lr = np.array([width, height])
    r_ll = np.array([0, height])

    dst_pts = np.float32([
        sosi_ul + flip_y(r_ll - sosi_ll),
        sosi_ur + flip_y(r_lr - sosi_lr),
        sosi_lr + flip_y(r_ur - sosi_ur),
        sosi_ll + flip_y(r_ul - sosi_ul),
    ])

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    corrected_geoms = gdf[overlap]['geometry'].apply(lambda g: apply_homography(g, H, affine))
    merged = unary_union(corrected_geoms)

    return geometry_mask(
        [merged],
        transform=affine,
        invert=True,
        out_shape=(height, width)
    )


def geo_to_pixel(x:float, y:float, inv_affine: Affine) -> tuple[float, float]:
    """
    Turns geographical coordinates into pixel coordinates
    :param x: the x coordinate
    :param y: the y coordinate
    :param inv_affine: the inverted affine transformation matrix
    :return: the pixel coordinates
    """
    col, row_p = inv_affine * (x, y)
    return [col, row_p]

def flip_y(v: np.ndarray) -> np.ndarray:
    """
    Flips a vector along the x-axis, used to correct for the Y-axis flip in the SOSI boundary polygon.
    :param v: vector to flip
    :return: flipped vector
    """
    return np.array([v[0], -v[1]])


def apply_homography(geom, h: np.ndarray, affine: Affine) -> Polygon | MultiPolygon:
    def transform_polygon(poly):
        coords = np.array(poly.exterior.coords)[:, :2]
        px_coords = np.float32([geo_to_pixel(x, y, ~affine) for x, y in coords]).reshape(-1, 1, 2)
        transformed_px = cv2.perspectiveTransform(px_coords, h).reshape(-1, 2)
        geo_coords = [affine * (c, r) for c, r in transformed_px]
        return Polygon(geo_coords)

    if isinstance(geom, MultiPolygon):
        return MultiPolygon([transform_polygon(p) for p in geom.geoms])
    return transform_polygon(geom)


def refine_mask_hsl(img_data: ndarray, polygon_mask: ndarray[tuple[int, int]], increment: int) -> ndarray:
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
    mask = find_water_polygon_mask(path_gpkq, path_sosi, img_name, ds)
    mask = clean_water_mask(mask)
    rows, cols = np.nonzero(mask)
    mask = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    cropped_img = img_arr[:, rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    mask = refine_mask_hsl(cropped_img, mask, 30)
    after = datetime.now()
    print(after - before)


    masked_img = cropped_img * mask[np.newaxis, ...]
    masked_img = np.ascontiguousarray(masked_img.transpose(1, 2, 0))
    plt.imshow(masked_img)
    plt.show()



