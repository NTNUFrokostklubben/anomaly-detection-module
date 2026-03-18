
import math
from typing import Any

import geopandas as gp
import cv2
from numba import njit, prange, cuda
import numpy as np
from numpy import ndarray
from osgeo import gdal
from skimage import morphology
from scipy import ndimage
from shapely.ops import unary_union
from affine import Affine
from rasterio.features import geometry_mask
from shapely.geometry import Polygon, MultiPolygon



# RGB normalisation
RGB_MAX: float = 255.0

# HSL hue computation
HUE_SECTOR_DEGREES: float = 60.0
HUE_GREEN_OFFSET: float = 120.0
HUE_BLUE_OFFSET: float = 240.0
HUE_FULL_CIRCLE: float = 360.0

# Water detection hue range (blue-cyan band)
WATER_HUE_MIN: float = 170.0
WATER_HUE_MAX: float = 290.0
WATER_CHROMA_MIN: float = 6.0

# CUDA kernel thread block size (per axis)
CUDA_THREADS_PER_BLOCK: int = 16

# Mask cleaning thresholds (pixels)
CLEAN_MASK_MAX_SPLOTCH_SIZE: int = 1_000_000
CLEAN_MASK_MAX_HOLE_SIZE: int = 300_000


def load_geotiff_dataset(path: str) ->  gdal.Dataset:
    """
    Load geotiff image into memory. Temporary function

    :param path: path to the tiff image
    :return: the image as array in shape(bands, H, W) and the gdal dataset.
    """
    ds = gdal.OpenEx(path)
    print("fin read")
    return ds


def _find_image_row(gdf: gp.GeoDataFrame , img_name: str):
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



@njit(cache=True)
def _block_has_mask(polygon_mask: np.ndarray, y_start: int, y_end: int, x_start: int, x_end: int) -> bool:
    """
    Checks whether the given block contains at least one pixel inside the polygon mask.

    :param polygon_mask: 2D boolean array where True marks pixels inside the polygon.
    :param y_start: first row index of the block (inclusive).
    :param y_end: last row index of the block (exclusive).
    :param x_start: first column index of the block (inclusive).
    :param x_end: last column index of the block (exclusive).
    :return: True if any pixel in the block is inside the polygon mask, False otherwise.
    """
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            if polygon_mask[y, x]:
                return True
    return False


@njit(cache=True)
def _block_mean_rgb(data: np.ndarray, polygon_mask: np.ndarray, y_start: int, y_end: int, x_start: int, x_end: int) -> tuple:
    """
    Computes the mean normalised RGB values for all masked pixels within the block.

    :param data: image array of shape (3, H, W) with uint8 pixel values.
    :param polygon_mask: 2D boolean array where True marks pixels inside the polygon.
    :param y_start: first row index of the block (inclusive).
    :param y_end: last row index of the block (exclusive).
    :param x_start: first column index of the block (inclusive).
    :param x_end: last column index of the block (exclusive).
    :return: tuple (r_mean, g_mean, b_mean) in the range [0.0, 1.0].
    """
    r_sum = 0.0
    g_sum = 0.0
    b_sum = 0.0
    count = 0
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            if polygon_mask[y, x]:
                r_sum += data[0, y, x]
                g_sum += data[1, y, x]
                b_sum += data[2, y, x]
                count += 1
    r_mean = (r_sum / count) / RGB_MAX
    g_mean = (g_sum / count) / RGB_MAX
    b_mean = (b_sum / count) / RGB_MAX
    return r_mean, g_mean, b_mean


@njit(cache=True)
def _rgb_to_hue(r_mean: float, g_mean: float, b_mean: float) -> tuple[float, float]:
    """
    Converts normalized RGB values to an HSL hue angle in degrees [0, 360) and chroma [0, 100].

    :param r_mean: normalized red channel value in [0.0, 1.0].
    :param g_mean: normalized green channel value in [0.0, 1.0].
    :param b_mean: normalized blue channel value in [0.0, 1.0].
    :return: tuple of (hue angle in degrees, chroma in [0, 100]). Hue is 0.0 for achromatic colors.
    """
    max_c = max(r_mean, g_mean, b_mean)
    min_c = min(r_mean, g_mean, b_mean)
    delta = max_c - min_c
    chroma = delta * 100.0
    if delta == 0:
        return 0.0, chroma
    if max_c == r_mean:
        return (HUE_SECTOR_DEGREES * ((g_mean - b_mean) / delta)) % HUE_FULL_CIRCLE, chroma
    if max_c == g_mean:
        return HUE_SECTOR_DEGREES * ((b_mean - r_mean) / delta) + HUE_GREEN_OFFSET, chroma
    return HUE_SECTOR_DEGREES * ((r_mean - g_mean) / delta) + HUE_BLUE_OFFSET, chroma


@njit(cache=True)
def _find_coast_x(data: np.ndarray, polygon_mask: np.ndarray, y_start: int, y_end: int, x_start: int, x_end: int) -> int:
    """
    Finds the x-coordinate of the first column in the block where blue dominates,
    indicating the coastline transition from land to water.

    :param data: image array of shape (3, H, W) with uint8 pixel values.
    :param polygon_mask: 2D boolean array where True marks pixels inside the polygon.
    :param y_start: first row index of the block (inclusive).
    :param y_end: last row index of the block (exclusive).
    :param x_start: first column index of the block (inclusive).
    :param x_end: last column index of the block (exclusive).
    :return: column index of the detected coastline, or x_end if none is found.
    """
    for i in range(x_start, x_end):
        r_col = 0.0
        g_col = 0.0
        b_col = 0.0
        col_count = 0
        for y in range(y_start, y_end):
            if polygon_mask[y, i]:
                r_col += data[0, y, i]
                g_col += data[1, y, i]
                b_col += data[2, y, i]
                col_count += 1
        if col_count > 0 and b_col > r_col and b_col > g_col:
            return i
    return x_end


@njit(cache=True)
def _fill_block(mask: np.ndarray, polygon_mask: np.ndarray, y_start: int, y_end: int, x_start: int, x_end: int) -> None:
    """
    Sets all polygon-masked pixels within the block to True in the output mask.

    :param mask: 2D boolean output array to write water pixels into.
    :param polygon_mask: 2D boolean array where True marks pixels inside the polygon.
    :param y_start: first row index of the block (inclusive).
    :param y_end: last row index of the block (exclusive).
    :param x_start: first column index of the block (inclusive).
    :param x_end: last column index of the block (exclusive).
    """
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            if polygon_mask[y, x]:
                mask[y, x] = True


@njit(parallel=True, cache=True)
def create_water_mask_hsl(data: np.ndarray[tuple[int, int, int]], increment: int, *, constraint_region: np.ndarray[tuple[bool, bool]]) -> np.ndarray:
    """
    Create a mask outlining the water on an image using a jumping block algorithm. Optionally allows for looking
     only within a constrained region of the full image.

    :param data: the image array
    :param increment: the amount the block should jump. Not max pixels, the size of the square is increment squared.
    :param constraint_region: the region to not run the algortithm in.
    :return: a mask outlining water for the corresponding image.
    """
    y_shape = data.shape[1]
    x_shape = data.shape[2]
    if constraint_region is None:
        constraint_region = np.ones((data.shape[1], data.shape[2]), dtype=np.bool_)

    mask = np.zeros((y_shape, x_shape), dtype=np.bool_)

    y_blocks = (y_shape + increment - 1) // increment
    x_blocks = (x_shape + increment - 1) // increment

    for by in prange(y_blocks):
        previous = False
        for bx in range(x_blocks):
            y_start = by * increment
            x_start = bx * increment
            y_end = min(y_start + increment, y_shape)
            x_end = min(x_start + increment, x_shape)

            if not _block_has_mask(constraint_region, y_start, y_end, x_start, x_end):
                previous = False
                continue

            r_mean, g_mean, b_mean = _block_mean_rgb(data, constraint_region, y_start, y_end, x_start, x_end)
            h, chroma = _rgb_to_hue(r_mean, g_mean, b_mean)

            if WATER_HUE_MIN < h < WATER_HUE_MAX and chroma > WATER_CHROMA_MIN:
                if not previous:
                    x_start = _find_coast_x(data, constraint_region, y_start, y_end, x_start, x_end)
                _fill_block(mask, constraint_region, y_start, y_end, x_start, x_end)
                previous = True
            else:
                previous = False

    return mask


@cuda.jit
def _hsl_compute_blocks_kernel(data: np.ndarray[tuple[int, int, int]], mask: np.ndarray[tuple[bool, bool]], increment: int):
    """
    Helper function for computing the blocks in the jumping block algorithm for CUDA processing of the water mask.

    :param data: the image array
    :param mask:
    :param increment:
    :return:
    """
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

    r_mean = (r_sum / count) / RGB_MAX
    g_mean = (g_sum / count) / RGB_MAX
    b_mean = (b_sum / count) / RGB_MAX

    max_c = max(r_mean, g_mean, b_mean)
    min_c = min(r_mean, g_mean, b_mean)
    delta = max_c - min_c

    h = 0.0
    if delta > 0:
        if max_c == r_mean:
            h = (HUE_SECTOR_DEGREES * ((g_mean - b_mean) / delta)) % HUE_FULL_CIRCLE
        elif max_c == g_mean:
            h = HUE_SECTOR_DEGREES * ((b_mean - r_mean) / delta) + HUE_GREEN_OFFSET
        else:
            h = HUE_SECTOR_DEGREES * ((r_mean - g_mean) / delta) + HUE_BLUE_OFFSET

    if WATER_HUE_MIN < h < WATER_HUE_MAX:
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                mask[y, x] = True

def create_water_mask_hsl_cuda(data: ndarray[tuple[int, int, int]], increment: int) -> ndarray[tuple[bool]]:
    """
    Computes the water mask in HSL using cuda cores, slower than CPU processing if processor is of similar quality as
     GPU.

    :param data: the image array
    :param increment: The size of the square to compute water detection on. Not total pixels.
    :return: the mask outlining water.
    """
    y_shape = data.shape[1]
    x_shape = data.shape[2]
    y_blocks = (y_shape + increment - 1) // increment
    x_blocks = (x_shape + increment - 1) // increment

    data_gpu = cuda.to_device(data)
    mask_gpu = cuda.to_device(np.zeros((y_shape, x_shape), dtype=np.bool_))

    threads_2d = (CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK)
    blocks_2d = (math.ceil(x_blocks / CUDA_THREADS_PER_BLOCK), math.ceil(y_blocks / CUDA_THREADS_PER_BLOCK))
    _hsl_compute_blocks_kernel[blocks_2d, threads_2d](data_gpu, mask_gpu, increment)

    return mask_gpu.copy_to_host()


def clean_water_mask(mask_array: ndarray[tuple[int, int]], max_size=CLEAN_MASK_MAX_SPLOTCH_SIZE) -> ndarray[tuple[int, int]]:
    """
    Remove shadow splotches from a water mask ndarray.

    :param mask_array: ndarray as type bool or binary
    :param max_size: int - maximum size of the splotches to remove in pixels, default 500,000

    :return: cleaned: ndarray (bool) - True where water, False elsewhere
    """

    cleaned = morphology.remove_small_objects(mask_array, max_size=max_size)
    return cleaned




def detect_holes(mask: ndarray[tuple[bool, bool]]) -> ndarray[tuple[int, int]]:
    """
    Detects holes in masks. Optimized for large images like tif.

    :param mask: the mask array to detect holes in.
    """
    max_size = CLEAN_MASK_MAX_HOLE_SIZE
    filled_mask = ndimage.binary_fill_holes(mask)
    holes = filled_mask ^ mask
    cleaned = morphology.remove_small_objects(holes, max_size=max_size)
    labeled_holes, _ = ndimage.label(cleaned)
    return labeled_holes




def create_water_polygon_mask(contour_gdf: gp.GeoDataFrame, sosi_df: gp.GeoDataFrame, img_name: str, ds: gdal.Dataset) -> np.ndarray:
    """
    Builds a water mask for the given image by aligning water contours from a GeoPackage
    to the raster extent, correcting for Y-axis flip in the SOSI boundary polygon.

    :param sosi_df: the GeoDataFrame containing the image polygon.
    :param contour_gdf: the GeoDataFrame containing the contour polygons.
    :param img_name: Image name used to look up the corresponding SOSI boundary row.
    :param ds: GDAL dataset of the raster image.
    :return: Boolean mask array of shape (height, width), True where water is present.
    """

    raster_crs = ds.GetProjection()
    if not raster_crs:
        raise RuntimeError("Raster has no CRS; can't align vectors.")

    contour_gdf = contour_gdf.to_crs(raster_crs)
    sosi_df = sosi_df.to_crs(raster_crs)

    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    affine = Affine.from_gdal(*gt)
    inv_affine = ~affine

    row = __find_image_row(sosi_df, img_name)
    overlap = contour_gdf['geometry'].intersects(row['geometry'])

    sosi_corners_flat = [
        pt
        for ring in [list(geom.exterior.coords)[:-1] for geom in row['geometry'].geoms]
        for pt in ring
    ]
    sosi_px = [(geo_to_pixel(x, y, inv_affine), (x, y)) for x, y in sosi_corners_flat]

    image_corners_px = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    src_pts = np.float32([
        min(sosi_px, key=lambda p, _ic=ic: np.linalg.norm(np.array(p[0]) - _ic))[0]
        for ic in image_corners_px
    ])

    sosi_ul, sosi_ur, sosi_lr, sosi_ll = src_pts
    raster_ul = np.array([0, 0])
    raster_ur = np.array([width, 0])
    raster_lr = np.array([width, height])
    raster_ll = np.array([0, height])

    # Flips the polygon along the x-axis due to suspected mirroring of the polygon from sosi geopackage.
    dst_pts = np.float32([
        sosi_ul + flip_y(raster_ll - sosi_ll),
        sosi_ur + flip_y(raster_lr - sosi_lr),
        sosi_lr + flip_y(raster_ur - sosi_ur),
        sosi_ll + flip_y(raster_ul - sosi_ul),
    ])

    transform = cv2.getPerspectiveTransform(src_pts, dst_pts)

    corrected_geoms = contour_gdf[overlap]['geometry'].apply(lambda g: apply_homography(g, transform, affine).buffer(0))
    merged = unary_union(corrected_geoms)

    return geometry_mask(
        [merged],
        transform=affine,
        invert=True,
        out_shape=(height, width)
    )


def geo_to_pixel(x:float, y:float, inv_affine: Affine) -> list[Any]:
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


def apply_homography(geom: Polygon | MultiPolygon, transform: np.ndarray, affine: Affine) -> Polygon | MultiPolygon:
    """
    Applies a perspective homography to a Shapely geometry in geographic coordinates.
    Converts each vertex to pixel space, applies the transform, then converts back to
    geographic coordinates using the affine transformation.

    :param geom: the input geometry (Polygon or MultiPolygon) in geographic coordinates.
    :param transform: 3x3 perspective transformation matrix from cv2.getPerspectiveTransform.
    :param affine: affine transformation mapping pixel coordinates to geographic coordinates.
    :return: transformed geometry in geographic coordinates.
    """

    def transform_polygon(poly):
        coords = np.array(poly.exterior.coords)[:, :2]
        px_coords = np.float32([geo_to_pixel(x, y, ~affine) for x, y in coords]).reshape(-1, 1, 2)
        transformed_px = cv2.perspectiveTransform(px_coords, transform).reshape(-1, 2)
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
    refined_mask = create_water_mask_hsl(img_data, increment, polygon_mask)

    # Only keep pixels where both masks agree


    return  refined_mask.astype(np.uint8)

def find_disagreement_ratio(mask: ndarray[tuple[bool, bool]], other_mask: ndarray[tuple[bool, bool]]) -> float:
    """
    Finds the amount of disagreement between two binary masks. The stricter of the two masks should be in other_mask.

    :param mask: The first binary mask
    :param other_mask: the other binary mask.
    :return: the disagreement ratio normalized to [0, 1]
    """
    disagreement = mask.astype(np.bool_) ^ other_mask.astype(np.bool_)
    disagreement_count = np.sum(disagreement)
    return disagreement_count / mask.size



