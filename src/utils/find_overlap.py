from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from skimage.transform import AffineTransform

from services.config_parser.ConfigHandler import Config


def find_image_row(gdf: gpd.GeoDataFrame, img_num: int, strip_num: int):
    """
    Find the row data for a given image number and strip number from the GeoPackage
    Args:
        gdf (GeoDataFrame): GeoDataFrame containing the image metadata:
        img_num (int): image number to find
        strip_num (int): strip number to find

    Returns:
        row data for the given image number and strip number from the GeoPackage

    """
    config = Config()
    gdf[config.get("sosi_column_headers", "image_number_column_header")] = gdf[config.get("sosi_column_headers", "image_number_column_header")].astype(int)
    gdf[config.get("sosi_column_headers", "image_stripe_column_header")] = gdf[config.get("sosi_column_headers", "image_stripe_column_header")].astype(int)

    row = gdf[
        (gdf[config.get("sosi_column_headers", "image_number_column_header")] == img_num) &
        (gdf[config.get("sosi_column_headers", "image_stripe_column_header")] == strip_num)
    ].iloc[0]    

    if row.empty:
        raise ValueError(f"Image with number {img_num} and strip {strip_num} not found")
    
    return row


def find_image_row_img_name(gdf: gpd.GeoDataFrame , img_name: str):
    """
    Finds an image in a GeoDataFrame from a converted sosi file based on  image name

    Args:
        gdf: geo dataframe that contains rows for images
        img_name: the image name for the image.

    Returns:
        the row that matches the image name.
    """
    config = Config()
    matches = gdf[gdf[config.get("sosi_column_headers", "image_path_column_header")] == img_name]

    if matches.empty:
        raise ValueError(f"Image with name {img_name} not found")
    row = matches.iloc[0]
    return row



def find_image_from_gpkg(gdf: gpd.GeoDataFrame, img_num: int, strip_num: int) -> tuple[Polygon, int, int]:
    """
    Find the polygon for a given image number and strip number from the GeoPackage
    Returns the polygon and the image dimensions (width, height) as integers
    
    Args:
        gdf (_type_): polygon geometry for the image footprint
        img_num (int): width of the image in pixels
        strip_num (int): Strip number for the image

    Returns:
        bounds for the overlapping region in pixel coordinates for both images, as tuples (min_x, max_x, min_y, max_y)
    """
    config = Config()
    row = find_image_row(gdf, img_num, strip_num)
    poly = row.geometry
    width = int(row[config.get("sosi_column_headers", "image_shape_column_header")])
    height = int(row[config.get("sosi_column_headers", "image_shape_column_header")])

    return poly, width, height

def build_transform_from_polygon(poly: Polygon | MultiPolygon, width: int, height: int) -> AffineTransform:
    """
    Build affine transform from world coordinates -> pixel coordinates
    Assumes poly.exterior.coords order:
    [bottom-right, top-right, top-left, bottom-left]

    Args:
        poly (Polygon | MultiPolygon): polygon geometry for the image footprint
        width (int): width of the image in pixels
        height (int): height of the image in pixels

    Returns:
        AffineTransform object that maps world coordinates to pixel coordinates
    """
    if isinstance(poly, MultiPolygon):
        poly = poly.geoms[0]

    world_coords = np.array(poly.exterior.coords[:-1])  # remove duplicate last point
    if len(world_coords) != 4:
        raise ValueError("Polygon must have exactly 4 corners")

    # Reorder to match pixel coordinates
    world_coords_ordered = np.array([
        world_coords[2],  # top-left
        world_coords[1],  # top-right
        world_coords[0],  # bottom-right
        world_coords[3]   # bottom-left
    ])
    
    # Pixel coordinates
    pixel_coords = np.array([
        [0, 0],             # top-left
        [width, 0],         # top-right
        [width, height],    # bottom-right
        [0, height]         # bottom-left
    ])

    tform = AffineTransform.from_estimate(
        world_coords_ordered,
        pixel_coords
    )

    return tform


def get_bounds(px_coords: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """
    Get safe integer crop bounds

    Args:
        px_coords (np.ndarray): Nx2 array of pixel coordinates for the overlap polygon
        width (int): width of the image in pixels
        height (int): height of the image in pixels

    Returns:
        bounds for the overlapping region in pixel coordinates, as tuples (min_x, max_x, min_y, max_y)
    """
    min_x = int(np.floor(px_coords[:, 0].min()))
    max_x = int(np.ceil(px_coords[:, 0].max()))
    min_y = int(np.floor(px_coords[:, 1].min()))
    max_y = int(np.ceil(px_coords[:, 1].max()))

    # Clamp to image size
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(width, max_x)
    max_y = min(height, max_y)

    return min_x, max_x, min_y, max_y


def get_overlap_pixel_images(gdf: gpd.GeoDataFrame, img1_num: int, strip1: int, img2_num: int, strip2: int) \
        -> tuple[ None, None] |  tuple[tuple[int, int, int, int],tuple[int, int, int, int]]:
    """ Find the pixel bounds of the overlapping region between two images defined in a GeoPackage

    Args:
        gdf (str): path to the GeoPackage file
        img1_num (int): image number for the first image
        strip1 (int): strip number for the first image
        img2_num (int): image number for the second image
        strip2 (int): strip number for the second image

    Returns:
        bounds for the overlapping region in pixel coordinates for both images, as tuples (min_x, max_x, min_y, max_y)
    """

    poly1, width1, height1 = find_image_from_gpkg(gdf, img1_num, strip1)
    poly2, width2, height2 = find_image_from_gpkg(gdf, img2_num, strip2)
    
    # Compute world overlap
    overlap_world = poly1.intersection(poly2)

    if overlap_world.is_empty:
        return None, None

    # Build affine transforms
    tform1 = build_transform_from_polygon(poly1, width1, height1)
    tform2 = build_transform_from_polygon(poly2, width2, height2)

    # Convert overlap polygon to pixel space
    overlap_geom = overlap_world.geoms[0] if isinstance(overlap_world, MultiPolygon) else overlap_world
    overlap_coords = np.array(overlap_geom.exterior.coords)

    # After transforming the overlap polygon
    overlap_px1 = tform1(overlap_coords)
    overlap_px2 = tform2(overlap_coords)

    # Flip y-axis for NumPy/OpenCV image coordinates
    overlap_px1[:, 1] = height1 - overlap_px1[:, 1]
    overlap_px2[:, 1] = height2 - overlap_px2[:, 1]

    # Compute crop bounds
    bounds1 = get_bounds(overlap_px1, width1, height1)
    bounds2 = get_bounds(overlap_px2, width2, height2)

    return bounds1, bounds2