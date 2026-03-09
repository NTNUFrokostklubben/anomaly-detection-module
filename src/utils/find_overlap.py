from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from skimage.transform import AffineTransform

def find_image_row(gdf, img_num, strip_num):
    gdf["bildenummer"] = gdf["bildenummer"].astype(int)
    gdf["stripenummer"] = gdf["stripenummer"].astype(int)

    row = gdf[
        (gdf["bildenummer"] == img_num) &
        (gdf["stripenummer"] == strip_num)
    ].iloc[0]    

    if row.empty:
        raise ValueError(f"Image with number {img_num} and strip {strip_num} not found")
    
    return row

def find_image_path(gdf, img_num, strip_num):
    """
    Find the file path for a given image number and strip number from the GeoPackage

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing the image metadata
        img_num (int): image number to find
        strip_num (int): strip number to find
    Returns:
        str: file path for the image
    """
    row = find_image_row(gdf, img_num, strip_num)
    
    return row["bildefilRGB"]    

def find_image_from_gpkg(gdf, img_num, strip_num):
    """
    Find the polygon for a given image number and strip number from the GeoPackage
    Returns the polygon and the image dimensions (width, height) as integers
    
    Args:
        gdf (_type_): polygon geometry for the image footprint
        img_num (int): witdh of the image in pixels
        strip_num (int): Strip number for the image

    Returns:
        _type_: bounds for the overlapping region in pixel coordinates for both images, as tuples (min_x, max_x, min_y, max_y)
    """
    row = find_image_row(gdf, img_num, strip_num)
    
    return row.geometry, int(row["ccdBrikkeside"]), int(row["ccdBrikkelengde"])

def build_transform_from_polygon(poly, width, height):
    """
    Build affine transform from world coordinates -> pixel coordinates
    Assumes poly.exterior.coords order:
    [bottom-right, top-right, top-left, bottom-left]
    
    Args:
        poly (Polygon): polygon geometry for the image footprint
        width (int): witdh of the image in pixels
        height (int): height of the image in pixels

    Returns:
        AffineTransform: AffineTransform object that maps world coordinates to pixel coordinates
    """

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


def get_bounds(px_coords, width, height):
    """
    Get safe integer crop bounds

    Args:
        px_coords (np.ndarray): Nx2 array of pixel coordinates for the overlap polygon
        width (int): width of the image in pixels
        height (int): height of the image in pixels

    Returns:
        tuple[int, int, int, int]: bounds for the overlapping region in pixel coordinates, as tuples (min_x, max_x, min_y, max_y)
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


def get_overlap_pixel_images(gdf, img1_num, strip1, img2_num, strip2):
    """ Find the pixel bounds of the overlapping region between two images defined in a GeoPackage

    Args:
        gpkg_path (str): path to the GeoPackage file
        img1_num (int): image number for the first image
        strip1 (int): strip number for the first image
        img2_num (int): image number for the second image
        strip2 (int): strip number for the second image

    Returns:
        tuple[int, int, int, int]: bounds for the overlapping region in pixel coordinates for both images, as tuples (min_x, max_x, min_y, max_y)
    """

    poly1, width1, height1 = find_image_from_gpkg(gdf, img1_num, strip1)
    poly2, width2, height2 = find_image_from_gpkg(gdf, img2_num, strip2)
    
    # Compute world overlap
    overlap_world = poly1.intersection(poly2)

    if overlap_world.is_empty:
        print("No overlap found.")
        return None, None

    # Build affine transforms
    tform1 = build_transform_from_polygon(poly1, width1, height1)
    tform2 = build_transform_from_polygon(poly2, width2, height2)

    # Convert overlap polygon to pixel space
    overlap_coords = np.array(overlap_world.exterior.coords)

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