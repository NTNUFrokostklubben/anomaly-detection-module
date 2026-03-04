from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from skimage.transform import AffineTransform


def polygon_to_bbox(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    return int(minx), int(miny), int(maxx), int(maxy)

def safe_crop(img, x0, y0, x1, y1):
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img.shape[1], x1)
    y1 = min(img.shape[0], y1)
    return img[y0:y1, x0:x1]


def build_transform_from_polygon(poly, width, height):
    """
    Build affine transform from world coordinates -> pixel coordinates
    Assumes poly.exterior.coords order:
    [bottom-right, top-right, top-left, bottom-left]
    """

    world_coords = np.array(poly.exterior.coords[:-1])  # remove duplicate last point
    if len(world_coords) != 4:
        raise ValueError("Polygon must have exactly 4 corners")

    # Reorder to match pixel coordinates
    # Pixel coordinates are:
    # [top-left, top-right, bottom-right, bottom-left]
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

    tform = AffineTransform()
    success = tform.estimate(world_coords_ordered, pixel_coords)

    if not success:
        raise RuntimeError("Affine transform estimation failed")

    return tform


def get_bounds(px_coords, width, height):
    """
    Get safe integer crop bounds
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


def get_overlap_pixel_images(gpkg_path, img1_num, strip1, img2_num, strip2):
    gdf = gpd.read_file(gpkg_path, layer="polygons", encoding="ISO-8859-1")

    gdf["bildenummer"] = gdf["bildenummer"].astype(int)
    gdf["stripenummer"] = gdf["stripenummer"].astype(int)

    # Select images
    img1_row = gdf[
        (gdf["bildenummer"] == img1_num) &
        (gdf["stripenummer"] == strip1)
    ].iloc[0]

    img2_row = gdf[
        (gdf["bildenummer"] == img2_num) &
        (gdf["stripenummer"] == strip2)
    ].iloc[0]

    poly1 = img1_row.geometry
    poly2 = img2_row.geometry

    print("Image 1 polygon:", poly1)
    print("Image 2 polygon:", poly2)

    # Compute world overlap
    overlap_world = poly1.intersection(poly2)

    print("Overlap polygon:", overlap_world)

    if overlap_world.is_empty:
        print("No overlap found.")
        return None, None

    # Image pixel dimensions
    width1 = int(img1_row["ccdBrikkeside"])
    height1 = int(img1_row["ccdBrikkelengde"])

    width2 = int(img2_row["ccdBrikkeside"])
    height2 = int(img2_row["ccdBrikkelengde"])

    print("Image1 size:", width1, height1)
    print("Image2 size:", width2, height2)

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

    print("Image1 pixel bounds:", bounds1)
    print("Image2 pixel bounds:", bounds2)

    return bounds1, bounds2

