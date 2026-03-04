# overlap_utils.py
import geopandas as gpd
from shapely.affinity import affine_transform
import cv2
from pathlib import Path

# ------------------------------------------------------------
# Image loader
# ------------------------------------------------------------
def load_image_safe(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV could not read image: {path}")

    return img

# ------------------------------------------------------------
# Affine transform
# ------------------------------------------------------------
def compute_affine_from_polygon(poly, width, height):
    coords = list(poly.exterior.coords)[:4]
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x3, y3 = coords[3]

    a = (x1 - x0) / width
    b = (x3 - x0) / height
    d = (y1 - y0) / width
    e = (y3 - y0) / height

    return [a, b, d, e, x0, y0]

def invert_affine(a, b, d, e, xoff, yoff):
    det = a * e - b * d
    inv_a = e / det
    inv_b = -b / det
    inv_d = -d / det
    inv_e = a / det
    inv_xoff = -(inv_a * xoff + inv_b * yoff)
    inv_yoff = -(inv_d * xoff + inv_e * yoff)
    return [inv_a, inv_b, inv_d, inv_e, inv_xoff, inv_yoff]

# ------------------------------------------------------------
# Polygon to bounding box
# ------------------------------------------------------------
def polygon_to_bbox(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    return int(minx), int(miny), int(maxx), int(maxy)

def safe_crop(img, x0, y0, x1, y1):
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img.shape[1], x1)
    y1 = min(img.shape[0], y1)
    return img[y0:y1, x0:x1]

# ------------------------------------------------------------
# Compute overlap in pixel coordinates
# ------------------------------------------------------------
def get_overlap_pixel_images(gpkg_path, img1_num, strip1, img2_num, strip2, base_path="."):
    gdf = gpd.read_file(gpkg_path, layer="polygons", encoding="ISO-8859-1")
    gdf["bildenummer"] = gdf["bildenummer"].astype(int)
    gdf["stripenummer"] = gdf["stripenummer"].astype(int)

    img1_row = gdf[(gdf["bildenummer"] == img1_num) & (gdf["stripenummer"] == strip1)].iloc[0]
    img2_row = gdf[(gdf["bildenummer"] == img2_num) & (gdf["stripenummer"] == strip2)].iloc[0]

    poly1, poly2 = img1_row.geometry, img2_row.geometry
    print("Image 1 polygon:", poly1)
    print("Image 2 polygon:", poly2)
    overlap_world = poly1.intersection(poly2)

    if overlap_world.is_empty:
        print("No overlap")
        return None, None

    # Build transforms
    width1, height1 = img1_row["ccdBrikkeside"], img1_row["ccdBrikkelengde"]
    width2, height2 = img2_row["ccdBrikkeside"], img2_row["ccdBrikkelengde"]

    aff1 = compute_affine_from_polygon(poly1, width1, height1)
    aff2 = compute_affine_from_polygon(poly2, width2, height2)
    inv_aff1 = invert_affine(*aff1)
    inv_aff2 = invert_affine(*aff2)

    # Map overlap to pixel coordinates
    overlap_px1 = affine_transform(overlap_world, inv_aff1)
    overlap_px2 = affine_transform(overlap_world, inv_aff2)

    # Load images
    img1 = load_image_safe(Path(base_path) / img1_row["bildefilRGB"])
    img2 = load_image_safe(Path(base_path) / img2_row["bildefilRGB"])

    # Crop overlapping areas
    x0, y0, x1, y1 = polygon_to_bbox(overlap_px1)
    overlap_img1 = safe_crop(img1, x0, y0, x1, y1)

    x0, y0, x1, y1 = polygon_to_bbox(overlap_px2)
    overlap_img2 = safe_crop(img2, x0, y0, x1, y1)

    # print("Poly1 area:", poly1.area)
    # print("Poly2 area:", poly2.area)
    # print("Overlap area:", overlap_world.area)

    # print("Overlap % of image 1:", overlap_world.area / poly1.area)
    # print("Overlap % of image 2:", overlap_world.area / poly2.area)
    
    return overlap_img1, overlap_img2, overlap_px1, overlap_px2, img1, img2
