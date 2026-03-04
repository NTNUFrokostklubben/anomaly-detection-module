from find_overlap import get_overlap_pixel_images
from pathlib import Path
import cv2
import numpy as np

gpkg_path = Path(__file__).parent / "file_test.gpkg"
base_path = Path(__file__).parent

img1_num, strip1 = 5, 1
img2_num, strip2 = 6, 1

SCALE = 0.15

result = get_overlap_pixel_images(
    gpkg_path,
    img1_num,
    strip1,
    img2_num,
    strip2,
    base_path
)

if result[0] is None:
    print("No overlap")
    exit()

overlap_img1, overlap_img2, overlap_px1, overlap_px2, img1, img2 = result

print("Full image 1 shape:", img1.shape)
print("Full image 2 shape:", img2.shape)

# ------------------------------------------------------------
# Downscale full images
# ------------------------------------------------------------
def resize(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )

img1_small = resize(img1, SCALE)
img2_small = resize(img2, SCALE)

# ------------------------------------------------------------
# Draw overlap polygon
# ------------------------------------------------------------
def draw_polygon(img_small, poly_px, scale):
    coords = np.array(poly_px.exterior.coords)
    coords = (coords * scale).astype(int)
    cv2.polylines(img_small, [coords], True, (0, 0, 255), 3)
    return img_small

img1_debug = draw_polygon(img1_small.copy(), overlap_px1, SCALE)
img2_debug = draw_polygon(img2_small.copy(), overlap_px2, SCALE)

cv2.imshow("Image 1 with Overlap", img1_debug)
cv2.imshow("Image 2 with Overlap", img2_debug)

cv2.waitKey(0)
cv2.destroyAllWindows()