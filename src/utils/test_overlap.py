from find_overlap import get_overlap_pixel_images
from pathlib import Path
import cv2
import numpy as np

#TODO Refactor this into a proper unit test with pytest, and add more test cases for different images and edge cases (no overlap, full overlap, etc.)

gpkg_path = Path(__file__).parent / "file_test.gpkg"
base_path = Path(__file__).parent

img1_num, strip1 = 5, 1
img2_num, strip2 = 6, 1

SCALE = 0.15

# ------------------------------
# Step 1: Get pixel bounds of overlap
# ------------------------------
bounds1, bounds2 = get_overlap_pixel_images(
    gpkg_path,
    img1_num,
    strip1,
    img2_num,
    strip2
)

if bounds1 is None:
    print("No overlap")
    exit()

# ------------------------------
# Step 2: Load full images
# (replace with your actual paths)
# ------------------------------
img1_path = base_path / "HX-14365_001_005_00005.tif"
img2_path = base_path / "HX-14365_001_006_00006.tif"

img1 = cv2.imread(str(img1_path))
img2 = cv2.imread(str(img2_path))

if img1 is None or img2 is None:
    print("Could not load images")
    exit()

print("Full image 1 shape:", img1.shape)
print("Full image 2 shape:", img2.shape)

# ------------------------------
# Step 3: Crop overlap
# ------------------------------
x1_min, x1_max, y1_min, y1_max = bounds1
x2_min, x2_max, y2_min, y2_max = bounds2

overlap_img1 = img1[y1_min:y1_max, x1_min:x1_max]
overlap_img2 = img2[y2_min:y2_max, x2_min:x2_max]

# ------------------------------
# Step 4: Downscale full images for display
# ------------------------------
def resize(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

img1_small = resize(img1, SCALE)
img2_small = resize(img2, SCALE)

# Also downscale overlap images
overlap_img1_small = resize(overlap_img1, SCALE)
overlap_img2_small = resize(overlap_img2, SCALE)

# ------------------------------
# Step 5: Draw overlap bounds on full images (optional)
# ------------------------------
def draw_rect(img_small, bounds, scale):
    x_min, x_max, y_min, y_max = bounds
    top_left = (int(x_min * scale), int(y_min * scale))
    bottom_right = (int(x_max * scale), int(y_max * scale))
    cv2.rectangle(img_small, top_left, bottom_right, (0, 0, 255), 2)
    return img_small

img1_debug = draw_rect(img1_small.copy(), bounds1, SCALE)
img2_debug = draw_rect(img2_small.copy(), bounds2, SCALE)

# ------------------------------
# Step 6: Show images
# ------------------------------
cv2.imshow("Image 1 with Overlap", img1_debug)
cv2.imshow("Image 2 with Overlap", img2_debug)
cv2.waitKey(0)
cv2.destroyAllWindows()