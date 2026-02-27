import numpy as np
from osgeo import gdal
import cv2
from scipy.stats import pearsonr

def load_geotiff(path):
    ds = gdal.OpenEx(path)
    data = ds.ReadAsArray()  # shape: (bands, H, W)
    print("fin read")
    return data, ds

def create_water_mask(data):
    print("starting mask creation")
    if data.ndim == 2:
        raise ValueError("Single band image, cannot create water mask")

    r = data[0].astype(np.float32)
    g = data[1].astype(np.float32)
    b = data[2].astype(np.float32)

    blue_dominant = ((b > r) & (b > g)).astype(np.uint8)

    # Local variance using OpenCV box filter
    b_sq = cv2.boxFilter(b ** 2, -1, (15, 15))
    b_mean = cv2.boxFilter(b, -1, (15, 15))
    local_var = b_sq - b_mean ** 2

    smooth = (local_var < np.percentile(local_var, 40)).astype(np.uint8)
    mask = cv2.bitwise_and(blue_dominant, smooth)
    print("getting there")
    # Morphological cleanup with OpenCV
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove small blobs using connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_size = 500
    # Vectorized: find all labels that meet the size threshold at once
    large_labels = np.nonzero(stats[1:, cv2.CC_STAT_AREA] >= min_size)[0] + 1  # +1 to skip background
    clean_mask = np.isin(labels, large_labels).astype(np.uint8)
    return clean_mask.astype(bool)

def detect_vignetting(data, water_mask,
                      brightness_r_threshold=0.3,
                      color_std_threshold=15.0):
    """
    Returns a dict with:
      - vignetting_detected: bool
      - brightness_correlation: float  (radial brightness falloff strength)
      - color_cast_std: float          (color variance across water pixels)
      - details: dict of per-channel findings
    """
    H, W = water_mask.shape
    cy, cx = H / 2.0, W / 2.0

    # Distance from center map using OpenCV
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist_from_center = cv2.magnitude(xx - cx, yy - cy)
    max_dist = float(np.sqrt(cy**2 + cx**2))
    dist_norm = dist_from_center / max_dist

    water_pixels_dist = dist_norm[water_mask]

    results = {'details': {}}
    correlations = []

    channel_names = ['R', 'G', 'B', 'NIR']
    for i in range(min(data.shape[0], 3)):  # Only RGB for correlation
        channel = data[i].astype(np.float32)
        water_pixels_val = channel[water_mask]

        if len(water_pixels_val) > 100:
            r, p = pearsonr(water_pixels_dist, water_pixels_val)
            correlations.append(abs(r))
            results['details'][channel_names[i]] = {
                'pearson_r': round(float(r), 3),
                'p_value': round(float(p), 5)
            }

    # Color cast: std of R/B ratio across water pixels
    if data.shape[0] >= 3:
        r_ch = data[0][water_mask].astype(np.float32)
        b_ch = data[2][water_mask].astype(np.float32)
        rb_ratio = r_ch / (b_ch + 1e-9)
        color_cast_std = float(np.std(rb_ratio))
    else:
        color_cast_std = 0.0

    mean_brightness_r = float(np.mean(correlations)) if correlations else 0.0
    vignetting_detected = (mean_brightness_r > brightness_r_threshold or
                           color_cast_std > color_std_threshold)

    results.update({
        'vignetting_detected': vignetting_detected,
        'brightness_correlation': round(mean_brightness_r, 3),
        'color_cast_std': round(color_cast_std, 3),
    })
    return results


# --- Main ---
def main():
    gdal.DontUseExceptions()
    data, ds = load_geotiff(r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\HX-14365_073_044_14865-mirage.tif")
    water_mask = create_water_mask(data)

    if water_mask.sum() < 1000:
        print("Not enough water pixels to assess vignetting reliably")
    else:
        result = detect_vignetting(data, water_mask)
        print(f"Vignetting detected: {result['vignetting_detected']}")
        print(f"Brightness radial correlation: {result['brightness_correlation']}")
        print(f"Color cast std (R/B ratio): {result['color_cast_std']}")
        print(f"Per-channel details: {result['details']}")