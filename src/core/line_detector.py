import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

IMAGE_PATH1 = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/CO-12825_029_027_0644.tif"
OUTPUT_PATH1 = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/detected_glare_anomaly.png"

IMAGE_PATH2 = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/HX-14365_073_001_14822.tif"
OUTPUT_PATH2 = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/detected_glare_control.png"

# Gaussian σ (pixels) for scene-brightness blur along one axis.
HIGHPASS_SIGMA = 80

# Number of perpendicular bands used for the sign-consistency test.
N_BANDS = 25

# A column/row is a glare candidate when BOTH hold:
CONSISTENCY_MIN = 0.90   # ≥ 90 % of bands agree in sign
MAGNITUDE_MIN   = 0.10   # median per-band |z-score| ≥ 0.10

# Peak-detection parameters
PEAK_MIN_DISTANCE   = 15    # minimum px separation between two distinct peaks
PEAK_MIN_PROMINENCE = 0.05  # peak must rise above its valley neighbours by this

# Decay-walk: extent boundary is where score drops below this fraction of peak.
EXTENT_DECAY = 0.35

# Minimum final width (px) after extent-fitting. Drops noise spikes.
MIN_STRIPE_WIDTH = 5

# A stripe must span ≥ this fraction of the image in the perpendicular direction.
COVERAGE_MIN = 0.85

# For drawing the lines
COLOR_VERTICAL   = (0, 0, 255)   # red  (BGR) for vertical   glare lines
COLOR_HORIZONTAL = (255, 0, 0)   # blue (BGR) for horizontal glare lines
OVERLAY_ALPHA    = 0.30          # opacity of the filled extent rectangle

def _load_gray(path:str) -> tuple[NDArray[np.uint8 | np.uint16], NDArray[np.float32]]:
    """
    Loads the images based on its path, converts it to a greyscale and normalises it
    Args:
        path: the path to the image

    Returns:
        The raw file and the greyscale image
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw.copy()
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    return raw, gray


def _highpass(gray: NDArray[np.float32], sigma:int, axis:str) -> NDArray[np.float32]:
    """ Subtract a wide 1-D Gaussian along `axis` to remove scene brightness.

    Blurs the grey scale image with a gaussian filter and subtracts the blurred image from the original
    to get the high frequency components of the image

    Args:
        gray: the grey scale image to process
        sigma: Gaussian σ (pixels) for scene-brightness blur along one axis
        axis: the axis to apply the filter on, either 'col' or 'row'

    Returns:
        The grey image subtracted with the blurred image from the original
    """
    ksize = int(6 * sigma) | 1  # must be odd
    if axis == 'col':
        blur = cv2.GaussianBlur(gray, (ksize, 1), sigmaX=sigma, sigmaY=0)
    else:
        blur = cv2.GaussianBlur(gray, (1, ksize), sigmaX=0, sigmaY=sigma)
    return gray - blur


def _score_profile(residual:NDArray[np.float32], axis:str, n_bands:int) \
        -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """ Finds the consistency, magnitude, score of 1-D arrays.

    Args:
        residual: the high frequency components of the image after applying the high pass filter
        axis: the axis to apply the filter on, either 'col' or 'row'
        n_bands: the number of bands in the image

    Returns:
          The consistency, magnitude, score of 1-D arrays.
    """
    img_h, img_w = residual.shape
    if axis == 'col':
        band_len = img_h // n_bands
        band_means = np.stack([np.mean(residual[i * band_len:(i + 1) * band_len, :], axis=0)
                               for i in range(n_bands)])
        band_stds = np.array([np.std(residual[i * band_len:(i + 1) * band_len, :]) + 1e-5
                              for i in range(n_bands)])
    else:
        band_len = img_w // n_bands
        band_means = np.stack([np.mean(residual[:, i * band_len:(i + 1) * band_len], axis=1)
                               for i in range(n_bands)])
        band_stds = np.array([np.std(residual[:, i * band_len:(i + 1) * band_len]) + 1e-5
                              for i in range(n_bands)])

    z = band_means / band_stds[:, np.newaxis]
    dominant_sign = np.sign(np.median(z, axis=0))
    consistency = (np.sign(z) == dominant_sign).mean(axis=0).astype(np.float32)
    magnitude = np.median(np.abs(z), axis=0).astype(np.float32)
    return consistency, magnitude, consistency * magnitude


def _valley_boundary(score:NDArray[np.float32], peaks:list[int], i:int) -> tuple[int, int]:
    """
    For peak at index `i`, find left/right bounds using the score-minimum
    (valley) between adjacent peaks as the boundary.

    Args:
        score: The score profile of the 1-D array
        peaks: the peak locations
        i: the index of the peak

    Returns:
          The left and right bounds of the peak at index `i` based on the valley-split method.
    """
    p = peaks[i]
    n = len(score)
    if i == 0:
        lo = 0
    else:
        seg = score[peaks[i-1] : p + 1]
        lo  = peaks[i-1] + int(np.argmin(seg))
    if i == len(peaks) - 1:
        hi = n - 1
    else:
        seg = score[p : peaks[i+1] + 1]
        hi  = p + int(np.argmin(seg)) - 1
    return lo, hi


def _decay_boundary(score:NDArray[np.float32], p:int, decay: float) -> tuple[int, int]:
    """
    Walk outward from peak p until score < decay × peak_score.

    Args:
        score: the score profile of the 1-D array
        p: the peak locations
        decay: the decay of the 1-D array
    """
    threshold = score[p] * decay
    n = len(score)
    lo, hi = p, p
    while lo > 0     and score[lo - 1] >= threshold: lo -= 1
    while hi < n - 1 and score[hi + 1] >= threshold: hi += 1
    return lo, hi

#TODO refactor this to instead create a confidence score of image and add to the db
def _detect_axis(gray:NDArray[np.float32], axis:str, sigma:int, n_bands:int, cons_min:float, mag_min:float,
                 peak_min_dist:int, peak_min_prom:float, extent_decay:float,
                 min_width:int) -> list[dict]:
    """
    Detects the axis of the glare lines in the image and returns a list of line dicts with the detected lines.

    Args:
        gray: grayscale image
        axis: the axis to apply the filter on
        sigma: Gaussian σ (pixels) for scene-brightness blur along one axis
        n_bands: the number of bands in the image
        cons_min: the minimum consistency for a column/row to be considered a glare candidate
        mag_min: the minimum magnitude profile to be considered a glare candidate
        peak_min_dist: the minimum pixel separation between two distinct peaks in the score profile
        peak_min_prom: the minimum proportion of peaks in the score profile
        extent_decay: the decay of the 1-D arrays
        min_width: the minimum width of the glare lines in the image

    Returns:
        The list of line dicts with the detected lines in the image. Each dict contains the type of line
        (vertical or horizontal), the centre, start and end positions, width in pixels, peak score,
        and coordinates for drawing.
    """

    h, w = gray.shape
    residual = _highpass(gray, sigma, axis)
    cons, mag, score = _score_profile(residual, axis, n_bands)

    candidate_mask = (cons >= cons_min) & (mag >= mag_min)

    # Find all candidate peaks in the score profile
    all_peaks, _ = find_peaks(
        score,
        distance=peak_min_dist,
        prominence=peak_min_prom,
        height=mag_min * cons_min,
    )

    # Keep only peaks whose neighbourhood passes the consistency + magnitude gates
    peaks = [p for p in all_peaks
             if candidate_mask[max(0, p - 2): min(len(score), p + 3)].any()]

    lines = []
    for i, p in enumerate(peaks):
        # Extent = intersection of valley-split and decay-walk
        v_lo, v_hi = _valley_boundary(score, peaks, i)
        d_lo, d_hi = _decay_boundary(score, p, extent_decay)
        lo = max(v_lo, d_lo)
        hi = min(v_hi, d_hi)

        width = hi - lo + 1
        if width < min_width:
            continue

        # Score-weighted centre within the fitted extent
        region = np.arange(lo, hi + 1)
        weights = score[lo: hi + 1]
        centre = int(np.average(region, weights=weights)) \
            if weights.sum() > 0 else (lo + hi) // 2

        if axis == 'col':
            lines.append({'type': 'vertical',
                          'centre': centre, 'start_col': lo, 'end_col': hi,
                          'width_px': width, 'peak_score': float(score[p]),
                          'x1': centre, 'y1': 0, 'x2': centre, 'y2': h - 1})
        else:
            lines.append({'type': 'horizontal',
                          'centre': centre, 'start_row': lo, 'end_row': hi,
                          'width_px': width, 'peak_score': float(score[p]),
                          'x1': 0, 'y1': centre, 'x2': w - 1, 'y2': centre})
    return lines


def _to_vis(raw:NDArray) -> NDArray[np.uint8]:
    """
    Normalises the image and sets it to a grayscale.
    Args:
        raw: The raw image to normalise

    Returns:
        The normalised image in grayscale
    """
    vis = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) \
          if raw.dtype != np.uint8 else raw.copy()
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR) if vis.ndim == 2 else vis


def detect_glare(image_path:str, output_path:str,
                 highpass_sigma:int        = HIGHPASS_SIGMA,
                 n_bands:int               = N_BANDS,
                 consistency_min:float     = CONSISTENCY_MIN,
                 magnitude_min:float       = MAGNITUDE_MIN,
                 peak_min_distance:int     = PEAK_MIN_DISTANCE,
                 peak_min_prominence:float = PEAK_MIN_PROMINENCE,
                 extent_decay:float        = EXTENT_DECAY,
                 min_stripe_width:int      = MIN_STRIPE_WIDTH) -> list[dict]:
    """
    Detect glare lines in the image, write annotated PNG to an output png image and returns a list of line dicts.

    Args:
        image_path: Path to the image
        output_path: Path to the output image
        highpass_sigma: Highpass filter sigma
        n_bands: Number of bands
        consistency_min: Consistency threshold
        magnitude_min: Magnitude threshold
        peak_min_distance: Peak distance threshold
        peak_min_prominence: Peak prominence threshold
        extent_decay: Extent decay threshold
        min_stripe_width: Minimum stripe-width threshold

    Returns:
        The list of all the detected glare lines in the image for both axis
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    raw, gray = _load_gray(image_path) #Gets the raw and greyscale image
    h, w = gray.shape
    print(f"  Size: {w}×{h}  |  σ={highpass_sigma}  bands={n_bands}")

    kw = dict(sigma=highpass_sigma, n_bands=n_bands,
              cons_min=consistency_min, mag_min=magnitude_min,
              peak_min_dist=peak_min_distance, peak_min_prom=peak_min_prominence,
              extent_decay=extent_decay, min_width=min_stripe_width)

    print("  Scanning vertical glare …")
    v_lines = _detect_axis(gray, 'col', **kw)
    print(f"    → {len(v_lines)} line(s)")

    print("  Scanning horizontal glare …")
    h_lines = _detect_axis(gray, 'row', **kw)
    print(f"    → {len(h_lines)} line(s)")

    all_lines = v_lines + h_lines

    # ── Draw ──────────────────────────────────────────────────────────────────
    vis = _to_vis(raw)

    for ln in all_lines:
        color = COLOR_VERTICAL if ln['type'] == 'vertical' else COLOR_HORIZONTAL

        if ln['type'] == 'vertical':
            lo, hi, cx = ln['start_col'], ln['end_col'], ln['centre']
            overlay = vis.copy()
            cv2.rectangle(overlay, (lo, 0), (hi, h-1), color, -1)
            cv2.addWeighted(overlay, OVERLAY_ALPHA, vis, 1-OVERLAY_ALPHA, 0, vis)
            cv2.line(vis, (cx, 0), (cx, h-1), color, 2)
            cv2.putText(vis, f"V x={cx} ({ln['width_px']}px)",
                        (max(0, cx - 55), 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        else:
            lo, hi, cy = ln['start_row'], ln['end_row'], ln['centre']
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, lo), (w-1, hi), color, -1)
            cv2.addWeighted(overlay, OVERLAY_ALPHA, vis, 1-OVERLAY_ALPHA, 0, vis)
            cv2.line(vis, (0, cy), (w-1, cy), color, 2)
            cv2.putText(vis, f"H y={cy} ({ln['width_px']}px)",
                        (8, max(18, cy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    cv2.imwrite(output_path, vis)
    print(f"  Saved → {output_path}")

    # Report
    print(f"\n  {'─'*50}")
    print(f"  Total glare lines detected: {len(all_lines)}")
    for i, ln in enumerate(all_lines):
        if ln['type'] == 'vertical':
            print(f"    [{i+1}] VERTICAL    cols {ln['start_col']}–{ln['end_col']}"
                  f"  centre={ln['centre']}  width={ln['width_px']}px"
                  f"  score={ln['peak_score']:.3f}")
        else:
            print(f"    [{i+1}] HORIZONTAL  rows {ln['start_row']}–{ln['end_row']}"
                  f"  centre={ln['centre']}  width={ln['width_px']}px"
                  f"  score={ln['peak_score']:.3f}")

    return all_lines


if __name__ == "__main__":
    glare1 = detect_glare(IMAGE_PATH1, OUTPUT_PATH1)
    print(f"\n  → Anomaly image: {len(glare1)} glare line(s)\n")
    print("=" * 60)
    glare2 = detect_glare(IMAGE_PATH2, OUTPUT_PATH2)
    print(f"\n  → Control image: {len(glare2)} glare line(s)\n")