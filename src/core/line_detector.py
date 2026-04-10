import cv2
import numpy as np
from scipy.signal import find_peaks

IMAGE_PATH1 = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/CO-12825_029_027_0644.tif"
OUTPUT_PATH1 = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/detected_glare_anomaly.png"

IMAGE_PATH2 = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/HX-14365_073_001_14822.tif"
OUTPUT_PATH2 = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/detected_glare_control.png"

# Gaussian σ (pixels) for scene-brightness blur along one axis.
HIGHPASS_SIGMA = 80

# Number of perpendicular bands used for the sign-consistency test.
N_BANDS = 20

# A column/row is a glare candidate when BOTH hold:
CONSISTENCY_MIN = 0.90   # ≥ 90 % of bands agree in sign
MAGNITUDE_MIN   = 0.10   # median per-band |z-score| ≥ 0.10

# Peak-detection parameters
PEAK_MIN_DISTANCE   = 15    # minimum px separation between two distinct peaks
PEAK_MIN_PROMINENCE = 0.05  # peak must rise above its valley neighbours by this

# Decay-walk: extent boundary is where score drops below this fraction of peak.
EXTENT_DECAY = 0.35

# Minimum final width (px) after extent-fitting.  Drops noise spikes.
MIN_STRIPE_WIDTH = 5

# A stripe must span ≥ this fraction of the image in the perpendicular direction.
COVERAGE_MIN = 0.85

# For drawing the lines
COLOR_VERTICAL   = (0, 0, 255)   # red  (BGR) for vertical   glare lines
COLOR_HORIZONTAL = (255, 0, 0)   # blue (BGR) for horizontal glare lines
OVERLAY_ALPHA    = 0.30          # opacity of the filled extent rectangle

def _load_gray(path):
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


def _highpass(gray, sigma, axis):
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


def _score_profile(residual, axis, n_bands):
    """
    Return (consistency, magnitude, score) 1-D arrays.
    axis='col' → length w  (vertical   stripe detector)
    axis='row' → length h  (horizontal stripe detector)
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


def _valley_boundary(score, peaks, i):
    """
    For peak at index `i`, find left/right bounds using the score-minimum
    (valley) between adjacent peaks as the boundary.
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


def _decay_boundary(score, p, decay):
    """Walk outward from peak p until score < decay × peak_score."""
    threshold = score[p] * decay
    n = len(score)
    lo, hi = p, p
    while lo > 0     and score[lo - 1] >= threshold: lo -= 1
    while hi < n - 1 and score[hi + 1] >= threshold: hi += 1
    return lo, hi

def _detect_axis(gray, axis, sigma, n_bands, cons_min, mag_min,
                 peak_min_dist, peak_min_prom, extent_decay,
                 min_width, coverage_min):
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


def _to_vis(raw):
    vis = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) \
          if raw.dtype != np.uint8 else raw.copy()
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR) if vis.ndim == 2 else vis


def detect_glare(image_path, output_path,
                 highpass_sigma      = HIGHPASS_SIGMA,
                 n_bands             = N_BANDS,
                 consistency_min     = CONSISTENCY_MIN,
                 magnitude_min       = MAGNITUDE_MIN,
                 peak_min_distance   = PEAK_MIN_DISTANCE,
                 peak_min_prominence = PEAK_MIN_PROMINENCE,
                 extent_decay        = EXTENT_DECAY,
                 min_stripe_width    = MIN_STRIPE_WIDTH,
                 coverage_min        = COVERAGE_MIN):
    """
    Detect glare lines in `image_path`, write annotated PNG to `output_path`,
    and return a list of line dicts.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    raw, gray = _load_gray(image_path) #Gets the raw and greyscale image
    h, w = gray.shape
    print(f"  Size: {w}×{h}  |  σ={highpass_sigma}  bands={n_bands}")

    kw = dict(sigma=highpass_sigma, n_bands=n_bands,
              cons_min=consistency_min, mag_min=magnitude_min,
              peak_min_dist=peak_min_distance, peak_min_prom=peak_min_prominence,
              extent_decay=extent_decay, min_width=min_stripe_width,
              coverage_min=coverage_min)

    print("  Scanning vertical glare …")
    vlines = _detect_axis(gray, 'col', **kw)
    print(f"    → {len(vlines)} line(s)")

    print("  Scanning horizontal glare …")
    hlines = _detect_axis(gray, 'row', **kw)
    print(f"    → {len(hlines)} line(s)")

    all_lines = vlines + hlines

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