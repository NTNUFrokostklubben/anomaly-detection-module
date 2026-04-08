import cv2
import numpy as np

# config

IMAGE_PATH1 = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/CO-12825_029_027_0644.tif"
OUTPUT_PATH1 = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/detected_glare_anomaly.png"

IMAGE_PATH2 = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/HX-14365_073_001_14822.tif"
OUTPUT_PATH2 = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/detected_glare_control.png"

# Width of the Gaussian used to estimate scene-level brightness per column/row. Must be odd
HIGHPASS_SIGMA = 80

# Number of horizontal bands used to test sign-consistency.
# More bands = finer resolution but noisier per-band estimates.
N_BANDS = 20

# A stripe is accepted when:
CONSISTENCY_MIN = 0.90  # % of bands that must agree in sign
MAGNITUDE_MIN = 0.15  # median per-band z-score magnitude

# Two candidate columns are part of the same stripe when they are this close.
STRIPE_MERGE_GAP = 80

# A stripe must be at least this many pixels wide to count.
MIN_STRIPE_WIDTH = 3

# A detected stripe must span at least this fraction of the image dimension.
COVERAGE_MIN = 0.85

# colors for debugging
COLOR_VERTICAL = (0, 0, 255)
COLOR_HORIZONTAL = (255, 0, 0)

# CORE FUNCTIONS
def load_as_float_gray(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load image (8 or 16-bit, colour or grey) and return
    (original_bgr_or_gray, float32_gray_0-255).
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot open: {path}")

    if raw.ndim == 3:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw.copy()

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    return raw, gray


def highpass_residual(gray: np.ndarray, sigma: int, axis: str) -> np.ndarray:
    """
    Subtract a heavy Gaussian blur along one axis to remove scene-level
    brightness variation, leaving only stripe-like anomalies.

    axis='col' → blur horizontally  (removes left-right scene gradient)
    axis='row' → blur vertically    (removes top-bottom scene gradient)
    """
    # Kernel size: 6*sigma + 1, rounded to odd
    ksize = int(6 * sigma) | 1

    if axis == 'col':
        blur = cv2.GaussianBlur(gray, (ksize, 1), sigmaX=sigma, sigmaY=0)
    else:
        blur = cv2.GaussianBlur(gray, (1, ksize), sigmaX=0, sigmaY=sigma)

    return gray - blur


def band_signals(residual: np.ndarray, indices, axis: str,
                 n_bands: int) -> np.ndarray:
    """
    For a set of columns (axis='col') or rows (axis='row'), compute the
    mean residual value in each of n_bands slices of the perpendicular
    dimension, normalised by that band's overall std.

    Returns array of shape (n_bands,).
    """
    h, w = residual.shape

    if axis == 'col':
        band_len = h // n_bands
        signals = np.empty(n_bands, dtype=np.float32)
        for i in range(n_bands):
            band = residual[i * band_len:(i + 1) * band_len, :]
            stripe_mean = np.mean(band[:, indices])
            band_std = np.std(band) + 1e-5
            signals[i] = stripe_mean / band_std
    else:
        band_len = w // n_bands
        signals = np.empty(n_bands, dtype=np.float32)
        for i in range(n_bands):
            band = residual[:, i * band_len:(i + 1) * band_len]
            stripe_mean = np.mean(band[indices, :])
            band_std = np.std(band) + 1e-5
            signals[i] = stripe_mean / band_std

    return signals


def glare_scores_per_column(residual: np.ndarray,
                            n_bands: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (consistency, magnitude) arrays of length w.
    Vectorised: avoids per-pixel Python loops.
    """
    h, w = residual.shape
    band_h = h // n_bands

    # Shape: (n_bands, w)
    band_means = np.stack([
        np.mean(residual[i * band_h:(i + 1) * band_h, :], axis=0)
        for i in range(n_bands)
    ])  # (n_bands, w)

    band_stds = np.array([
        np.std(residual[i * band_h:(i + 1) * band_h, :]) + 1e-5
        for i in range(n_bands)
    ])  # (n_bands,)

    # Normalise each band
    z = band_means / band_stds[:, np.newaxis]  # (n_bands, w)

    dominant_sign = np.sign(np.median(z, axis=0))  # (w,)
    same_sign = (np.sign(z) == dominant_sign[np.newaxis, :]).mean(axis=0)
    magnitude = np.median(np.abs(z), axis=0)

    return same_sign.astype(np.float32), magnitude.astype(np.float32)


def glare_scores_per_row(residual: np.ndarray,
                         n_bands: int) -> tuple[np.ndarray, np.ndarray]:
    """Same as above but for horizontal glare (per-row analysis)."""
    h, w = residual.shape
    band_w = w // n_bands

    band_means = np.stack([
        np.mean(residual[:, i * band_w:(i + 1) * band_w], axis=1)
        for i in range(n_bands)
    ])  # (n_bands, h)

    band_stds = np.array([
        np.std(residual[:, i * band_w:(i + 1) * band_w]) + 1e-5
        for i in range(n_bands)
    ])  # (n_bands,)

    z = band_means / band_stds[:, np.newaxis]  # (n_bands, h)

    dominant_sign = np.sign(np.median(z, axis=0))
    same_sign = (np.sign(z) == dominant_sign[np.newaxis, :]).mean(axis=0)
    magnitude = np.median(np.abs(z), axis=0)

    return same_sign.astype(np.float32), magnitude.astype(np.float32)


def find_candidate_indices(consistency: np.ndarray,
                           magnitude: np.ndarray,
                           cons_min: float,
                           mag_min: float) -> np.ndarray:
    """Return indices where both thresholds are met."""
    return np.where((consistency >= cons_min) & (magnitude >= mag_min))[0]


def group_indices(indices: np.ndarray,
                  gap: int) -> list[tuple[int, int]]:
    """
    Merge nearby indices into (start, end) groups.
    Gap-filling: indices within `gap` pixels of each other are joined.
    """
    if len(indices) == 0:
        return []

    groups = []
    start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx - prev <= gap:
            prev = idx
        else:
            groups.append((start, prev))
            start = prev = idx

    groups.append((start, prev))
    return groups


def groups_to_lines(groups: list[tuple[int, int]],
                    img_w: int, img_h: int,
                    axis: str,
                    coverage_min: float) -> list[dict]:
    """
    Convert (start_col, end_col) or (start_row, end_row) groups into
    line dicts, discarding stripes that don't span the image.
    """
    lines = []
    for start, end in groups:
        centre = (start + end) // 2
        width = end - start + 1

        if axis == 'col':
            span = img_h
            if span < coverage_min * img_h:
                continue
            lines.append({
                'type': 'vertical',
                'x1': centre, 'y1': 0,
                'x2': centre, 'y2': img_h - 1,
                'width_px': width,
                'start_col': start, 'end_col': end,
            })
        else:
            span = img_w
            if span < coverage_min * img_w:
                continue
            lines.append({
                'type': 'horizontal',
                'x1': 0, 'y1': centre,
                'x2': img_w - 1, 'y2': centre,
                'width_px': width,
                'start_row': start, 'end_row': end,
            })

    return lines

def detect_glare(image_path: str, output_path: str,
                 highpass_sigma: int = HIGHPASS_SIGMA,
                 n_bands: int = N_BANDS,
                 consistency_min: float = CONSISTENCY_MIN,
                 magnitude_min: float = MAGNITUDE_MIN,
                 stripe_merge_gap: int = STRIPE_MERGE_GAP,
                 min_stripe_width: int = MIN_STRIPE_WIDTH,
                 coverage_min: float = COVERAGE_MIN) -> list[dict]:
    print(f"\n{'=' * 60}")
    print(f"Processing: {image_path}")

    raw, gray = load_as_float_gray(image_path)
    h, w = gray.shape
    print(f"  Size: {w}x{h}  |  high-pass σ={highpass_sigma}")

    all_lines: list[dict] = []

    # Check vertical glare
    print("  Scanning for vertical glare …")
    res_col = highpass_residual(gray, highpass_sigma, axis='col')
    cons_c, mag_c = glare_scores_per_column(res_col, n_bands)

    cands_c = find_candidate_indices(cons_c, mag_c, consistency_min, magnitude_min)
    groups_c = group_indices(cands_c, stripe_merge_gap)
    groups_c = [(s, e) for s, e in groups_c if (e - s + 1) >= min_stripe_width]
    vlines = groups_to_lines(groups_c, w, h, axis='col', coverage_min=coverage_min)
    all_lines.extend(vlines)
    print(f"    → {len(vlines)} vertical glare stripe(s)")

    # Check horizontal glare
    print("  Scanning for horizontal glare …")
    res_row = highpass_residual(gray, highpass_sigma, axis='row')
    cons_r, mag_r = glare_scores_per_row(res_row, n_bands)

    cands_r = find_candidate_indices(cons_r, mag_r, consistency_min, magnitude_min)
    groups_r = group_indices(cands_r, stripe_merge_gap)
    groups_r = [(s, e) for s, e in groups_r if (e - s + 1) >= min_stripe_width]
    hlines = groups_to_lines(groups_r, w, h, axis='row', coverage_min=coverage_min)
    all_lines.extend(hlines)
    print(f"    → {len(hlines)} horizontal glare stripe(s)")

    # Convert raw to BGR 8-bit for drawing
    if raw.ndim == 2:
        vis = cv2.cvtColor(
            cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )
    elif raw.dtype != np.uint8:
        vis = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    else:
        vis = raw.copy()

    for line in all_lines:
        color = COLOR_VERTICAL if line['type'] == 'vertical' else COLOR_HORIZONTAL
        draw_thickness = max(3, line['width_px'])

        if line['type'] == 'vertical':
            cx = (line['start_col'] + line['end_col']) // 2
            cv2.line(vis, (cx, 0), (cx, h - 1), color, draw_thickness)
            label = f"V x={cx} w={line['width_px']}px"
            cv2.putText(vis, label, (cx + 5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        else:
            cy = (line['start_row'] + line['end_row']) // 2
            cv2.line(vis, (0, cy), (w - 1, cy), color, draw_thickness)
            label = f"H y={cy} w={line['width_px']}px"
            cv2.putText(vis, label, (5, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    cv2.imwrite(output_path, vis)
    print(f"  Saved: {output_path}")

    # Print for debugging
    print(f"\n  {'─' * 50}")
    print(f"  Total glare lines detected: {len(all_lines)}")
    for i, ln in enumerate(all_lines):
        if ln['type'] == 'vertical':
            print(f"    [{i + 1}] VERTICAL   cols {ln['start_col']}–{ln['end_col']}"
                  f"  width={ln['width_px']}px")
        else:
            print(f"    [{i + 1}] HORIZONTAL rows {ln['start_row']}–{ln['end_row']}"
                  f"  width={ln['width_px']}px")

    return all_lines


if __name__ == "__main__":
    glare1 = detect_glare(IMAGE_PATH1, OUTPUT_PATH1)
    print(f"\n  → Anomaly image total: {len(glare1)} glare line(s)\n")
    print("=" * 60)
    glare2 = detect_glare(IMAGE_PATH2, OUTPUT_PATH2)
    print(f"\n  → Control image total: {len(glare2)} glare line(s)\n")