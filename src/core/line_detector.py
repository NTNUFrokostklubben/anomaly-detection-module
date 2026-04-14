import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from pathlib import Path
from utils.timer import Timer

# Gaussian σ (pixels) for scene-brightness blur along one axis.
HIGHPASS_SIGMA = 80

# Number of perpendicular bands used for the sign-consistency test.
N_BANDS = 25

# A column/row is a glare candidate when BOTH hold:
CONSISTENCY_MIN = 0.90  # ≥% of bands agree in sign
MAGNITUDE_MIN = 0.10  # median per-band |z-score|

# Peak-detection parameters
PEAK_MIN_DISTANCE = 15  # minimum px separation between two distinct peaks
PEAK_MIN_PROMINENCE = 0.05  # peak must rise above its valley neighbours by this

# Decay-walk: extent boundary is where score drops below this fraction of peak.
EXTENT_DECAY = 0.35

# Minimum final width (px) after extent-fitting. Drops noise spikes.
MIN_STRIPE_WIDTH = 5

# A stripe must span ≥ this fraction of the image in the perpendicular direction.
COVERAGE_MIN = 0.85


def _load_gray_from_array(arr: NDArray, timer: Timer) -> NDArray[np.float32]:
    """
    Convert a cached (C, H, W) array to a normalised float32 greyscale.

    Args:
        arr:   Array from ImageCache, shape (C, H, W) or (H, W)
        timer: shared Timer instance

    Returns:
        gray: (H, W) float32 array normalised to [0, 255]
    """
    with timer.measure("load_gray: array conversion"):
        if arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))

        raw_u8 = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gray_u8 = cv2.cvtColor(raw_u8, cv2.COLOR_RGB2GRAY) if raw_u8.ndim == 3 else raw_u8

    with timer.measure("load_gray: normalize to float32"):
        return cv2.normalize(gray_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)


def _highpass(gray: NDArray[np.float32], sigma: int, axis: str, timer: Timer, ) -> NDArray[np.float32]:
    """
    Creates a highpass downsampled blur on the images to
    Args:
        gray: the normalised float32 greyscale image
        sigma: the gaussian blur sigma
        axis: the axis to blur
        timer: timer for optimisation debugging

    Returns:
        the array of the greyscale and blured image
    """
    with timer.measure(f"highpass: downsampled blur ({axis})"):
        SCALE = 4
        h, w = gray.shape
        sigma_s = sigma / SCALE
        ksize = int(6 * sigma_s) | 1

        small = cv2.resize(gray, (w // SCALE, h // SCALE),
                           interpolation=cv2.INTER_AREA)
        if axis == 'col':
            blur_s = cv2.GaussianBlur(small, (ksize, 1), sigmaX=sigma_s, sigmaY=0)
        else:
            blur_s = cv2.GaussianBlur(small, (1, ksize), sigmaX=0, sigmaY=sigma_s)
        del small

        blur = cv2.resize(blur_s, (w, h), interpolation=cv2.INTER_LINEAR)
        del blur_s

        return gray - blur


def _score_profile(
        residual: NDArray[np.float32],
        axis: str,
        n_bands: int,
        timer: Timer,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Compute per-column (or per-row) consistency, magnitude, and score.

    Args:
        residual: high-frequency residual image
        axis:     'col' or 'row'
        n_bands:  number of perpendicular bands for the sign-consistency test
        timer:    shared Timer instance

    Returns:
        Tuple of (consistency, magnitude, score) float32 arrays, one value per column/row.
    """
    img_h, img_w = residual.shape

    with timer.measure("score_profile: band means + stds"):
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

    with timer.measure("score_profile: z-score + consistency + magnitude"):
        z = band_means / band_stds[:, np.newaxis]
        dominant_sign = np.sign(np.median(z, axis=0))
        consistency = (np.sign(z) == dominant_sign).mean(axis=0).astype(np.float32)
        magnitude = np.median(np.abs(z), axis=0).astype(np.float32)

    return consistency, magnitude, consistency * magnitude


def _valley_boundary(
        score: NDArray[np.float32],
        peaks: list[int],
        i: int,
) -> tuple[int, int]:
    """
    Find left/right bounds for peak in using the score-minimum (valley) between
    adjacent peaks as the boundary. Not timed individually — called in a loop
    inside _detect_axis which is already timed.

    Args:
        score: score profile array
        peaks: list of peak indices
        i:     index into peaks for the peak of interest

    Returns:
        (lo, hi) pixel indices of the valley boundary.
    """
    p = peaks[i]
    n = len(score)
    if i == 0:
        lo = 0
    else:
        seg = score[peaks[i - 1]: p + 1]
        lo = peaks[i - 1] + int(np.argmin(seg))
    if i == len(peaks) - 1:
        hi = n - 1
    else:
        seg = score[p: peaks[i + 1] + 1]
        hi = p + int(np.argmin(seg)) - 1
    return lo, hi


def _decay_boundary(
        score: NDArray[np.float32],
        p: int,
        decay: float,
) -> tuple[int, int]:
    """
    Walk outward from peak p until score < decay × peak_score.
    Not timed individually — called in a loop inside _detect_axis.

    Args:
        score: score profile array
        p:     index of the peak
        decay: fraction of the peak score used as the walk threshold

    Returns:
        (lo, hi) pixel indices of the decay boundary.
    """
    threshold = score[p] * decay
    n = len(score)
    lo, hi = p, p
    while lo > 0 and score[lo - 1] >= threshold: lo -= 1
    while hi < n - 1 and score[hi + 1] >= threshold: hi += 1
    return lo, hi


def _detect_axis(
        gray: NDArray[np.float32],
        axis: str,
        sigma: int,
        n_bands: int,
        cons_min: float,
        mag_min: float,
        peak_min_dist: int,
        peak_min_prom: float,
        extent_decay: float,
        min_width: int,
        timer: Timer,
) -> list[dict]:
    """
    Detect glare lines along one axis and return a list of line descriptors.

    Args:
        gray:          normalised greyscale image
        axis:          'col' (vertical lines) or 'row' (horizontal lines)
        sigma:         Gaussian σ for the high-pass filter
        n_bands:       number of bands for the sign-consistency test
        cons_min:      minimum consistency threshold
        mag_min:       minimum magnitude threshold
        peak_min_dist: minimum pixel separation between peaks
        peak_min_prom: minimum peak prominence
        extent_decay:  decay-walk fraction of peak score
        min_width:     minimum stripe width in pixels
        timer:         shared Timer instance

    Returns:
        List of dicts, one per detected glare line, containing type, centre,
        extent, width, peak score, and draw coordinates.
    """
    h, w = gray.shape

    residual = _highpass(gray, sigma, axis, timer)
    cons, mag, score = _score_profile(residual, axis, n_bands, timer)

    with timer.measure(f"detect_axis ({axis}): candidate mask"):
        candidate_mask = (cons >= cons_min) & (mag >= mag_min)

    with timer.measure(f"detect_axis ({axis}): find_peaks"):
        all_peaks, _ = find_peaks(
            score,
            distance=peak_min_dist,
            prominence=peak_min_prom,
            height=mag_min * cons_min,
        )

    with timer.measure(f"detect_axis ({axis}): filter peaks"):
        peaks: list[int] = [p for p in all_peaks
                            if candidate_mask[max(0, p - 2): min(len(score), p + 3)].any()]

    with timer.measure(f"detect_axis ({axis}): extent fitting + line building"):
        lines: list[dict] = []
        for i, p in enumerate(peaks):
            v_lo, v_hi = _valley_boundary(score, peaks, i)
            d_lo, d_hi = _decay_boundary(score, p, extent_decay)
            lo = max(v_lo, d_lo)
            hi = min(v_hi, d_hi)

            width = hi - lo + 1
            if width < min_width:
                continue

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


def detect_glare(
        img_arr: NDArray,
        img_path: Path,
        highpass_sigma: int = HIGHPASS_SIGMA,
        n_bands: int = N_BANDS,
        consistency_min: float = CONSISTENCY_MIN,
        magnitude_min: float = MAGNITUDE_MIN,
        peak_min_distance: int = PEAK_MIN_DISTANCE,
        peak_min_prominence: float = PEAK_MIN_PROMINENCE,
        extent_decay: float = EXTENT_DECAY,
        min_stripe_width: int = MIN_STRIPE_WIDTH,
) -> list[dict]:
    """
    Detect glare lines in an image, write an annotated PNG, and return line descriptors.

    Args:
        img_arr:             the image array to analyse
        highpass_sigma:      Gaussian σ for scene-brightness removal
        n_bands:             number of bands for the sign-consistency test
        consistency_min:     minimum fraction of bands agreeing in sign
        magnitude_min:       minimum median per-band |z-score|
        peak_min_distance:   minimum pixel separation between peaks
        peak_min_prominence: minimum peak prominence above valley neighbours
        extent_decay:        decay-walk fraction of peak score
        min_stripe_width:    minimum stripe width in pixels

    Returns:
        List of dicts describing all detected glare lines for both axes.
    """
    timer = Timer()

    with timer.measure("Load: image gray from array"):
        gray = _load_gray_from_array(img_arr, timer)

    kw = dict(sigma=highpass_sigma, n_bands=n_bands,
              cons_min=consistency_min, mag_min=magnitude_min,
              peak_min_dist=peak_min_distance, peak_min_prom=peak_min_prominence,
              extent_decay=extent_decay, min_width=min_stripe_width, timer=timer)

    print("Analysing image: ", img_path)

    print("  Scanning vertical glare …")
    v_lines = _detect_axis(gray, 'col', **kw)
    print(f"    → {len(v_lines)} line(s)")

    print("  Scanning horizontal glare …")
    h_lines = _detect_axis(gray, 'row', **kw)
    print(f"    → {len(h_lines)} line(s)")

    all_lines = v_lines + h_lines

    # Report
    print(f"\n  {'─' * 50}")
    print(f"  Total glare lines detected: {len(all_lines)}")
    for i, ln in enumerate(all_lines):
        if ln['type'] == 'vertical':
            print(f"    [{i + 1}] VERTICAL    cols {ln['start_col']}–{ln['end_col']}"
                  f"  centre={ln['centre']}  width={ln['width_px']}px"
                  f"  score={ln['peak_score']:.3f}")
        else:
            print(f"    [{i + 1}] HORIZONTAL  rows {ln['start_row']}–{ln['end_row']}"
                  f"  centre={ln['centre']}  width={ln['width_px']}px"
                  f"  score={ln['peak_score']:.3f}")

    # timer.report(title="Timing report")
    return all_lines
