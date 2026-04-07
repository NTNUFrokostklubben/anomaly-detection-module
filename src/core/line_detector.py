import cv2
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────────────────
IMAGE_PATH = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/CO-12825_029_027_0644.tif"
OUTPUT_PATH = "/mnt/c/Users/sigbe/Documents/Skoleaar_25_26/Semester_6/Bachelor/HX_14365_NORDMORE_GSD10/lines_nordmore_Nord_2021-CO12825/detected_artifacts.png"

# A line must span at least this fraction of the image dimension to be an artifact
# e.g. 0.85 = line must cover 85% of image width (horizontal) or height (vertical)
COVERAGE_THRESHOLD = 0.95

# How close to perfectly horizontal/vertical (in degrees) a line must be
ANGLE_TOLERANCE_DEG = 2.0

# Canny edge detection thresholds
CANNY_LOW  = 30
CANNY_HIGH = 100

# HoughLinesP parameters
HOUGH_THRESHOLD   = 100    # min votes
HOUGH_MIN_LENGTH  = 100   # min line segment length in pixels
HOUGH_MAX_GAP     = 30   # max gap to still join as one line
# ───────────────────────────────────────────────────────────────────────────────


def compute_angle_deg(x1, y1, x2, y2):
    """Return the angle of the line in degrees (0 = horizontal, 90 = vertical)."""
    dx = x2 - x1
    dy = y2 - y1
    angle = np.degrees(np.arctan2(abs(dy), abs(dx)))  # 0–90 range
    return angle


def is_horizontal(angle_deg, tolerance=ANGLE_TOLERANCE_DEG):
    """True if the line is within `tolerance` degrees of horizontal (0°)."""
    return angle_deg <= tolerance


def is_vertical(angle_deg, tolerance=ANGLE_TOLERANCE_DEG):
    """True if the line is within `tolerance` degrees of vertical (90°)."""
    return angle_deg >= (90.0 - tolerance)


def spans_image(x1, y1, x2, y2, img_w, img_h, threshold=COVERAGE_THRESHOLD):
    """
    True if the line spans the required fraction of the image.
    - Horizontal lines: check x-span vs image width
    - Vertical lines  : check y-span vs image height
    """
    angle = compute_angle_deg(x1, y1, x2, y2)

    if is_horizontal(angle):
        span = abs(x2 - x1)
        return span >= img_w * threshold

    if is_vertical(angle):
        span = abs(y2 - y1)
        return span >= img_h * threshold

    return False  # diagonal — not an artifact


def merge_collinear_segments(segments, gap_px=50):
    """
    Merge nearby parallel segments into single long lines so that a broken
    artefact (split by HoughLinesP) is treated as one line.
    Groups horizontal segments by similar y, vertical by similar x.
    """
    horizontals = []
    verticals   = []

    for (x1, y1, x2, y2) in segments:
        angle = compute_angle_deg(x1, y1, x2, y2)
        if is_horizontal(angle):
            horizontals.append((min(x1, x2), max(x1, x2), (y1 + y2) // 2))
        elif is_vertical(angle):
            verticals.append(((x1 + x2) // 2, min(y1, y2), max(y1, y2)))

    merged = []

    # Merge horizontal segments that share roughly the same y row
    horizontals.sort(key=lambda s: s[2])  # sort by y
    while horizontals:
        x_min, x_max, y = horizontals.pop(0)
        group = [(x_min, x_max, y)]
        remaining = []
        for seg in horizontals:
            if abs(seg[2] - y) <= gap_px:
                group.append(seg)
            else:
                remaining.append(seg)
        horizontals = remaining
        gx_min = min(s[0] for s in group)
        gx_max = max(s[1] for s in group)
        gy     = int(np.mean([s[2] for s in group]))
        merged.append((gx_min, gy, gx_max, gy))

    # Merge vertical segments that share roughly the same x column
    verticals.sort(key=lambda s: s[0])  # sort by x
    while verticals:
        x, y_min, y_max = verticals.pop(0)
        group = [(x, y_min, y_max)]
        remaining = []
        for seg in verticals:
            if abs(seg[0] - x) <= gap_px:
                group.append(seg)
            else:
                remaining.append(seg)
        verticals = remaining
        gx     = int(np.mean([s[0] for s in group]))
        gy_min = min(s[1] for s in group)
        gy_max = max(s[2] for s in group)
        merged.append((gx, gy_min, gx, gy_max))

    return merged


def detect_artifacts(image_path, output_path):

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img_h, img_w = image.shape[:2]
    print(f"Image loaded: {img_w} x {img_h} px")

    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges   = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH, apertureSize=3)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP,
    )

    if raw_lines is None:
        print("No lines detected at all.")
        return []

    segments = [tuple(l[0]) for l in raw_lines]
    print(f"Raw Hough segments detected: {len(segments)}")

    # ── Merge broken segments ───────────────────────────────────────────────
    merged = merge_collinear_segments(segments)
    print(f"After merging collinear segments: {len(merged)}")

    # ── Strict artifact filter ──────────────────────────────────────────────
    # Keep only lines that are:
    #   1. Nearly perfectly horizontal OR vertical
    #   2. Span the vast majority of the image width/height
    artifact_lines = []
    for (x1, y1, x2, y2) in merged:
        angle = compute_angle_deg(x1, y1, x2, y2)
        if (is_horizontal(angle) or is_vertical(angle)) and \
           spans_image(x1, y1, x2, y2, img_w, img_h):
            artifact_lines.append((x1, y1, x2, y2))

    print(f"Artifact lines after strict filtering: {len(artifact_lines)}")

    # ── Draw results ────────────────────────────────────────────────────────
    output = image.copy()
    for (x1, y1, x2, y2) in artifact_lines:
        angle = compute_angle_deg(x1, y1, x2, y2)
        color = (0, 0, 255) if is_horizontal(angle) else (255, 0, 0)
        cv2.line(output, (x1, y1), (x2, y2), color, 20)

        # Label each line with its position and orientation
        label_x = min(x1, x2) + 5
        label_y = min(y1, y2) - 8 if min(y1, y2) > 20 else min(y1, y2) + 18
        orientation = "H" if is_horizontal(angle) else "V"
        cv2.putText(output, f"{orientation} y={y1}" if is_horizontal(angle) else f"{orientation} x={x1}",
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, output)
    print(f"Result saved to: {output_path}")

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n── Detected artifact lines ──")
    for i, (x1, y1, x2, y2) in enumerate(artifact_lines):
        angle = compute_angle_deg(x1, y1, x2, y2)
        kind  = "Horizontal" if is_horizontal(angle) else "Vertical"
        span  = abs(x2 - x1) if is_horizontal(angle) else abs(y2 - y1)
        dim   = img_w if is_horizontal(angle) else img_h
        print(f"  [{i+1}] {kind:10s}  start=({x1},{y1})  end=({x2},{y2})  "
              f"span={span}px  coverage={span/dim*100:.1f}%")

    return artifact_lines


if __name__ == "__main__":
    artifacts = detect_artifacts(IMAGE_PATH, OUTPUT_PATH)
    print(f"\nTotal artifacts found: {len(artifacts)}")