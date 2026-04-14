"""
Manual visualisation test for glare line detection.

Uses the production line_detector.detect_glare() for all detection logic,
then draws the results onto the image for inspection.

Usage:
    python test_glare_visualise.py
    python test_glare_visualise.py --image path/to/image.tif --output path/to/out.png
    python test_glare_visualise.py --image path/to/image.tif --no-cache   # use cv2 directly
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

from core.line_detector import detect_glare

# Paths (edit these for quick runs without CLI args)
DEFAULT_IMAGE = Path(
    __file__).parent.parent.parent.parent / "HX_14365_NORDMORE_GSD10" / "lines_nordmore_Nord_2021-CO12825" / "CO-12825_029_027_0644.tif"
DEFAULT_OUTPUT = Path(__file__).parent / "glare_visualised.png"

# Draw settings
COLOR_VERTICAL = (0, 0, 255)  # red   (BGR)
COLOR_HORIZONTAL = (255, 0, 0)  # blue  (BGR)
OVERLAY_ALPHA = 0.30


def _to_vis(raw: np.ndarray) -> np.ndarray:
    """Normalise any array to uint8 BGR for drawing.

    Args:
        raw: uint8 numpy array with shape (H, W, C)

    Returns:
        the normalised array for visualisation.
    """
    vis = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) \
        if raw.dtype != np.uint8 else raw.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.ndim == 3 and vis.shape[2] == 3:
        # GDAL/cache arrays come as RGB — swap to BGR for OpenCV
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    return vis


def draw_glare_lines(vis: np.ndarray, lines: list[dict]) -> np.ndarray:
    """
    Draw detected glare lines onto a BGR visualisation image.

    Args:
        vis:   BGR uint8 image to draw on (will be modified in-place)
        lines: list of line dicts returned by detect_glare()

    Returns:
        Annotated BGR image.
    """
    h, w = vis.shape[:2]

    for ln in lines:
        color = COLOR_VERTICAL if ln['type'] == 'vertical' else COLOR_HORIZONTAL

        if ln['type'] == 'vertical':
            lo, hi, cx = ln['start_col'], ln['end_col'], ln['centre']
            overlay = vis.copy()
            cv2.rectangle(overlay, (lo, 0), (hi, h - 1), color, -1)
            cv2.addWeighted(overlay, OVERLAY_ALPHA, vis, 1 - OVERLAY_ALPHA, 0, vis)
            cv2.line(vis, (cx, 0), (cx, h - 1), color, 2)
            cv2.putText(vis,
                        f"V x={cx} ({ln['width_px']}px) s={ln['peak_score']:.2f}",
                        (max(0, cx - 60), 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        else:
            lo, hi, cy = ln['start_row'], ln['end_row'], ln['centre']
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, lo), (w - 1, hi), color, -1)
            cv2.addWeighted(overlay, OVERLAY_ALPHA, vis, 1 - OVERLAY_ALPHA, 0, vis)
            cv2.line(vis, (0, cy), (w - 1, cy), color, 2)
            cv2.putText(vis,
                        f"H y={cy} ({ln['width_px']}px) s={ln['peak_score']:.2f}",
                        (8, max(18, cy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return vis


def load_via_cache(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load image through the production ImageCache.
    img_arr is (C, H, W), raw_for_vis is (H, W, C) uint8.

    Args:
        image_path: path to image

    Returns:
          img_arr for detect_glare and raw_for_vis
    """
    from controller.image_cache_controller import load_image_array
    img_arr, _ = load_image_array(image_path)
    # Build a uint8 (H,W,C) view for visualisation
    raw = np.transpose(img_arr, (1, 2, 0))
    raw_vis = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_arr, raw_vis


def load_via_cv2(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load image directly with cv2 (no cache dependency).

    Args:
        image_path: path to image

    Returns:
          img_arr for detect_glare and raw_for_vis
    """
    raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw_bgr is None:
        raise FileNotFoundError(f"cv2 could not open: {image_path}")
    raw_u8 = cv2.normalize(raw_bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if raw_u8.ndim == 2:
        raw_u8 = cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2BGR)
    # Convert BGR→RGB then to (C,H,W) so detect_glare sees the same layout as the cache
    raw_rgb = cv2.cvtColor(raw_u8, cv2.COLOR_BGR2RGB)
    img_arr = np.transpose(raw_rgb, (2, 0, 1))
    return img_arr, raw_u8


def main() -> None:
    """
    The main function for testing the line glare through visualising it in an output file.
    """
    parser = argparse.ArgumentParser(description="Visualise glare line detection results.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Path to input TIF")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output PNG")
    parser.add_argument("--no-cache", action="store_true",
                        help="Load with cv2 instead of ImageCache (no pipeline dependency)")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Image : {args.image}")
    print(f"Output: {args.output}")
    print(f"Loader: {'cv2 (no-cache)' if args.no_cache else 'ImageCache'}")
    print(f"{'=' * 60}\n")

    if args.no_cache:
        img_arr, raw_vis = load_via_cv2(args.image)
    else:
        try:
            img_arr, raw_vis = load_via_cache(args.image)
        except ImportError:
            print("WARNING: ImageCache not importable — falling back to cv2 loader.")
            img_arr, raw_vis = load_via_cv2(args.image)

    h, w = raw_vis.shape[:2]
    print(f"Loaded: {w}×{h}\n")

    lines = detect_glare(img_arr, img_path=args.image)

    vis = _to_vis(raw_vis)
    vis = draw_glare_lines(vis, lines)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), vis, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    print(f"\nSaved → {args.output}")
    print(f"Total glare lines drawn: {len(lines)}")


if __name__ == "__main__":
    main()
