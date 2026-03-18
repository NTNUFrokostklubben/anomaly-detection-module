from datetime import datetime
from osgeo import gdal
import numpy as np



from core.artifact_detector import calculate_average_color_block, detect_artifact_naive, detect_artifact_consistency
from utils.io_tools import  read_tiff_fast


def make_image(bands=3, height=10, width=10, value=128):
    """Helper to create a uniform test image (Band, H, W)."""
    return np.full((bands, height, width), value, dtype=np.uint8)


def test_calculate_average_color_block_output_length():
    """Block count should equal ceil(H/increment) * ceil(W/increment), with 3 channels."""
    img = make_image(height=10, width=10)
    increment = 5
    result = calculate_average_color_block(img, increment)
    y_blocks = (10 + increment - 1) // increment
    x_blocks = (10 + increment - 1) // increment
    assert result.shape == (y_blocks * x_blocks, 3)


def test_calculate_average_color_block_uniform_image():
    """Uniform image should yield the same average for every block."""
    img = make_image(height=10, width=10, value=100)
    result = calculate_average_color_block(img, 5)
    assert np.all(result == result[0, :])


def test_detect_artifact_naive_shape_mismatch_returns_zero():
    """Mismatched image shapes should return 0."""
    img1 = make_image(height=10, width=10)
    img2 = make_image(height=20, width=10)
    result = detect_artifact_naive(img1, img2, increment=5)
    assert result == 0


def test_detect_artifact_naive_identical_images():
    """Identical images should produce zero difference for all blocks."""
    img = make_image(height=10, width=10, value=100)
    result = detect_artifact_naive(img, img.copy(), increment=5)
    assert isinstance(result, np.ndarray)
    assert np.all(result == 0)


def test_detect_artifact_naive_different_images():
    """Clearly different images should produce non-zero block differences."""
    img1 = make_image(height=10, width=10, value=50)
    img2 = make_image(height=10, width=10, value=200)
    result = detect_artifact_naive(img1, img2, increment=5)
    assert isinstance(result, np.ndarray)
    assert np.any(result > 0)


def test_detect_artifact_consistency_identical_images():
    """Identical images should score 0 for all blocks (perfectly consistent)."""
    img = make_image(height=10, width=10, value=100)
    result = detect_artifact_consistency([img, img.copy(), img.copy()], increment=5)
    assert isinstance(result, np.ndarray)
    assert np.all(result == 0.0)


def test_detect_artifact_consistency_different_images():
    """Varying images should score higher than identical images."""
    img1 = make_image(height=10, width=10, value=50)
    img2 = make_image(height=10, width=10, value=150)
    img3 = make_image(height=10, width=10, value=200)
    result = detect_artifact_consistency([img1, img2, img3], increment=5)
    assert isinstance(result, np.ndarray)
    assert np.all(result > 0.0)


def test_detect_artifact_consistency_mixed():
    """Artifact block (same across images) scores lower than varying block."""
    artifact_block = make_image(height=10, width=10, value=100)
    varying_block = make_image(height=10, width=10, value=100)
    varying_block[:, :5, :5] = 200  # top-left block differs in varying images

    images = [artifact_block.copy(), artifact_block.copy(), artifact_block.copy()]
    artifact_scores = detect_artifact_consistency(images, increment=5)

    images[1][:, :5, :5] = 200
    images[2][:, :5, :5] = 50
    mixed_scores = detect_artifact_consistency(images, increment=5)

    # The top-left block (index 0) should score higher in mixed than artifact
    assert mixed_scores[0] > artifact_scores[0]


def test_detect_artifact_naive_different_blocks_positive():
    folder = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\\"
    img_names = [
        "HX-14365_073_001_14822.tif",
        "HX-14365_073_002_14823.tif",
        "HX-14365_073_003_14824.tif",
        "HX-14365_073_004_14825.tif",
        "HX-14365_073_005_14826.tif",
        "HX-14365_073_006_14827.tif",
        "HX-14365_073_007_14828.tif",
        "HX-14365_073_008_14829.tif",
        "HX-14365_073_009_14830.tif",
        "HX-14365_073_010_14831.tif",
        "HX-14365_073_011_14832.tif",
        "HX-14365_073_012_14833.tif",
        "HX-14365_073_013_14834.tif",
        "HX-14365_073_014_14835.tif",
        "HX-14365_073_015_14836.tif",
        "HX-14365_073_016_14837.tif",
    ]
    images = [read_tiff_fast(folder + name) for name in img_names]

    result = detect_artifact_consistency(images, increment=100)
    print(np.sort(result.flatten())[:100])


def test_detect_artifact_naive_different_blocks_negative():
    folder = r"C:\Users\name\Skule\2026-vaar\IDATA2901-bachelor-thesis\testing-images\artifact-images\\"
    img_names = [
        "HX-14365_073_001_14822.tif",
        "HX-14365_073_002_14823.tif",
        "HX-14365_073_003_14824.tif",
        "HX-14365_073_004_14825.tif",
        "HX-14365_073_005_14826.tif",
        "HX-14365_073_006_14827.tif",
        "HX-14365_073_007_14828.tif",
    ]
    images = [read_tiff_fast(folder + name)[:, :, :3] for name in img_names]
    result = detect_artifact_consistency(images, increment=30)
    print(images[0].shape)
    """for name, img in zip(img_names, images):
        region = img[ 2900:3000, 100:200, :3]
        print(f"{name}: min={region.min()}, max={region.max()}, mean={region.mean():.4f}")"""


    print(np.sort(result.flatten())[:100])
