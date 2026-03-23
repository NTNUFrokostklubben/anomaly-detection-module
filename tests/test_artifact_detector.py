import numpy as np
import pytest

from core.artifact_detector import calculate_average_color_block, detect_artifact_consistency
from entity.image.Image import Image
from utils.db_connector import DbConnector
from utils.io_tools import read_tiff_fast
from utils.string_manip import slice_image_name


@pytest.fixture(autouse=True)
def reset_db_singleton():
    DbConnector._instance = None
    DbConnector._conn = None
    DbConnector._db_file = ":memory:"
    yield
    if DbConnector._conn is not None:
        DbConnector._conn.close()
    DbConnector._instance = None
    DbConnector._conn = None
    DbConnector._db_file = "database.db"


def make_image(img_id="HX-14365_073_001_14822.tif", bands=3, height=10, width=10, value=128):
    """Helper to create an Image with a uniform array (Band, H, W)."""
    arr = np.full((bands, height, width), value, dtype=np.uint8)
    return Image(img_id=img_id, prefix=None, line=None, line_number=None, abs_number=None, img_arr=arr)


def test_calculate_average_color_block_output_length():
    """Block count should equal ceil(H/increment) * ceil(W/increment), with 3 channels."""
    img = make_image(height=10, width=10)
    increment = 5
    result = calculate_average_color_block(img.img_arr, increment)
    y_blocks = (10 + increment - 1) // increment
    x_blocks = (10 + increment - 1) // increment
    assert result.shape == (y_blocks * x_blocks, 3)


def test_calculate_average_color_block_uniform_image():
    """Uniform image should yield the same average for every block."""
    img = make_image(height=10, width=10, value=100)
    result = calculate_average_color_block(img.img_arr, 5)
    assert np.all(result == result[0, :])


def test_detect_artifact_consistency_identical_images():
    """Identical images should score 0 for all blocks (perfectly consistent)."""
    img = make_image(value=100)
    images = [
        make_image(img_id="HX-14365_073_001_14822.tif", value=100),
        make_image(img_id="HX-14365_073_002_14823.tif", value=100),
        make_image(img_id="HX-14365_073_003_14824.tif", value=100),
    ]
    result = detect_artifact_consistency(images, increment=5)
    assert isinstance(result, np.ndarray)
    assert np.all(result == 0.0)


def test_detect_artifact_consistency_different_images():
    """Varying images should score higher than identical images."""
    images = [
        make_image(img_id="HX-14365_073_001_14822.tif", value=50),
        make_image(img_id="HX-14365_073_002_14823.tif", value=150),
        make_image(img_id="HX-14365_073_003_14824.tif", value=200),
    ]
    result = detect_artifact_consistency(images, increment=5)
    assert isinstance(result, np.ndarray)
    assert np.all(result > 0.0)


def test_detect_artifact_consistency_mixed():
    """Artifact block (same across images) scores lower than varying block."""
    images = [
        make_image(img_id="HX-14365_073_001_14822.tif", value=100),
        make_image(img_id="HX-14365_073_002_14823.tif", value=100),
        make_image(img_id="HX-14365_073_003_14824.tif", value=100),
    ]
    artifact_scores = detect_artifact_consistency(images, increment=5)

    images[1].img_arr[:, :5, :5] = 200
    images[2].img_arr[:, :5, :5] = 50
    mixed_scores = detect_artifact_consistency(images, increment=5)

    # The top-left block should score higher in mixed than artifact
    assert mixed_scores[0] > artifact_scores[0]


def _load_images(folder, img_names):
    """Load a list of image files into Image objects."""
    images = []
    for name in img_names:
        arr = read_tiff_fast(folder + name)
        prefix, line, line_number, abs_number = slice_image_name(name)
        images.append(Image(
            img_id=name,
            prefix=prefix,
            line=line,
            line_number=line_number,
            abs_number=abs_number,
            img_arr=arr,
        ))
    return images


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
    images = _load_images(folder, img_names)
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
    images = _load_images(folder, img_names)
    result = detect_artifact_consistency(images, increment=100)
    print(images[0].img_arr.shape)
    print(np.sort(result.flatten())[:100])
