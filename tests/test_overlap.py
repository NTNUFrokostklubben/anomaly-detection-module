import pytest
from pathlib import Path

from utils.find_overlap import get_overlap_pixel_images
from utils.load_sosi_content import get_gdf_content


gpgk_path = Path(__file__).parent.parent / "tests" / "testdata" / "test_file_short.gpkg"
gdf = get_gdf_content(gpgk_path)


def test_full_overlap():
    """
    Two images that are identical spatially -> 100% overlap
    """

    img1_num, strip1 = 1, 1
    img2_num, strip2 = 1, 1  # same image

    bounds1, bounds2 = get_overlap_pixel_images(
        gdf,
        img1_num,
        strip1,
        img2_num,
        strip2,
    )

    assert bounds1 is not None
    assert bounds2 is not None

    # Full overlap → bounds should be identical
    assert bounds1 == bounds2


def test_partial_overlap():
    """
    Two images that overlap partially
    """

    img1_num, strip1 = 5, 1
    img2_num, strip2 = 6, 1

    bounds1, bounds2 = get_overlap_pixel_images(
        gdf,
        img1_num,
        strip1,
        img2_num,
        strip2,
    )

    assert bounds1 is not None
    assert bounds2 is not None

    x1_min, x1_max, y1_min, y1_max = bounds1
    x2_min, x2_max, y2_min, y2_max = bounds2

    # Compute overlap area in pixel space
    overlap_width = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    overlap_height = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    assert overlap_width > 0
    assert overlap_height > 0


def test_no_overlap():
    """
    Two images that are far apart spatially
    """

    img1_num, strip1 = 1, 1
    img2_num, strip2 = 10, 1  # assume far away / non-existent overlap

    bounds1, bounds2 = get_overlap_pixel_images(
        gdf,
        img1_num,
        strip1,
        img2_num,
        strip2,
    )

    # No overlap → function should return None
    assert bounds1 is None
    assert bounds2 is None