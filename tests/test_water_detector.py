import numpy as np
import pytest
import geopandas as gpd
from affine import Affine
from shapely.geometry import MultiPolygon, Polygon
from unittest.mock import MagicMock

from core.water_detector import _affine_from_sosi_polygon, create_water_polygon_mask, dissimilarity_confidence

# Image dimensions used across all tests
_W = 100
_H = 100

# Geo corners for a simple axis-aligned footprint.
# SOSI corner order assumed by _affine_from_sosi_polygon: [BR, TR, TL, BL]
_TL = (0.0, 100.0)    # → pixel (0,   0)
_TR = (100.0, 100.0)  # → pixel (W,   0)
_BR = (100.0, 0.0)
_BL = (0.0, 0.0)      # → pixel (0,   H)

_SOSI_POLYGON    = Polygon([_BR, _TR, _TL, _BL])
_SOSI_MULTI      = MultiPolygon([_SOSI_POLYGON])

# Water contour inside the footprint (geo space)
_WATER_CONTOUR = Polygon([(30, 40), (70, 40), (70, 60), (30, 60)])


def _make_mock_ds(has_crs: bool) -> MagicMock:
    ds = MagicMock()
    ds.RasterXSize = _W
    ds.RasterYSize = _H
    ds.GetProjection.return_value = "WGS84" if has_crs else ""
    return ds


def _make_sosi_df(img_name: str, geom=_SOSI_MULTI) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"bildefilRGB": [img_name], "geometry": [geom]},
        crs="EPSG:25833",
    )


def _make_contour_df() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": [_WATER_CONTOUR]}, crs="EPSG:25833")


# ---------------------------------------------------------------------------
# _affine_from_sosi_polygon
# ---------------------------------------------------------------------------

class TestAffineFromSosiPolygon:
    def test_pixel_origin_maps_to_tl(self):
        affine = _affine_from_sosi_polygon(_SOSI_POLYGON, _W, _H)
        x, y = affine * (0, 0)
        assert x == pytest.approx(_TL[0])
        assert y == pytest.approx(_TL[1])

    def test_top_right_pixel_maps_to_tr(self):
        affine = _affine_from_sosi_polygon(_SOSI_POLYGON, _W, _H)
        x, y = affine * (_W, 0)
        assert x == pytest.approx(_TR[0])
        assert y == pytest.approx(_TR[1])

    def test_bottom_left_pixel_maps_to_bl(self):
        affine = _affine_from_sosi_polygon(_SOSI_POLYGON, _W, _H)
        x, y = affine * (0, _H)
        assert x == pytest.approx(_BL[0])
        assert y == pytest.approx(_BL[1])

    def test_accepts_multipolygon(self):
        affine = _affine_from_sosi_polygon(_SOSI_MULTI, _W, _H)
        x, y = affine * (0, 0)
        assert x == pytest.approx(_TL[0])
        assert y == pytest.approx(_TL[1])

    def test_non_quad_raises(self):
        triangle = Polygon([(0, 0), (1, 0), (0.5, 1)])
        with pytest.raises(ValueError):
            _affine_from_sosi_polygon(triangle, _W, _H)

    def test_returns_affine_instance(self):
        result = _affine_from_sosi_polygon(_SOSI_POLYGON, _W, _H)
        assert isinstance(result, Affine)


# ---------------------------------------------------------------------------
# create_water_polygon_mask — no-CRS path
# ---------------------------------------------------------------------------

class TestCreateWaterPolygonMaskNoGeoref:
    _IMG = "test_image.tif"

    def test_returns_boolean_mask_with_correct_shape(self):
        ds = _make_mock_ds(has_crs=False)
        result = create_water_polygon_mask(
            _make_contour_df(), _make_sosi_df(self._IMG), self._IMG, ds
        )
        assert result is not None
        assert result.shape == (_H, _W)
        assert result.dtype == bool

    def test_geotransform_not_called_without_crs(self):
        ds = _make_mock_ds(has_crs=False)
        create_water_polygon_mask(
            _make_contour_df(), _make_sosi_df(self._IMG), self._IMG, ds
        )
        ds.GetGeoTransform.assert_not_called()

    def test_water_pixels_detected_in_mask(self):
        ds = _make_mock_ds(has_crs=False)
        result = create_water_polygon_mask(
            _make_contour_df(), _make_sosi_df(self._IMG), self._IMG, ds
        )
        assert result.any(), "Expected at least some water pixels in the mask"


# ---------------------------------------------------------------------------
# dissimilarity_confidence
# ---------------------------------------------------------------------------

class TestDissimilarityConfidence:
    def test_zero_input_returns_zero(self):
        assert dissimilarity_confidence(0.0) == pytest.approx(0.0)

    def test_threshold_returns_one(self):
        assert dissimilarity_confidence(0.3) == pytest.approx(1.0)

    def test_above_threshold_returns_one(self):
        assert dissimilarity_confidence(0.5) == pytest.approx(1.0)
        assert dissimilarity_confidence(1.0) == pytest.approx(1.0)

    def test_midpoint_is_between_zero_and_one(self):
        result = dissimilarity_confidence(0.15)
        assert 0.0 < result < 1.0

    def test_monotonically_increasing(self):
        xs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        values = [dissimilarity_confidence(x) for x in xs]
        assert values == sorted(values)

    def test_output_bounded(self):
        for x in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
            result = dissimilarity_confidence(x)
            assert 0.0 <= result <= 1.0

    def test_higher_k_steeper_at_midpoint(self):
        """Higher k should give a lower value at x=0.15 (slower start, sharper end)."""
        low_k = dissimilarity_confidence(0.15, k=4.0)
        high_k = dissimilarity_confidence(0.15, k=10.0)
        assert high_k < low_k
