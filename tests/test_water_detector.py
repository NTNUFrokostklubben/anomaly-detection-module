import numpy as np
import pytest
import geopandas as gpd
from affine import Affine
from shapely.geometry import MultiPolygon, Polygon
from unittest.mock import MagicMock

from core.water_detector import _affine_from_sosi_polygon, create_water_polygon_mask

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
