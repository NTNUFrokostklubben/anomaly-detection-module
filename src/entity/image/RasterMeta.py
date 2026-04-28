from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from osgeo import gdal


@dataclass
class BandMeta:
    """
    Picklable metadata for a single raster band — mirrors the per-band section of ``gdalinfo``.

    Attributes:
        index: 1-based band index within the dataset.
        data_type: GDAL data-type constant (e.g. ``gdal.GDT_Byte``).
        data_type_name: Human-readable name of the data type (e.g. ``'Byte'``, ``'UInt16'``).
        color_interp: GDAL colour-interpretation constant (e.g. ``gdal.GCI_RedBand``).
        color_interp_name: Human-readable colour interpretation (e.g. ``'Red'``).
        description: Band description string (often empty).
        no_data_value: No-data sentinel value, or ``None`` when not set.
        scale: Band scale factor, or ``None`` when not set.
        offset: Band offset value, or ``None`` when not set.
        unit_type: Physical unit string (e.g. ``'m'``), empty when not set.
        block_width: Tile/strip width in pixels (native I/O block size).
        block_height: Tile/strip height in pixels (native I/O block size).
        metadata: Band-level metadata key-value pairs from the default domain.
    """

    index: int
    data_type: int
    data_type_name: str
    color_interp: int
    color_interp_name: str
    description: str
    no_data_value: Optional[float]
    scale: Optional[float]
    offset: Optional[float]
    unit_type: str
    block_width: int
    block_height: int
    metadata: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_band(band: gdal.Band, index: int) -> "BandMeta":
        """
                Build a ``BandMeta`` from an open GDAL dataset band.

                The band is only read here; the returned object holds no reference
                to it and is safe to pickle.
                """
        from osgeo import gdal as _gdal
        block_w, block_h = band.GetBlockSize()
        no_data = band.GetNoDataValue()
        scale = band.GetScale()
        offset = band.GetOffset()
        return BandMeta(
            index=index,
            data_type=band.DataType,
            data_type_name=_gdal.GetDataTypeName(band.DataType),
            color_interp=band.GetColorInterpretation(),
            color_interp_name=_gdal.GetColorInterpretationName(band.GetColorInterpretation()),
            description=band.GetDescription() or "",
            no_data_value=float(no_data) if no_data is not None else None,
            scale=float(scale) if scale is not None else None,
            offset=float(offset) if offset is not None else None,
            unit_type=band.GetUnitType() or "",
            block_width=block_w,
            block_height=block_h,
            metadata=dict(band.GetMetadata() or {}),
        )


@dataclass
class RasterMeta:
    """
    Fully picklable snapshot of a GDAL Dataset — equivalent to the output of ``gdalinfo``.

    Safe to pass across ``ProcessPoolExecutor`` / ``multiprocessing`` boundaries because it
    contains only Python primitives; no C-extension objects are retained.

    Attributes:
        driver_short: Short driver name (e.g. ``'GTiff'``).
        driver_long: Long driver name (e.g. ``'GeoTIFF'``).
        width: Raster width in pixels.
        height: Raster height in pixels.
        band_count: Number of raster bands.
        projection: Coordinate reference system as a WKT string (empty when not set).
        geotransform: Six-element GDAL affine geotransform tuple
            ``(origin_x, pixel_w, rot_x, origin_y, rot_y, pixel_h)``.
        metadata: Dataset-level metadata from the default domain.
        image_structure: Metadata from the ``IMAGE_STRUCTURE`` domain
            (e.g. compression, interleave).
        bands: Per-band metadata, ordered by band index.
    """

    driver_short: str
    driver_long: str
    width: int
    height: int
    band_count: int
    projection: str
    geotransform: tuple[float, float, float, float, float, float]
    metadata: dict[str, str] = field(default_factory=dict)
    image_structure: dict[str, str] = field(default_factory=dict)
    bands: list[BandMeta] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived geometry helpers (computed, not stored — keeps the object lean)
    # ------------------------------------------------------------------

    @property
    def origin_x(self) -> float:
        """Top-left corner X in the dataset's CRS."""
        return self.geotransform[0]

    @property
    def origin_y(self) -> float:
        """Top-left corner Y in the dataset's CRS."""
        return self.geotransform[3]

    @property
    def pixel_width(self) -> float:
        """Pixel width in CRS units (positive east)."""
        return self.geotransform[1]

    @property
    def pixel_height(self) -> float:
        """Pixel height in CRS units (negative for north-up rasters)."""
        return self.geotransform[5]

    @property
    def rotation_x(self) -> float:
        """Row rotation (0 for north-up rasters)."""
        return self.geotransform[2]

    @property
    def rotation_y(self) -> float:
        """Column rotation (0 for north-up rasters)."""
        return self.geotransform[4]

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @staticmethod
    def from_dataset(ds: gdal.Dataset) -> "RasterMeta":
        """
        Build a ``RasterMeta`` from an open GDAL dataset.

        The dataset is only read here; the returned object holds no reference
        to it and is safe to pickle.
        """
        driver = ds.GetDriver()
        bands = [
            BandMeta.from_band(ds.GetRasterBand(i), i)
            for i in range(1, ds.RasterCount + 1)
        ]
        return RasterMeta(
            driver_short=driver.ShortName,
            driver_long=driver.LongName,
            width=ds.RasterXSize,
            height=ds.RasterYSize,
            band_count=ds.RasterCount,
            projection=ds.GetProjection() or "",
            geotransform=tuple(ds.GetGeoTransform()),
            metadata=dict(ds.GetMetadata() or {}),
            image_structure=dict(ds.GetMetadata("IMAGE_STRUCTURE") or {}),
            bands=bands,
        )

    @staticmethod
    def from_rasterio(ds) -> "RasterMeta":
        """
        Build a ``RasterMeta`` from an open ``rasterio.DatasetReader``.

        ``ds.transform`` is an ``affine.Affine`` object; ``.to_gdal()`` converts
        it to the standard GDAL 6-tuple so the rest of the codebase keeps working
        without change.
        """
        color_interps = ds.colorinterp if hasattr(ds, "colorinterp") else []
        block_shapes = ds.block_shapes or []

        bands = []
        for i in range(1, ds.count + 1):
            ci = color_interps[i - 1] if color_interps else None
            bs = block_shapes[i - 1] if block_shapes else (256, 256)
            bands.append(BandMeta(
                index=i,
                data_type=0,
                data_type_name=ds.dtypes[i - 1],
                color_interp=ci.value if ci is not None else 0,
                color_interp_name=ci.name if ci is not None else "",
                description=ds.descriptions[i - 1] or "",
                no_data_value=float(ds.nodata) if ds.nodata is not None else None,
                scale=ds.scales[i - 1] if hasattr(ds, "scales") and ds.scales else None,
                offset=ds.offsets[i - 1] if hasattr(ds, "offsets") and ds.offsets else None,
                unit_type=(ds.units[i - 1] if hasattr(ds, "units") and ds.units else "") or "",
                block_width=bs[1],
                block_height=bs[0],
                metadata=dict(ds.tags(i) or {}),
            ))

        return RasterMeta(
            driver_short=ds.driver,
            driver_long=ds.driver,
            width=ds.width,
            height=ds.height,
            band_count=ds.count,
            projection=ds.crs.wkt if ds.crs else "",
            geotransform=tuple(ds.transform.to_gdal()),
            metadata=dict(ds.tags() or {}),
            image_structure=dict(ds.tags(ns="IMAGE_STRUCTURE") or {}),
            bands=bands,
        )
