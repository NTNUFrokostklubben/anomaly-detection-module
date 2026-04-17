from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import numpy as np

if TYPE_CHECKING:
    from osgeo.gdal import Dataset

import entity.image.Artifact as Art
from entity.enums.analysis_t import AnalysisType
from entity.image.RasterMeta import RasterMeta
from utils.string_manip import slice_image_name


@dataclass
class Image:
    """
    Represents a tiff image from of The Norwegian Mapping Authority's aerial survey projects.
     Contains metadata parsed from the file name.

    Attributes:
        img_id: Full file name used as the primary key (e.g. 'HX-14365_073_001_14822.tif').
        prefix: Mission/sensor prefix parsed from the file name (e.g. 'HX-14365').
        line: Flight-line number parsed from the file name (e.g. 73).
        line_number: Image index within the flight line (e.g. 1).
        abs_number: Absolute image index across the whole project (e.g. 14822).
        project: Optional project name sourced from SKAVL metadata.
        img_arr: Optional image array in shape (bands, H, W).
        metadata: Optional dataset for geo-referenced metadata access. Can be pure dataset or picklable metadata.
    """
    img_id: str
    prefix: str
    line: int
    line_number: int
    abs_number: int
    project: Optional[str] = None
    img_arr: Optional[np.ndarray] = None
    metadata: Optional[Dataset] | Optional[RasterMeta] = None
    max_confidence: Optional[float] = None
    multi_analysis: Optional[list[tuple[AnalysisType, float]]] = None
    artifact_data: Optional[Art.ArtifactData] = None

    @classmethod
    def from_filename(cls, img_file: str) -> "Image":
        prefix, line, line_number, abs_number = slice_image_name(img_file)
        return cls(
            img_id=img_file,
            prefix=prefix,
            line=line,
            line_number=line_number,
            abs_number=abs_number,
        )

    def get_anomaly_set(self) -> tuple[str, float, int, int]:
        """
        Returns the anomaly set for this image.
        :return: A tuple containing the values for the anomaly set and its confidence.
        """
        return self.img_id, self.max_confidence, self.line, self.line_number
