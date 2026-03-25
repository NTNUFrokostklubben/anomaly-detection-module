from dataclasses import dataclass
from typing import Optional
import numpy as np
from osgeo.gdal import Dataset

import entity.image.Artifact as Art
from utils.string_manip import slice_image_name


@dataclass
class Image:
    """
    Metadata for a geo-referenced aerial image.

    Attributes:
        img_id: Full file name used as the primary key (e.g. 'HX-14365_073_001_14822.tif').
        prefix: Mission/sensor prefix parsed from the file name (e.g. 'HX-14365').
        line: Flight-line number parsed from the file name (e.g. 73).
        line_number: Image index within the flight line (e.g. 1).
        abs_number: Absolute image index across the whole project (e.g. 14822).
        project: Optional project name sourced from SKAVL metadata.
        artifact_data: Optional artifact detection results associated with this image.
        img_arr: Optional image array in shape (bands, H, W).
        dataset: Optional GDAL dataset for geo-referenced metadata access.
    """
    img_id: str
    prefix: Optional[str] = None
    line: Optional[int] = None
    line_number: Optional[int] = None
    abs_number: Optional[int] = None
    project: Optional[str] = None
    artifact_data: Optional[Art.ArtifactData] = None
    img_arr: Optional[np.ndarray] = None
    dataset: Optional[Dataset] = None

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
