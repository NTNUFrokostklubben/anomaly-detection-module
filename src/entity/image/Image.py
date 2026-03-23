from dataclasses import dataclass
from typing import Optional
import numpy as np
import Artifact as Art


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
    """
    img_id: str
    prefix: Optional[str]
    line: Optional[int]
    line_number: Optional[int]
    abs_number: Optional[int]
    project: Optional[str] = None
    artifact_data: Optional[Art.ArtifactData] = None
    img_arr: Optional[np.ndarray[tuple[int, int, int]]] = None
