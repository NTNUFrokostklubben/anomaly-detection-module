from dataclasses import dataclass
import numpy as np


@dataclass
class ArtifactCandidate:
    """
    A candidate anomaly block detected within an image.

    Attributes:
        coord_x: X coordinate of the upper-left corner of the candidate block.
        coord_y: Y coordinate of the upper-left corner of the candidate block.
        color_value: Average color value of the block.
        diff_value: Largest color difference found between this block and the
                    corresponding block in a different image on the same line.
        offset: Side length of the candidate block in pixels.
    """
    coord_x: int
    coord_y: int
    color_value: float
    diff_value: float
    offset: int


@dataclass
class ArtifactData:
    """
    Raw output from an artifact detection run on an image.

    Attributes:
        dtype: NumPy dtype string of the stored array (e.g. 'float32').
        shape: Shape of the stored array as a tuple.
        offset: Block size used during the detection run.
        data: Array of detection values; orders of magnitude smaller than
              the total pixel count of the source image.
    """
    data: np.ndarray
    dtype: str
    shape: tuple
    offset: int
