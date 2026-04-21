from enum import Enum

class AnalysisType(Enum):
    """
    Enum used to define the type of analysis to use. New analysis types must be added here.
    types:
    COLOR_AVERAGE, WATER_MASK, ARTIFACT, ARTIFACT_LINE
    """

    COLOR_AVERAGE = 'color_avg'
    WATER_MASK = 'water_mask'
    ARTIFACT = 'artifact'
    ARTIFACT_LINE = 'artifact_line'
