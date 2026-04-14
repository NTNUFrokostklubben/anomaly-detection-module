from enum import Enum

class AnalysisType(Enum):
    """
    Enum used to define the type of analysis to use.
    types:
    COLOR_AVERAGE, WATER_MASK, ARTIFACT
    """

    COLOR_AVERAGE = 'color_avg'
    WATER_MASK = 'water_mask'
    ARTIFACT = 'artifact'
