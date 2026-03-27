from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectMetadata():
    """
    Adapter Entity for ProjectMetadata derived from Proto.
    The Proto acts as the DTO whilst this is the entity we should populate and use internally
    """
    project_name: str
    sosi_path: str
    image_folder_path: str
    last_processed_image_index: int
    sosi_water_mask_path: Optional[str] = None

    @classmethod
    def from_row(cls, row: tuple):
        return cls(*row)
