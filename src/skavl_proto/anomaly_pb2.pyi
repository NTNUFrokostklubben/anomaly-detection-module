from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StartMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    START_RESTART: _ClassVar[StartMode]
    START_CONTINUE: _ClassVar[StartMode]

class AnomalyTypes(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNDEFINED: _ClassVar[AnomalyTypes]
    COLOR_AVERAGE: _ClassVar[AnomalyTypes]
    WATER_MASK: _ClassVar[AnomalyTypes]
    ARTIFACT: _ClassVar[AnomalyTypes]
    ARTIFACT_LINE: _ClassVar[AnomalyTypes]
START_RESTART: StartMode
START_CONTINUE: StartMode
UNDEFINED: AnomalyTypes
COLOR_AVERAGE: AnomalyTypes
WATER_MASK: AnomalyTypes
ARTIFACT: AnomalyTypes
ARTIFACT_LINE: AnomalyTypes

class UtmCoordinate(_message.Message):
    __slots__ = ("easting", "northing")
    EASTING_FIELD_NUMBER: _ClassVar[int]
    NORTHING_FIELD_NUMBER: _ClassVar[int]
    easting: float
    northing: float
    def __init__(self, easting: _Optional[float] = ..., northing: _Optional[float] = ...) -> None: ...

class AnomalySet(_message.Message):
    __slots__ = ("image_name", "anomaly_confidence", "anomaly_type", "line_number", "image_number", "geotiff_coordinate", "user_classification")
    IMAGE_NAME_FIELD_NUMBER: _ClassVar[int]
    ANOMALY_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    ANOMALY_TYPE_FIELD_NUMBER: _ClassVar[int]
    LINE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    IMAGE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    GEOTIFF_COORDINATE_FIELD_NUMBER: _ClassVar[int]
    USER_CLASSIFICATION_FIELD_NUMBER: _ClassVar[int]
    image_name: str
    anomaly_confidence: float
    anomaly_type: AnomalyTypes
    line_number: int
    image_number: int
    geotiff_coordinate: UtmCoordinate
    user_classification: str
    def __init__(self, image_name: _Optional[str] = ..., anomaly_confidence: _Optional[float] = ..., anomaly_type: _Optional[_Union[AnomalyTypes, str]] = ..., line_number: _Optional[int] = ..., image_number: _Optional[int] = ..., geotiff_coordinate: _Optional[_Union[UtmCoordinate, _Mapping]] = ..., user_classification: _Optional[str] = ...) -> None: ...

class AnomalyResponse(_message.Message):
    __slots__ = ("project_metadata", "last_processed_index", "anomaly_sets")
    PROJECT_METADATA_FIELD_NUMBER: _ClassVar[int]
    LAST_PROCESSED_INDEX_FIELD_NUMBER: _ClassVar[int]
    ANOMALY_SETS_FIELD_NUMBER: _ClassVar[int]
    project_metadata: ProjectMetadata
    last_processed_index: int
    anomaly_sets: _containers.RepeatedCompositeFieldContainer[AnomalySet]
    def __init__(self, project_metadata: _Optional[_Union[ProjectMetadata, _Mapping]] = ..., last_processed_index: _Optional[int] = ..., anomaly_sets: _Optional[_Iterable[_Union[AnomalySet, _Mapping]]] = ...) -> None: ...

class ProjectMetadata(_message.Message):
    __slots__ = ("project_name", "sosi_file_path", "image_folder_path", "sosi_water_mask_path")
    PROJECT_NAME_FIELD_NUMBER: _ClassVar[int]
    SOSI_FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FOLDER_PATH_FIELD_NUMBER: _ClassVar[int]
    SOSI_WATER_MASK_PATH_FIELD_NUMBER: _ClassVar[int]
    project_name: str
    sosi_file_path: str
    image_folder_path: str
    sosi_water_mask_path: str
    def __init__(self, project_name: _Optional[str] = ..., sosi_file_path: _Optional[str] = ..., image_folder_path: _Optional[str] = ..., sosi_water_mask_path: _Optional[str] = ...) -> None: ...

class DescribeAnomalyProjectRequest(_message.Message):
    __slots__ = ("project_metadata",)
    PROJECT_METADATA_FIELD_NUMBER: _ClassVar[int]
    project_metadata: ProjectMetadata
    def __init__(self, project_metadata: _Optional[_Union[ProjectMetadata, _Mapping]] = ...) -> None: ...

class DescribeAnomalyProjectResponse(_message.Message):
    __slots__ = ("project_metadata", "images_in_folder", "last_processed_image")
    PROJECT_METADATA_FIELD_NUMBER: _ClassVar[int]
    IMAGES_IN_FOLDER_FIELD_NUMBER: _ClassVar[int]
    LAST_PROCESSED_IMAGE_FIELD_NUMBER: _ClassVar[int]
    project_metadata: ProjectMetadata
    images_in_folder: int
    last_processed_image: int
    def __init__(self, project_metadata: _Optional[_Union[ProjectMetadata, _Mapping]] = ..., images_in_folder: _Optional[int] = ..., last_processed_image: _Optional[int] = ...) -> None: ...

class DetectAnomalySetRequest(_message.Message):
    __slots__ = ("project_metadata", "start_mode")
    PROJECT_METADATA_FIELD_NUMBER: _ClassVar[int]
    START_MODE_FIELD_NUMBER: _ClassVar[int]
    project_metadata: ProjectMetadata
    start_mode: StartMode
    def __init__(self, project_metadata: _Optional[_Union[ProjectMetadata, _Mapping]] = ..., start_mode: _Optional[_Union[StartMode, str]] = ...) -> None: ...

class DetectAnomalySetResponse(_message.Message):
    __slots__ = ("anomaly_response",)
    ANOMALY_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    anomaly_response: AnomalyResponse
    def __init__(self, anomaly_response: _Optional[_Union[AnomalyResponse, _Mapping]] = ...) -> None: ...

class GetProgressRequest(_message.Message):
    __slots__ = ("project_name",)
    PROJECT_NAME_FIELD_NUMBER: _ClassVar[int]
    project_name: str
    def __init__(self, project_name: _Optional[str] = ...) -> None: ...

class GetProgressResponse(_message.Message):
    __slots__ = ("project_name", "last_processed_image", "total_images")
    PROJECT_NAME_FIELD_NUMBER: _ClassVar[int]
    LAST_PROCESSED_IMAGE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_IMAGES_FIELD_NUMBER: _ClassVar[int]
    project_name: str
    last_processed_image: int
    total_images: int
    def __init__(self, project_name: _Optional[str] = ..., last_processed_image: _Optional[int] = ..., total_images: _Optional[int] = ...) -> None: ...

class StopAnalysisRequest(_message.Message):
    __slots__ = ("project_name",)
    PROJECT_NAME_FIELD_NUMBER: _ClassVar[int]
    project_name: str
    def __init__(self, project_name: _Optional[str] = ...) -> None: ...

class StopAnalysisResponse(_message.Message):
    __slots__ = ("acknowledged",)
    ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    acknowledged: bool
    def __init__(self, acknowledged: _Optional[bool] = ...) -> None: ...
