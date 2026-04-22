import anomaly_pb2 as _anomaly_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ReportGenerationRequest(_message.Message):
    __slots__ = ("project_metadata", "anomaly_sets", "confidence_threshold", "locale")
    PROJECT_METADATA_FIELD_NUMBER: _ClassVar[int]
    ANOMALY_SETS_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    LOCALE_FIELD_NUMBER: _ClassVar[int]
    project_metadata: _anomaly_pb2.ProjectMetadata
    anomaly_sets: _containers.RepeatedCompositeFieldContainer[_anomaly_pb2.AnomalySet]
    confidence_threshold: float
    locale: str
    def __init__(self, project_metadata: _Optional[_Union[_anomaly_pb2.ProjectMetadata, _Mapping]] = ..., anomaly_sets: _Optional[_Iterable[_Union[_anomaly_pb2.AnomalySet, _Mapping]]] = ..., confidence_threshold: _Optional[float] = ..., locale: _Optional[str] = ...) -> None: ...

class ReportGenerationResponse(_message.Message):
    __slots__ = ("report_url",)
    REPORT_URL_FIELD_NUMBER: _ClassVar[int]
    report_url: str
    def __init__(self, report_url: _Optional[str] = ...) -> None: ...
