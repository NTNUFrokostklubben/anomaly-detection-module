from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ProgressReport(_message.Message):
    __slots__ = ("project_name", "progress")
    PROJECT_NAME_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    project_name: str
    progress: float
    def __init__(self, project_name: _Optional[str] = ..., progress: _Optional[float] = ...) -> None: ...

class ProgressRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
