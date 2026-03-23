from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TileState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TILE_STATE_UNSPECIFIED: _ClassVar[TileState]
    TILE_STATE_READY: _ClassVar[TileState]
    TILE_STATE_QUEUED: _ClassVar[TileState]
    TILE_STATE_GENERATING: _ClassVar[TileState]
    TILE_STATE_MISSING: _ClassVar[TileState]
    TILE_STATE_FAILED: _ClassVar[TileState]
TILE_STATE_UNSPECIFIED: TileState
TILE_STATE_READY: TileState
TILE_STATE_QUEUED: TileState
TILE_STATE_GENERATING: TileState
TILE_STATE_MISSING: TileState
TILE_STATE_FAILED: TileState

class TileCoord(_message.Message):
    __slots__ = ("level", "x", "y")
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    level: int
    x: int
    y: int
    def __init__(self, level: _Optional[int] = ..., x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...

class TileRef(_message.Message):
    __slots__ = ("coord", "state", "local_path", "is_prefetch")
    COORD_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    LOCAL_PATH_FIELD_NUMBER: _ClassVar[int]
    IS_PREFETCH_FIELD_NUMBER: _ClassVar[int]
    coord: TileCoord
    state: TileState
    local_path: str
    is_prefetch: bool
    def __init__(self, coord: _Optional[_Union[TileCoord, _Mapping]] = ..., state: _Optional[_Union[TileState, str]] = ..., local_path: _Optional[str] = ..., is_prefetch: _Optional[bool] = ...) -> None: ...

class ViewportTileManifest(_message.Message):
    __slots__ = ("source_id", "selected_level", "tiles")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    SELECTED_LEVEL_FIELD_NUMBER: _ClassVar[int]
    TILES_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    selected_level: int
    tiles: _containers.RepeatedCompositeFieldContainer[TileRef]
    def __init__(self, source_id: _Optional[str] = ..., selected_level: _Optional[int] = ..., tiles: _Optional[_Iterable[_Union[TileRef, _Mapping]]] = ...) -> None: ...

class SourceRef(_message.Message):
    __slots__ = ("source_id", "source_path")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PATH_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    source_path: str
    def __init__(self, source_id: _Optional[str] = ..., source_path: _Optional[str] = ...) -> None: ...

class RectPx(_message.Message):
    __slots__ = ("x", "y", "width", "height")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    width: int
    height: int
    def __init__(self, x: _Optional[int] = ..., y: _Optional[int] = ..., width: _Optional[int] = ..., height: _Optional[int] = ...) -> None: ...

class TilesetDescriptor(_message.Message):
    __slots__ = ("source_id", "source_width_px", "source_height_px", "tile_width_px", "tile_height_px", "min_level", "max_level")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_WIDTH_PX_FIELD_NUMBER: _ClassVar[int]
    SOURCE_HEIGHT_PX_FIELD_NUMBER: _ClassVar[int]
    TILE_WIDTH_PX_FIELD_NUMBER: _ClassVar[int]
    TILE_HEIGHT_PX_FIELD_NUMBER: _ClassVar[int]
    MIN_LEVEL_FIELD_NUMBER: _ClassVar[int]
    MAX_LEVEL_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    source_width_px: int
    source_height_px: int
    tile_width_px: int
    tile_height_px: int
    min_level: int
    max_level: int
    def __init__(self, source_id: _Optional[str] = ..., source_width_px: _Optional[int] = ..., source_height_px: _Optional[int] = ..., tile_width_px: _Optional[int] = ..., tile_height_px: _Optional[int] = ..., min_level: _Optional[int] = ..., max_level: _Optional[int] = ...) -> None: ...

class DescribeSourceRequest(_message.Message):
    __slots__ = ("source",)
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    source: SourceRef
    def __init__(self, source: _Optional[_Union[SourceRef, _Mapping]] = ...) -> None: ...

class DescribeSourceResponse(_message.Message):
    __slots__ = ("descriptor",)
    DESCRIPTOR_FIELD_NUMBER: _ClassVar[int]
    descriptor: TilesetDescriptor
    def __init__(self, descriptor: _Optional[_Union[TilesetDescriptor, _Mapping]] = ...) -> None: ...

class PlanViewportRequest(_message.Message):
    __slots__ = ("source", "viewport_source_rect_px", "screen_pixels_per_source_pixel", "prefetch_margin_tiles", "queue_missing_tiles")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    VIEWPORT_SOURCE_RECT_PX_FIELD_NUMBER: _ClassVar[int]
    SCREEN_PIXELS_PER_SOURCE_PIXEL_FIELD_NUMBER: _ClassVar[int]
    PREFETCH_MARGIN_TILES_FIELD_NUMBER: _ClassVar[int]
    QUEUE_MISSING_TILES_FIELD_NUMBER: _ClassVar[int]
    source: SourceRef
    viewport_source_rect_px: RectPx
    screen_pixels_per_source_pixel: float
    prefetch_margin_tiles: int
    queue_missing_tiles: bool
    def __init__(self, source: _Optional[_Union[SourceRef, _Mapping]] = ..., viewport_source_rect_px: _Optional[_Union[RectPx, _Mapping]] = ..., screen_pixels_per_source_pixel: _Optional[float] = ..., prefetch_margin_tiles: _Optional[int] = ..., queue_missing_tiles: _Optional[bool] = ...) -> None: ...

class PlanViewportResponse(_message.Message):
    __slots__ = ("manifest",)
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    manifest: ViewportTileManifest
    def __init__(self, manifest: _Optional[_Union[ViewportTileManifest, _Mapping]] = ...) -> None: ...
