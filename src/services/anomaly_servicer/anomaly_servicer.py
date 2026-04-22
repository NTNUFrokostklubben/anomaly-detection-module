import threading
from pathlib import Path

import grpc

from entity import Image
from entity.anomaly.ProjectMetadata import ProjectMetadata
from skavl_proto import anomaly_pb2_grpc, anomaly_pb2
from utils import DbConnector
from utils.io_tools import count_images_in_folder, convert_sosi_get_gdf
from services.sosi_converter_service import convert_sosi_to_gpkg, convert_sosi_to_geojson
from core.pipeline_anomaly_detection import start_anomaly_analysis
from utils.io_tools import get_gdf_content


def _canonicalize_path(p: str) -> str:
    """
    Expand user and resolve symlinks.
    Automatically regardless of system gets the total string with user dirs, etc. included.

    Args:
        p (str): Path string.
    """
    return str(Path(p).expanduser().resolve())


class AnomalyServiceServicer(anomaly_pb2_grpc.AnomalyDetectorServiceServicer):
    """
    gRPC servicer for starting and describing Anomaly Projects
    """

    def __init__(self):
        self.db_connection = DbConnector()
        self._stop_event = threading.Event()

    def DescribeAnomalyProject(self, request: anomaly_pb2.DescribeAnomalyProjectRequest, context):
        """
        Describes an anomaly project in addition to the last processed image.

        Args:
            request: DescribeAnomalyProjectRequest
            context:

        Returns:
            DescribeAnomalyProjectResponse containing the data requested
        """

        project_metadata = self._resolve_project_metadata(request.project_metadata,
                                                          context)
        found_project = self.db_connection.get_project(project_metadata.project_name)

        if found_project is not None:
            return anomaly_pb2.DescribeAnomalyProjectResponse(
                project_metadata=anomaly_pb2.ProjectMetadata(
                    project_name=found_project.project_name,
                    sosi_file_path=found_project.sosi_path,
                    image_folder_path=found_project.image_folder_path
                ),
                last_processed_image=found_project.last_processed_image_index,
                images_in_folder=count_images_in_folder(project_metadata.image_folder_path)
            )

        context.abort(grpc.StatusCode.NOT_FOUND, "Project Metadata not found or could not be added")
        return None

    def DetectAnomalySet(self, request: anomaly_pb2.DetectAnomalySetRequest, context):
        """
        Temporary entrypoint over gRPC to test triggering an analysis from flutter based on a supplied
        SOSI path and Image folder path

        Args:
            request: anomaly__pb2.DetectAnomalySetRequest
            context:

        Returns:

        """

        project_metadata = self._resolve_project_metadata(request.project_metadata, context)

        image_folder_path = Path(project_metadata.image_folder_path)

        # Clears the stop event to not stop immediately if it was set.
        self._stop_event.clear()

        # Convert sosi to gpkg
        gdf = convert_sosi_get_gdf(Path(project_metadata.sosi_path))

        def on_image_complete():
            DbConnector().increment_project_image_index(project_metadata.project_name)

        detected_anomalies: list[Image] = []

        if request.start_mode == anomaly_pb2.START_RESTART:
            DbConnector().set_project_image_index(project_metadata.project_name, 0)

        if project_metadata.sosi_water_mask_path:
            # Convert Water polygon sosi to gpkg and run analysis with water mask
            water_gdf = convert_sosi_get_gdf(Path(project_metadata.sosi_water_mask_path))
            detected_anomalies = start_anomaly_analysis(gdf, image_folder_path, water_gdf=water_gdf,
                                                        on_image_complete=on_image_complete,
                                                        stop_analysis_event=self._stop_event)
        else:
            detected_anomalies = start_anomaly_analysis(gdf, image_folder_path, on_image_complete=on_image_complete,
                                                        stop_analysis_event=self._stop_event)

        print("AnomalyServiceServicer.DetectAnomalySet")

        anomaly_sets: list[anomaly_pb2.AnomalySet] = []
        # Map to AnomalySet and build up a DetectAnomalySetResponse with all anomalies from here
        for anomaly in detected_anomalies:
            anomaly_sets.append(
                anomaly_pb2.AnomalySet(
                    image_name=anomaly.img_id,
                    anomaly_confidence=anomaly.max_confidence,
                    line_number=anomaly.line,
                    image_number=anomaly.line_number,
                    geotiff_coordinate=anomaly_pb2.UtmCoordinate(
                        easting=0,
                        northing=0
                    )
                )
            )

        anomaly_response: anomaly_pb2.AnomalyResponse = anomaly_pb2.AnomalyResponse(
            project_metadata=anomaly_pb2.ProjectMetadata(
                project_name=project_metadata.project_name,
                sosi_file_path=project_metadata.sosi_path,
                image_folder_path=project_metadata.image_folder_path,
                sosi_water_mask_path=project_metadata.sosi_water_mask_path
            ),
            last_processed_index=len(anomaly_sets),
            anomaly_sets=anomaly_sets
        )

        return anomaly_pb2.DetectAnomalySetResponse(
            anomaly_response=anomaly_response
        )

        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Not fully implemented yet")

    def GetProgress(self, request: anomaly_pb2.GetProgressRequest, context):
        """
        Returns progress of analysis.
        Currently polled from client every X seconds

        Args:
            request: GetProgressRequest
            context:

        Returns:
            Project time, processed images, total images to process
        """
        fetched_project: ProjectMetadata = DbConnector().get_project(request.project_name)
        total = count_images_in_folder(fetched_project.image_folder_path)
        # print(f"GetProgress: last={fetched_project.last_processed_image_index}, total={total}")
        return anomaly_pb2.GetProgressResponse(
            project_name=fetched_project.project_name,
            last_processed_image=fetched_project.last_processed_image_index,
            total_images=total
        )

        context.abort(grpc.StatusCode.UNIMPLEMENTED, "test")

    def StopAnalysis(self, request: anomaly_pb2.StopAnalysisRequest, context):
        self._stop_event.set()
        return anomaly_pb2.StopAnalysisResponse(acknowledged=True)

    def _resolve_project_metadata(self, pb_project_metadata: anomaly_pb2.ProjectMetadata, context) -> ProjectMetadata:
        """
        Resolves project metadata. Creates a new project in the database if one is not found.

        Args:
            pb_project_metadata: anomaly_pb2.ProjectMetadata
            context:

        Returns:
            ProjectMetadata - Entity containing project metadata as declared in SQL

        """
        entity = DbConnector().get_project(pb_project_metadata.project_name)

        if entity:
            if pb_project_metadata.HasField("sosi_water_mask_path"):
                entity.sosi_water_mask_path = _canonicalize_path(pb_project_metadata.sosi_water_mask_path)
            return entity

        mask = _canonicalize_path(pb_project_metadata.sosi_water_mask_path) if pb_project_metadata.HasField(
            'sosi_water_mask_path') else None

        _new_project_metadata = ProjectMetadata(
            project_name=pb_project_metadata.project_name,
            sosi_path=pb_project_metadata.sosi_file_path,
            image_folder_path=pb_project_metadata.image_folder_path,
            last_processed_image_index=0,
            sosi_water_mask_path=mask
        )

        DbConnector().add_project(_new_project_metadata)
        return _new_project_metadata
