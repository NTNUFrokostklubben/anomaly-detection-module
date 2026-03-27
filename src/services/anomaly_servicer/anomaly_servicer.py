from pathlib import Path

import grpc

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

    def DescribeAnomalyProject(self, request, context):
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

    def DetectAnomalySet(self, request, context):
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

        # Convert sosi to gpkg
        gdf = convert_sosi_get_gdf(Path(project_metadata.sosi_path))

        if project_metadata.sosi_water_mask_path:
            # Convert Water polygon sosi to gpkg and run analysis with water mask
            water_gdf = convert_sosi_get_gdf(Path(project_metadata.sosi_water_mask_path))
            start_anomaly_analysis(gdf, image_folder_path, water_gdf=water_gdf)
        else:
            start_anomaly_analysis(gdf, image_folder_path)

        print("AnomalyServiceServicer.DetectAnomalySet")

        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Not fully implemented yet")

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
