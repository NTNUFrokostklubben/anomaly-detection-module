
from pathlib import Path

import grpc
from skavl_proto import anomaly_pb2_grpc, anomaly_pb2
from utils import DbConnector
from utils.io_tools import count_images_in_folder
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

    """

    def __init__(self):
        self.db_connection = DbConnector()

    def DescribeAnomalyProject(self, request, context):
        """

        Args:
            request: DescribeAnomalyProjectRequest
            context:

        Returns:
            DescribeAnomalyProjectResponse containing the data requested
        """

        project_name, sosi_file_path, image_folder_path = self._resolve_project_metadata(request.project_metadata, context)
        found_project = self.db_connection.get_project(project_name)

        if (found_project is not None):
            return anomaly_pb2.DescribeAnomalyProjectResponse(
                project_metadata=anomaly_pb2.ProjectMetadata(
                    project_name=found_project.project_name,
                    sosi_file_path=found_project.sosi_path,
                    image_folder_path=found_project.image_folder_path
                ),
                last_processed_image=found_project.last_processed_image_index,
                images_in_folder=count_images_in_folder(image_folder_path)
            )

        if (self.db_connection.add_project(project_name, sosi_file_path, image_folder_path)):
            return anomaly_pb2.DescribeAnomalyProjectResponse(
                project_metadata=anomaly_pb2.ProjectMetadata(
                    project_name=project_name,
                    sosi_file_path=sosi_file_path,
                    image_folder_path=image_folder_path
                ),
                last_processed_image=0,
                images_in_folder=count_images_in_folder(image_folder_path)
            );
        context.abort(grpc.StatusCode.NOT_FOUND, "Project Metadata not found or could not be added")

    def DetectAnomalySet(self, request, context):
        """
        Temporary entrypoint over gRPC to test triggering an analysis from flutter based on a supplied
        SOSI path and Image folder path

        Args:
            request:
            context:

        Returns:

        """
        project_name, sosi_file_path, image_folder_path = self._resolve_project_metadata(request.project_metadata, context)
        found_project = self.db_connection.get_project(project_name)


        # Convert sosi to gpkg
        input_file = Path(sosi_file_path)
        converted_sosi = sosi_file_path.replace(".sos", ".gpkg")
        convert_sosi_to_gpkg(str(input_file), Path(__file__).parent.parent / "test_data" / converted_sosi)

        # Set image path from args
        image_folder_path = Path(image_folder_path)
        gpgk_path = Path(__file__).parent.parent / "test_data" / converted_sosi

        gdf = get_gdf_content(gpgk_path)
        start_anomaly_analysis(gdf, image_folder_path)

        print("AnomalyServiceServicer.DetectAnomalySet")

        anomaly_sets = []

        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Not fully implemented yet")

    def _resolve_project_metadata(self, project_metadata: anomaly_pb2.ProjectMetadata, context) -> tuple[str, str, str]:
        """
        Resolves project name, sosi file path, and image folder path from request

        Args:
            project_metadata:
            context:

        Returns:

        """
        project_name = project_metadata.project_name
        sosi_file_path = _canonicalize_path(project_metadata.sosi_file_path)
        image_folder_path = _canonicalize_path(project_metadata.image_folder_path)

        print(sosi_file_path, image_folder_path)
        return project_name, sosi_file_path, image_folder_path
