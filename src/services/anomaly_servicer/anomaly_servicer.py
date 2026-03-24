
from pathlib import Path
from skavl_proto import anomaly_pb2_grpc, anomaly_pb2
from utils import DbConnector


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

        """

        project_name, sosi_file_path, image_folder_path = self._resolve_project_metadata(request, context)
        found_project = self.db_connection.get_project(project_name)

        if (found_project is not None):
            print(found_project.project_name, sosi_file_path, image_folder_path)
            return anomaly_pb2.DescribeAnomalyProjectResponse(
                project_data=anomaly_pb2.ProjectMetadata(
                    project_name=found_project.project_name,
                    sosi_file_path=found_project.sosi_path,
                    image_folder_path=found_project.image_folder_path
                ),
                last_processed_image=1,
                images_in_folder=3
            )


        if (self.db_connection.add_project(project_name, sosi_file_path, image_folder_path)):
            return anomaly_pb2.DescribeAnomalyProjectResponse(
                project_data=anomaly_pb2.ProjectMetadata(
                    project_name=project_name,
                    sosi_file_path=sosi_file_path,
                    image_folder_path=image_folder_path
                ),
                last_processed_image=0,
                images_in_folder=0
            );

        return anomaly_pb2.DescribeAnomalyProjectResponse(
            project_data=anomaly_pb2.ProjectMetadata(
                project_name="No project found",
                sosi_file_path="",
                image_folder_path=""
            ),
            last_processed_image=0,
            images_in_folder=0
        );

    def DetectAnomalySet(self, request, context):
        """

        Args:
            request:
            context:

        Returns:

        """
        print("AnomalyServiceServicer.DetectAnomalySet")

    def _resolve_project_metadata(self, project_metadata: anomaly_pb2.ProjectMetadata, context) -> tuple[str, str, str]:
        """
        Resolves project name, sosi file path, and image folder path from request

        Args:
            project_metadata:
            context:

        Returns:

        """
        project_name = project_metadata.project_data.project_name
        sosi_file_path = _canonicalize_path(project_metadata.project_data.sosi_file_path)
        image_folder_path = _canonicalize_path(project_metadata.project_data.image_folder_path)

        print(sosi_file_path, image_folder_path)
        return project_name, sosi_file_path, image_folder_path