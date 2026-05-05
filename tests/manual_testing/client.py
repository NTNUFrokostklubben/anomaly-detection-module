"""
Simple gRPC client that spoofs the frontend for testing.
Usage:
    python src/client.py detect --sosi /path/to/file.sos --images /path/to/images [--water /path/to/water.SOS]
    python src/client.py describe --project my_project
"""
import argparse
import sys
import time
import grpc

from skavl_proto import anomaly_pb2, anomaly_pb2_grpc


DEFAULT_HOST = "localhost"
DEFAULT_PORT = 50052
DEFAULT_PROJECT = "test_project"

# Hardcoded test paths



def get_stub(host: str, port: int) -> anomaly_pb2_grpc.AnomalyDetectorServiceStub:
    """Create an insecure gRPC channel and return a stub for the AnomalyDetectorService."""
    channel = grpc.insecure_channel(f"{host}:{port}")
    return anomaly_pb2_grpc.AnomalyDetectorServiceStub(channel)


def detect(args):
    """
    Send a DetectAnomalySet RPC to the server and print the results.

    Builds a DetectAnomalySetRequest from the parsed CLI arguments, sends it with a
    1-hour timeout, and prints the top 10 anomalies sorted by confidence score.

    Args:
        args: Parsed argparse namespace. Expected fields: host, port, project, sosi,
              images, and optionally water.
    """
    stub = get_stub(args.host, args.port)

    metadata = anomaly_pb2.ProjectMetadata(
        project_name=args.project,
        sosi_file_path=args.sosi,
        image_folder_path=args.images,
    )
    if args.water:
        metadata.sosi_water_mask_path = args.water

    request = anomaly_pb2.DetectAnomalySetRequest(
        project_metadata=metadata,
        start_mode=anomaly_pb2.StartMode.START_RESTART,
    )

    print(f"Sending DetectAnomalySet to {args.host}:{args.port} ...")
    t0 = time.perf_counter()
    try:
        response = stub.DetectAnomalySet(request, timeout=3600)
    except grpc.RpcError as e:
        print(f"RPC failed: {e.code()} — {e.details()}")
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    anomaly_response = response.anomaly_response
    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Project:          {anomaly_response.project_metadata.project_name}")
    print(f"Anomalies found:  {len(anomaly_response.anomaly_sets)}")
    print(f"Last index:       {anomaly_response.last_processed_index}")

    if anomaly_response.anomaly_sets:
        print("\nTop anomalies (by confidence):")
        sorted_anomalies = sorted(
            anomaly_response.anomaly_sets,
            key=lambda a: a.anomaly_confidence,
            reverse=True,
        )
        for a in sorted_anomalies[:10]:
            print(f"  {a.image_name}  confidence={a.anomaly_confidence:.4f}  line={a.line_number}  img={a.image_number}")


def describe(args):
    """
    Send a DescribeAnomalyProject RPC to the server and print the project summary.

    Args:
        args: Parsed argparse namespace. Expected fields: host, port, project.
    """
    stub = get_stub(args.host, args.port)

    metadata = anomaly_pb2.ProjectMetadata(project_name=args.project)
    request = anomaly_pb2.DescribeAnomalyProjectRequest(project_metadata=metadata)

    print(f"Sending DescribeAnomalyProject to {args.host}:{args.port} ...")
    try:
        response = stub.DescribeAnomalyProject(request, timeout=10)
    except grpc.RpcError as e:
        print(f"RPC failed: {e.code()} — {e.details()}")
        sys.exit(1)

    print(f"Project:              {response.project_metadata.project_name}")
    print(f"SOSI path:            {response.project_metadata.sosi_file_path}")
    print(f"Image folder:         {response.project_metadata.image_folder_path}")
    print(f"Images in folder:     {response.images_in_folder}")
    print(f"Last processed image: {response.last_processed_image}")


def main():
    """
    Entry point for the test client. Parses CLI arguments and dispatches to detect() or describe().

    Subcommands:
        detect   -- run anomaly detection on a set of images via the gRPC server.
        describe -- retrieve and print metadata for an existing project.
    """
    parser = argparse.ArgumentParser(description="Anomaly detection gRPC test client")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)

    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Run anomaly detection")
    detect_parser.add_argument("--sosi", help="Path to coverage SOSI file")
    detect_parser.add_argument("--images", help="Path to image folder")
    detect_parser.add_argument("--water",help="Path to water polygon SOSI file")
    detect_parser.add_argument("--project", default=DEFAULT_PROJECT, help="Project name")

    describe_parser = subparsers.add_parser("describe", help="Describe an existing project")
    describe_parser.add_argument("--project", required=True, help="Project name")

    args = parser.parse_args()

    if args.command == "detect":
        detect(args)
    elif args.command == "describe":
        describe(args)


if __name__ == "__main__":
    main()
