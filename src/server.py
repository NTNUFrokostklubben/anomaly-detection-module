from concurrent import futures
import time

import grpc
from osgeo import gdal

from main import cli_run
from services.anomaly_servicer.anomaly_servicer import AnomalyServiceServicer

from skavl_proto import anomaly_pb2
from skavl_proto import anomaly_pb2_grpc
from utils import DbConnector

import argparse

def arg_checker():
    """
    Checks arguments and starts the appropriate mode
    """
    parser = argparse.ArgumentParser(
        prog="skavl-anomaly-detection-module",
        description="Anomaly detection in aerial images")
    subparsers = parser.add_subparsers(dest="mode")

    server_parser = subparsers.add_parser("server", help="Run as grpc server")
    server_parser.add_argument("-p","--port", help="Port to start server with")

    cli_parser = subparsers.add_parser("cli", help="Start a single run cli version that runs once based on argument paths")
    cli_parser.add_argument("-i","--sosi-input", required=True, help="Coverage polygon sosi file")
    cli_parser.add_argument("-p","--image-path", required=True, help="Path containing aerial images related to coverage SOSI")
    cli_parser.add_argument("-w","--water-input", help="Water polygon file")

    args = parser.parse_args()

    if args.mode == "cli":
        cli_run(args)
    else:
        serve(args)

def serve(args):
    """
    Server entrypoint
    """
    server_port = getattr(args, "port", None) or 50052
    db = DbConnector()
    db.init()
    gdal.UseExceptions()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    anomaly_pb2_grpc.add_AnomalyDetectorServiceServicer_to_server(AnomalyServiceServicer(), server)
    server.add_insecure_port(f"0.0.0.0:{server_port}")
    server.start()
    print(f"gRPC server listening on 0.0.0.0:{server_port}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    arg_checker()