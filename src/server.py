from concurrent import futures
import time

import grpc

from main import cli_run
from services.anomaly_servicer.anomaly_servicer import AnomalyServiceServicer
from services.logger.logger import setup_logging
from skavl_proto import anomaly_pb2_grpc, shutdown_pb2_grpc
from services.shutdown_servicer.shutdown_servicer import ShutdownServicer
from utils import DbConnector

import argparse


def arg_checker():
    """
    Checks arguments and starts the appropriate mode.

    **server** — Run as gRPC server.

    - `-p`, `--port`: Port to start server with.

    **cli** — Start a single run that processes once based on argument paths.

    - `-i`, `--sosi-input`: Coverage polygon SOSI file. Required.
    - `-p`, `--image-path`: Path containing aerial images related to coverage SOSI. Required.
    - `-w`, `--water-input`: Water polygon file. Optional.
    """
    listener = setup_logging()
    try:
        parser = argparse.ArgumentParser(
            prog="skavl-anomaly-detection-module",
            description="Anomaly detection in aerial images")
        subparsers = parser.add_subparsers(dest="mode")

        server_parser = subparsers.add_parser("server", help="Run as grpc server")
        server_parser.add_argument("-p", "--port", help="Port to start server with")
        server_parser.add_argument("-l", "--local", action="store_true",
                                   help="""Determines if all or only local connections should be accepted. 
                                   If this argument is present, the servers IP will be 127.0.0.1, 
                                   if this argument is omitted, the ip will be set to 0.0.0.0 meaning accept all connections""")

        cli_parser = subparsers.add_parser("cli",
                                           help="Start a single run cli version that runs once based on argument paths")
        cli_parser.add_argument("-i", "--sosi-input", required=True, help="Coverage polygon sosi file")
        cli_parser.add_argument("-p", "--image-path", required=True,
                                help="Path containing aerial images related to coverage SOSI")
        cli_parser.add_argument("-w", "--water-input", help="Water polygon file")

        args = parser.parse_args()

        if args.mode == "cli":
            cli_run(args)
        else:
            serve(args)
    finally:
        listener.stop()


def serve(args):
    """
    Server entrypoint
    """
    server_port = getattr(args, "port", None) or 50052
    db = DbConnector()
    db.init()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    anomaly_pb2_grpc.add_AnomalyDetectorServiceServicer_to_server(AnomalyServiceServicer(), server)
    shutdown_pb2_grpc.add_ShutdownServiceServicer_to_server(ShutdownServicer(server), server)


    # Accepts connections only locally when running locally.
    server_ip = ""
    if getattr(args, "local", False):
        server_ip = "127.0.0.1"
    else:
        server_ip = "0.0.0.0"
    server.add_insecure_port(f"{server_ip}:{server_port}")
    server.start()
    print(f"gRPC server listening on {server_ip}:{server_port}")
    server.wait_for_termination()


if __name__ == "__main__":
    """
    Main entrypoint of the application
    """
    arg_checker()
