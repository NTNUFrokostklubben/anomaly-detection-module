from concurrent import futures
import time

import grpc
from osgeo import gdal
from services.anomaly_servicer.anomaly_servicer import AnomalyServiceServicer

from skavl_proto import anomaly_pb2
from skavl_proto import anomaly_pb2_grpc
from utils import DbConnector


def serve():
    db = DbConnector()
    db.init()
    gdal.UseExceptions()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    anomaly_pb2_grpc.add_AnomalyDetectorServiceServicer_to_server(AnomalyServiceServicer(), server)
    server.add_insecure_port("0.0.0.0:50052")
    server.start()
    print("gRPC server listening on 0.0.0.0:50052")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()