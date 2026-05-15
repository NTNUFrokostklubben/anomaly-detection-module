import threading

from skavl_proto import shutdown_pb2_grpc, shutdown_pb2

class ShutdownServicer(shutdown_pb2_grpc.ShutdownServiceServicer):
    """
    Servicer to handle graceful shutdown from remote endpoints
    """
    def __init__(self, server):
        self._server = server

    def Shutdown(self, request, context):
        """
        Shuts down server
        Args:
            request:
            context:

        Returns:

        """
        print("shutdown received", flush=True)
        threading.Timer(0.2, self._server.stop, args=[2]).start()
        return shutdown_pb2.ShutdownResponse()