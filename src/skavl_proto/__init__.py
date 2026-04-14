import sys
import pkgutil
from importlib import import_module

for m in pkgutil.iter_modules(__path__):
    name = m.name
    if name.endswith("_pb2") or name.endswith("_pb2_grpc"):
        mod = import_module(f"{__name__}.{name}")
        setattr(sys.modules[__name__], name, mod)
        sys.modules[name] = mod