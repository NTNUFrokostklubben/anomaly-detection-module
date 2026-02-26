import os
import sys
from pathlib import Path


def get_bundle_root() -> Path:
    # Project root
    project_root = Path(__file__).resolve().parents[3]

    return project_root / "lib" / "gdal_bundle"


def setup_gdal_environment():
    bundle_root = get_bundle_root()

    os.environ["PATH"] = str(bundle_root / "bin") + ";" + os.environ.get("PATH", "")
    os.environ["GDAL_DRIVER_PATH"] = str(bundle_root / "gdalplugins")
    os.environ["PROJ_LIB"] = str(bundle_root / "share" / "proj")
    os.environ["GDAL_DATA"] = str(bundle_root / "share" / "gdal")