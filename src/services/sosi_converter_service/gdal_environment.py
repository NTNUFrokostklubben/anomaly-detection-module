import os
import sys
from pathlib import Path

def get_bundle_root() -> Path:
    """
    Get the bundle root of the environment
    Returns:
        the bundle root of the environment
    """
    if getattr(sys, 'frozen', False):
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).resolve().parents[3]
    return base_path / "lib" / "bundle"

def setup_gdal_environment():
    """
    Sets up the gdal environment
    """

    bundle_root = get_bundle_root()
    # print(f"DEBUG: bundle_root = {bundle_root}")
    if not bundle_root.exists():
        raise FileNotFoundError(f"GDAL bundle not found at {bundle_root}")

    os.environ["PATH"] = str(bundle_root / "bin") + ";" + os.environ.get("PATH", "")
    os.environ["GDAL_DRIVER_PATH"] = str(bundle_root / "gdalplugins")
    os.environ["PROJ_LIB"] = str(bundle_root / "share" / "proj")
    os.environ["GDAL_DATA"] = str(bundle_root / "share" / "gdal")