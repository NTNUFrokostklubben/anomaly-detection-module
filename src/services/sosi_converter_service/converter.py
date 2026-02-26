import subprocess
import tempfile
import shutil
from pathlib import Path
from .gdal_environment import setup_gdal_environment, get_bundle_root

# Convert SOSI to GPKG using GDAL's ogr2ogr
def convert_sosi_to_gpkg(input_file: str, output_file: str):
    setup_gdal_environment()
    bundle_root = get_bundle_root()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input = Path(tmpdir) / Path(input_file).name
        shutil.copy(input_file, tmp_input)

        ogr2ogr = bundle_root / "bin" / "ogr2ogr.exe"

        command = [
            str(ogr2ogr),
            "-f", "GPKG",
            str(output_file),
            str(tmp_input),
        ]

        subprocess.run(command, check=True)