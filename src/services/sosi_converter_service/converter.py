import chardet
import codecs
import tempfile
import shutil
import subprocess
from pathlib import Path
from .gdal_environment import setup_gdal_environment, get_bundle_root

def normalize_sosi_encoding(input_path: Path, output_path: Path):
    """Detect encoding, remove BOM, and normalize to ISO-8859-1."""

    # Read raw bytes
    raw = input_path.read_bytes()

    # Detect encoding
    detected = chardet.detect(raw)
    encoding = detected["encoding"]

    if not encoding:
        print("Could not detect encoding — assuming UTF-8")
        encoding = "utf-8"

    print(f"Detected encoding: {encoding} (confidence: {detected['confidence']})")

    # Decode safely
    try:
        text = raw.decode(encoding, errors="ignore")
    except Exception:
        print("⚠ Decode failed — forcing utf-8 fallback")
        text = raw.decode("utf-8", errors="ignore")

    # Remove BOM if present
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    # Re-encode to ISO-8859-1
    with open(output_path, "w", encoding="iso-8859-1", errors="ignore") as f:
        f.write(text)

# Convert SOSI to GPKG using GDAL's ogr2ogr
def convert_sosi_to_gpkg(input_file: str, output_file: str):
    setup_gdal_environment()
    bundle_root = get_bundle_root()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input = Path(tmpdir) / Path(input_file).name
        shutil.copy(input_file, tmp_input)
        
        normalize_sosi_encoding(Path(input_file), tmp_input)
        ogr2ogr = bundle_root / "bin" / "ogr2ogr.exe"

        command = [
            str(ogr2ogr),
            "-f", "GPKG",
            str(output_file),
            str(tmp_input),
        ]

        subprocess.run(command, check=True)
        

# Convert SOSI to GeoJSON using GDAL's ogr2ogr
def convert_sosi_to_GeoJson(input_file: str, output_file: str):
    setup_gdal_environment()
    bundle_root = get_bundle_root()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input = Path(tmpdir) / Path(input_file).name
        shutil.copy(input_file, tmp_input)

        ogr2ogr = bundle_root / "bin" / "ogr2ogr.exe"

        command = [
            str(ogr2ogr),
            "-f", "GeoJSON",
            str(output_file),
            str(tmp_input),
        ]

        subprocess.run(command, check=True)