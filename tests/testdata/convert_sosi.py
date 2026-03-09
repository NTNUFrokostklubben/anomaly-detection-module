from pathlib import Path
from services.sosi_converter_service.converter import convert_sosi_to_gpkg

input_file = Path(__file__).parent / "test_file.sos"
output_file = Path(__file__).parent / "test_file_short.gpkg"

print("Starting conversion...")
convert_sosi_to_gpkg(str(input_file), str(output_file))

print("Conversion finished")