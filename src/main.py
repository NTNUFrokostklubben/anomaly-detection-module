from utils.io_tools import get_gdf_content
from core.pipeline_anomaly_detection import start_anomaly_analysis
from pathlib import Path
from services.sosi_converter_service import convert_sosi_to_gpkg, convert_sosi_to_geojson
import argparse


def main():
    print("main method")

    # Args
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Anomaly detection in aerial images")
    parser.add_argument("-i","--sosi-input")
    parser.add_argument("-p","--image-path")

    args = parser.parse_args()
    sosi_input = args.sosi_input
    image_path = args.image_path

    # TODO: this should be moved and recived from the user in SKAVL

    # Convert sosi to gpkg
    input_file = Path(sosi_input)
    converted_sosi = sosi_input.replace(".sos", ".gpkg")
    convert_sosi_to_gpkg(str(input_file), Path(__file__).parent.parent / "test_data" / converted_sosi)

    # Set image path from args
    image_folder_path = Path(image_path)
    gpgk_path = Path(__file__).parent.parent / "test_data" / converted_sosi

    gdf = get_gdf_content(gpgk_path)
    start_anomaly_analysis(gdf, image_folder_path)


if __name__ == "__main__":
    main()
