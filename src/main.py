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
    parser.add_argument("-w","--water-input")
    parser.add_argument("-p","--image-path")

    args = parser.parse_args()
    sosi_input = args.sosi_input
    water_input = args.water_input
    image_path = args.image_path

    # TODO: this should be moved and recived from the user in SKAVL

    # Convert sosi to gpkg
    input_file = Path(sosi_input)
    converted_sosi = sosi_input.replace(".sos", ".gpkg")
    gpk_path = Path(__file__).parent.parent / "test_data" / converted_sosi
    convert_sosi_to_gpkg(str(input_file),gpk_path)

    # Convert water sosi to gpkg
    input_water_file = Path(water_input)
    converted_water_sosi = water_input.replace(".SOS", ".gpkg")
    water_gpk_path = Path(__file__).parent.parent / "test_data" / converted_water_sosi
    convert_sosi_to_gpkg(str(input_water_file), water_gpk_path)

    # Set image path from args
    image_folder_path = Path(image_path)

    sosi_gdf = get_gdf_content(gpk_path)
    water_gdf = get_gdf_content(water_gpk_path)
    start_anomaly_analysis(sosi_gdf, water_gdf, image_folder_path)


if __name__ == "__main__":
    main()