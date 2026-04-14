from utils.io_tools import get_gdf_content
from core.pipeline_anomaly_detection import start_anomaly_analysis
from pathlib import Path
from services.sosi_converter_service import convert_sosi_to_gpkg


def cli_run(args):

    if args.sosi_input == None:
        print("Missing Sosi Input path")
        return
    if args.image_path == None:
        print("Missing image path")
        return
    sosi_input = args.sosi_input
    water_input = args.water_input
    image_path = args.image_path

    # TODO: this should be moved and received from the user in SKAVL

    # Convert sosi to gpkg
    input_file = Path(sosi_input)
    converted_sosi = sosi_input.replace(".sos", ".gpkg")
    gpk_path = Path(__file__).parent.parent / "test_data" / converted_sosi
    convert_sosi_to_gpkg(str(input_file),gpk_path)

    # Convert water sosi to gpkg
    water_gdf = None
    if water_input is not None:
        input_water_file = Path(water_input)
        converted_water_sosi = water_input.replace(".SOS", ".gpkg")
        water_gpk_path = Path(__file__).parent.parent / "test_data" / converted_water_sosi
        convert_sosi_to_gpkg(str(input_water_file), water_gpk_path)
        water_gdf = get_gdf_content(water_gpk_path)

    # Set image path from args
    image_folder_path = Path(image_path)
    sosi_gdf = get_gdf_content(gpk_path)

    start_anomaly_analysis(sosi_gdf, image_folder_path, water_gdf=water_gdf)
