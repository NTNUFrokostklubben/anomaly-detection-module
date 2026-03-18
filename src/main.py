from utils.io_tools import get_gdf_content
from core.pipeline_anomaly_detection import start_anomaly_analysis
from pathlib import Path
from services.sosi_converter_service import convert_sosi_to_gpkg, convert_sosi_to_geojson

def main():
    print("main method")

    # input_file = Path(__file__).parent.parent / "test_data" / "HX-14365_Vertikalbilde.sos"
    # print(input_file)
    # convert_sosi_to_gpkg(str(input_file), Path(__file__).parent.parent / "test_data" / "HX-14365_Vertikalbilde.gpkg")
    # convert_sosi_to_geojson(str(input_file), Path(__file__).parent.parent / "test_data" / "14365_Vertikalbilde.GeoJson")
    #
    #
    # # TODO: this should be moved and recived from the user in SKAVL
    # image_folder_path = Path(r"C:\Users\Admin\Documents\bachelor-thesis\ImageDataTest\test")
    # gpgk_path = Path(__file__).parent.parent / "test_data" / "HX-14365_Vertikalbilde.gpkg"
    #
    #
    # gdf = get_gdf_content(gpgk_path)
    # start_anomaly_analysis(gdf, image_folder_path)


if __name__ == "__main__":
    main()