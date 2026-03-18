from utils.load_sosi_content import get_gdf_content
from core.pipeline_anomaly_detection import start_anomaly_analysis
from pathlib import Path

def main():

    # TODO: this should be moved and recived from the user in SKAVL
    image_folder_path = Path(__file__).parent.parent.parent / "HX_14365_NORDMORE_GSD10" / "RGB" # Shitty way to get the path, but it works for now
    gpgk_path = Path(__file__).parent.parent / "tests" / "testdata" / "test_file_short.gpkg"
    gdf = get_gdf_content(gpgk_path)
    start_anomaly_analysis(gdf, image_folder_path)

if __name__ == "__main__":
    main()