from pathlib import Path
import geopandas as gpd
from services.sosi_converter_service import convert_sosi_to_gpkg, convert_sosi_to_geojson

output_file = Path(__file__).parent.parent.parent /"tests" / "testdata" / "output.sos"

#TODO change the output file to a better location
def convert_sosi_get_gdf(input_file):
    """Load a SOSI file and return its content as a string.
    """
    print(output_file)
    convert_sosi_to_gpkg(str(input_file), str(output_file))
        
    return get_gdf_content(output_file)



def get_gdf_content(gpkg_path):
    """Load a GeoPackage file and return its content as a GeoDataFrame.
    """
    gdf = gpd.read_file(gpkg_path, layer="polygons", encoding="ISO-8859-1")
    return gdf

