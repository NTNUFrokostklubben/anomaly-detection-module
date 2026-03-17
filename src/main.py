
from utils.find_overlap import find_image_path
from pathlib import Path
import geopandas

from core import water_detector


# Start analysis
def check_img_existence(gdf: geopandas.GeoDataFrame, image_folder_path: Path, index: int) -> bool:
    """Checks if the image exists in the image folder based on the information in the GeoDataFrame at the given index.

    Args:
        gdf (geopandas.GeoDataFrame): The content of the gpkg file
        image_folder_path (Path): Path to the image to check
        index (int): The index of the photo in the gpkg file

    Returns:
        bool: Whenether the image exists in the image folder or not
    """
    row = gdf.iloc[index]
    img_num = row["bildenummer"]
    strip = 1

    img_path = Path(image_folder_path) / Path(find_image_path(gdf, img_num, strip))

    if img_path.exists():
        return True
    else:
        return False


def main():
    
    #TODO make the image folder path and gpkg path be dynamic based on the input from the user
    #TODO move the pipeline for checking all images into a separate function outside of main.

    WaterDetector.main()
    
    """image_folder_path = Path(__file__).parent.parent.parent / "HX_14365_NORDMORE_GSD10" / "RGB" # Shitty way to get the path, but it works for now
    gpgk_path = Path(__file__).parent.parent / "tests" / "testdata" / "test_file_short.gpkg"
    gdf = get_gdf_content(gpgk_path)
    image_count = len(gdf)

    t0 = time.perf_counter()
    for i in range(image_count - 1):
        img1_path = image_folder_path / find_image_path(gdf, gdf.iloc[i]["bildenummer"], 1)
        img2_path = image_folder_path / find_image_path(gdf, gdf.iloc[i + 1]["bildenummer"], 1)

        if not img1_path.exists() or not img2_path.exists():
            continue

        avg1, avg2, diff, t = check_difference_two_images(
            gdf,
            gdf.iloc[i]["bildenummer"],
            1,
            img1_path,
            gdf.iloc[i + 1]["bildenummer"],
            1,
            img2_path,
        )

        print("------------------------")
        print(f"Comparing image {gdf.iloc[i]['bildenummer']} and image {gdf.iloc[i+1]['bildenummer']}")
        print(f"Image {gdf.iloc[i]['bildenummer']} avg: {avg1}")
        print(f"Image {gdf.iloc[i+1]['bildenummer']} avg: {avg2}")
        print(f"Difference: {diff}")
        print(f"Time: {t:.6f}s\n")    
        
                
    print("Overall time:", time.perf_counter() - t0)  
    print(f"Found {image_count} images in the GeoPackage.")"""
    

if __name__ == "__main__":
    main()