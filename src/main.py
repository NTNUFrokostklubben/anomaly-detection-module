from utils.load_sosi_content import get_gdf_content 
from utils.find_overlap import find_image_path
from core.color_diff import check_difference_two_images_old
from core.color_diff import check_difference_two_images
from pathlib import Path


# Start analysis
def check_img_existence(gdf, image_folder_path, index):
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
    
    image_folder_path = Path(__file__).parent.parent.parent / "HX_14365_NORDMORE_GSD10" / "RGB" # Shitty way to get the path, but it works for now
    gpgk_path = Path(__file__).parent.parent / "tests" / "testdata" / "test_file_short.gpkg"
    gdf = get_gdf_content(gpgk_path)
    image_count = len(gdf)
    
    count = 0
    
    while (count < image_count-1):
        if check_img_existence(gdf, image_folder_path, count) and check_img_existence(gdf, image_folder_path, count+1):
            
            avg1, avg2, difference, time = check_difference_two_images(gdf, 
                                    gdf.iloc[count]["bildenummer"],
                                    1, 
                                    image_folder_path / find_image_path(gdf, gdf.iloc[count]["bildenummer"], 1),
                                    gdf.iloc[count+1]["bildenummer"],
                                    1, 
                                    image_folder_path / find_image_path(gdf, gdf.iloc[count+1]["bildenummer"], 1))
            
            
            print("------------------------")
            print(f"Comparing image {gdf.iloc[count]['bildenummer']} and image {gdf.iloc[count+1]['bildenummer']}")
            print("---------")
            print(f"Image {gdf.iloc[count]['bildenummer']} avg: {avg1}")
            print(f"Image {gdf.iloc[count+1]['bildenummer']} avg: {avg2}")
            print(f"Difference: {difference}")
            print(f"Time: {time:.6f}s")
            print("\n")
        count += 1            
        
    print(f"Found {image_count} images in the GeoPackage.")
    
    
    

if __name__ == "__main__":
    main()