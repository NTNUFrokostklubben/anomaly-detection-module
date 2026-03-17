from core.color_diff import check_difference_two_images
from pathlib import Path
import time
from controller.image_cache_controller import ImageCache


def load_image_array(img1_path, img2_path, cache):
    t0 = time.perf_counter()
    arr1 = cache.get(img1_path)
    arr2 = cache.get(img2_path)
    t_load = time.perf_counter() - t0
    return arr1, arr2, t_load

def start_water_detection_analysis():
    print("----------- Water Detection  -------------")
    # Todo connect with the finished result of the water detection analysis
    return

def start_color_difference_analysis(gdf, i, arr1, arr2 ):

    avg1, avg2, diff, t = check_difference_two_images(
        gdf,
        int(gdf.iloc[i]["bildenummer"]),
        int(gdf.iloc[i]["stripenummer"]),
        arr1,
        int(gdf.iloc[i + 1]["bildenummer"]),
        int(gdf.iloc[i + 1]["stripenummer"]),
        arr2,
    )

    print("----------- Color Difference -------------")
    print(f"Comparing image {gdf.iloc[i]['bildenummer']} and image {gdf.iloc[i + 1]['bildenummer']}")
    print(f"Image {gdf.iloc[i]['bildenummer']} avg: {avg1}")
    print(f"Image {gdf.iloc[i + 1]['bildenummer']} avg: {avg2}")
    print(f"Difference: {diff}")
    print(f"Time analysis: {t:.6f}s\n")

def start_anomaly_analysis(gdf, image_folder_path: Path):

    image_count = len(gdf)

    t0 = time.perf_counter()
    cache = ImageCache(max_size=2)
    for i in range(image_count - 1):
        img1_path = image_folder_path / gdf.iloc[i]["bildefilRGB"]
        img2_path = image_folder_path / gdf.iloc[i + 1]["bildefilRGB"]

        if not img1_path.exists() or not img2_path.exists():
            return

        arr1, arr2, t_load = load_image_array(img1_path, img2_path, cache)

        print("------------------------------------------")
        print(f"Comparing image {gdf.iloc[i]['bildenummer']} and image {gdf.iloc[i + 1]['bildenummer']}")
        print(f"Loading images to arr : {t_load:.6f}s \n")

        start_color_difference_analysis(gdf, i, arr1, arr2)
        start_water_detection_analysis()
        print("\n")


    print("Overall time:", time.perf_counter() - t0)
    print(f"Found {image_count} images in the GeoPackage.")
