from typing import Any

import numpy as np
import core.water_detector as wd
from numba import prange, njit
import utils.db_connector  as db
from entity import image


@njit(parallel=True, cache=True)
def calculate_average_color_block(img_arr: np.ndarray[tuple[int, int, int]],  increment: int) -> np.ndarray[tuple[float]]:
    """
    Calculate the average color in a block of size "increment" squared, and then return a list of values.
    :param img_arr: the image to process. Expects (Band, H, W)
    :param increment: the amount of jumps to do and the shape of the block.
    :return: the array of color values for the image.
    """

    _, y_shape,x_shape  = img_arr.shape
    y_blocks = (y_shape + increment - 1) // increment
    x_blocks = (x_shape + increment - 1) // increment
    num_blocks = y_blocks*x_blocks
    color_values = np.zeros((num_blocks, 3))

    for by in prange(y_blocks):
        for bx in prange(x_blocks):
            y_start = by * increment
            x_start = bx * increment
            y_end = min(y_start + increment, y_shape)
            x_end = min(x_start + increment, x_shape)
            r, g, b = wd.block_mean_rgb(img_arr, y_start, y_end, x_start, x_end, None)
            color_values[by * x_blocks + bx, 0] = r
            color_values[by * x_blocks + bx, 1] = g
            color_values[by * x_blocks + bx, 2] = b

    return color_values



def detect_artifact_consistency(images: list[image.Image], increment: int) -> float | None:
    """
    Detects artifact blocks by measuring color consistency across multiple images.
    A block that stays the same color across all images likely contains an artifact.

    :param images: list of image objects, Must contain at least img_arr and img_id
    :param increment: block size in pixels, total pixels is increment squared.
    :return: 1D array of per-block consistency scores across all images on the line.
             Low values (near 0) indicate the block has the same color across all images — likely an artifact.
             High values indicate the block varies across images — genuine scene content.
    """


    conn = db.DbConnector()
    line_values = conn.get_artifact_data_line(images[0].prefix, images[0].line)
    if line_values is None and len(images) < 2:
        return None
    for img in images:
        data = calculate_average_color_block(img.img_arr, increment)
        img.artifact_data = image.ArtifactData( data=data,dtype=data.dtype ,shape=data.shape,offset=increment )
        conn.add_artifact_data(img.img_id,data=img.artifact_data.data, offset= increment )

    all_data = line_values + [img.artifact_data.data for img in images]
    if len(all_data) < 10:
        return None

    stacked = np.stack(all_data)
    # (N, num_blocks, 3)
    consistency = (stacked.max(axis=0) - stacked.min(axis=0)).sum(axis=1) / 3.0  # (num_blocks,)
    return consistency

