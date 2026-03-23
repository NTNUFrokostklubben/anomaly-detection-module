

def slice_image_name(img_name: str) -> tuple[str, int, int, int]:
    """
    Slice an image file name and return its constituent parts as a tuple.
    :param img_name: the image file name.
    :return: a tuple containing prefix, img line, line number, and absolute number.
    """
    partitions = img_name.split('_')
    prefix = partitions[0]
    line = int(partitions[1])
    line_number = int(partitions[2])
    abs_number = int(partitions[3].split('.')[0])
    return prefix,line,line_number,abs_number
