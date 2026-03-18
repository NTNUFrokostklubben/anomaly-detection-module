import imagecodecs


def test_tifffile_uses_turbojpeg():
    """
    Ensures that tifffile uses turbojpeg for fast tiff file reading.
    :return:
    """
    assert imagecodecs.jpeg8_version() is not None
