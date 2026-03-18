import imagecodecs


def test_tifffile_uses_turbojpeg():
    """
    Ensures that tifffile uses turbojpeg for fast tiff file reading.
    :return:
    """
    assert imagecodecs.turbojpeg_version() is not None
