# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks.conda import collect_dynamic_libs as collect_conda_dynamic_libs

binaries = []
hiddenimports = []

# Collect Conda-shared DLLs from env/Library/bin for Windows (shared lib for Unix)
# This WILL need to be updated as source code changes and more code is added.
binaries += collect_conda_dynamic_libs("gdal", dependencies=True)
binaries += collect_conda_dynamic_libs("pillow", dependencies=True)
binaries += collect_conda_dynamic_libs("rasterio", dependencies=True)
binaries += collect_conda_dynamic_libs("opencv", dependencies=True)
binaries += collect_conda_dynamic_libs("pyogrio", dependencies=True)

# This WILL need to be updated as source code changes and more code is added.
hiddenimports += collect_submodules("rasterio")
hiddenimports += collect_submodules("numba")
hiddenimports += collect_submodules("skimage")
hiddenimports += collect_submodules("geopandas")
hiddenimports += collect_submodules("tifffile")
hiddenimports += collect_submodules("imagecodecs")
hiddenimports += collect_submodules("pyogrio")

# Linux-specific dependency (required by GDAL stack)
if sys.platform.startswith("linux"):
    conda_lib = Path(sys.prefix) / "lib"
    binaries += [(str(conda_lib / "libnsl.so.3"), ".")]

a = Analysis(
    [str(Path("src") / "main.py")],
    pathex=[],
    binaries=binaries,
    datas=[
        ('lib/bundle', 'lib/bundle'),
        (str(Path(sys.prefix) / 'Lib' / 'site-packages' / 'chardet' / 'models'), 'chardet/models'),
        (str(Path(sys.prefix) / 'Lib' / 'site-packages' / 'cv2' / 'python-3.12'), 'cv2/python-3.12'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="skavl-anomaly-detection-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory=".",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="server",
)