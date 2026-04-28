# SKAVL Anomaly Detection Module

Module for detecting anomalies


## Running and Building

Building this application is relatively platform-agnostic, but it requires that conda is installed and that [conda-lock](https://github.com/conda/conda-lock) is installed.
One of the easiest ways of installing this is to install it on the base conda environment.

```shell
conda install -n base -c conda-forge conda-lock
```

Install the conda environment from the lockfile (skavl-anomaly should be replaced with whatever environment name you want to use in conda)
```shell
conda-lock install -n skavl-anomaly .\conda-lock.yml
```
Verify that an environment was created
```shell
conda env list
```

If the environment exists, activate it (replace skavl-anomaly with the actual env name)
```shell
conda activate skavl-anomaly
```

### Run

SOSI conversion for windows requires a sosi+fyba driver bundle. This can be fetched and unzipped by running.
This is done as requiring users and developers to install all the correct drivers and dependencies for GDAL with the SOSI driver and FYBA is a pain.
```shell
.\scripts\fetch_gdal_bundle_win.ps1
```

Arguments can be checked by running.
```shell
python .\src\server.py --help
```

The application can run as a CLI interface where it processes a dataset once and prints results to console or as a grpc server that waits for client connections to trigger services.
To launch the application as a server or cli the following syntax should be used.
```shell
python .\src\server.py cli
python .\src\server.py server
```

#### CLI
When the application is started as CLI, there are some required arguments that need to be passed for the analysis to run.
```shell
python .\src\server.py cli -i "C:path/to/sosi/file" -p "C:path/to/geotiff/folder"
python .\src\server.py cli --sosi-input "C:path/to/sosi/file" --image-path "C:path/to/geotiff/folder"
```
use `--help` for more information about optional arguments

#### Server
When running the application as a server no arguments are required, however there are optional arguments that can be passed.

```shell
python .\src\server.py server -p 50052
```

### Build

For this application to work, the sosi fyba driver bundle is required. This can be fetched using the script under
```shell
.\scripts\fetch_gdal_bundle_win.ps1
```

Once the env has been activated and bundle has been fetched, build for the platform you are currently on
```shell
pyinstaller server.spec
```
The `server.spec` file should handle linux/windows lib linking automatically. This does NOT work for linux currently.
Once the build is finished, it should be present under dist/server in the root of the project.

## Expanding

If more packages are required from conda it is important that these are added manually to the environment.yaml file and that a new conda-lock.yml file is generated.
Exporting the environment will cause multiple platform specific packages to be included in the environment.yaml file which breaks cross-compiling.


Once a module has been installed and added manually to environment.yaml, run the following command to generate the new conda-lock.yml file.
```shell
conda-lock -f environment.yaml -p win-64 -p linux-64
```

When install the new lock-file has been created, verify that a new environment can be created from it using conda-lock install.
Also check that the project builds and that it runs.

## Documentation

Documentation for the latest main branch release will be available on gh-pages. For generating local docs, use the pdoc package using the command.
```
pdoc src/ !src.skavl_proto -o docs/_build
```
https://ntnufrokostklubben.github.io/anomaly-detection-module/


## License
Open-source: AGPL-3.0 (see LICENSE)
Commercial: available on inquiry (see COMMERCIAL.md)