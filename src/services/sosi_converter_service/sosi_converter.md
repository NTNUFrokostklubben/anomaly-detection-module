## Service for converting a sosi file into other file formats
To make it work right now you new to download the bundle files for gdal and the fyba library. 

```
├── lib/
│   └── gdal_bundle/
│       ├── bin/
│       ├── gdalplugins/
│       └── share/
```

#### Currently supported:
- SOSI --> GPKG


#### Example on how to use it (SOSI-->GPKG)

```python
from pathlib import Path
from src.services.sosi_converter_service import convert_sosi_to_gpkg

input_file = Path(__file__).parent / "file.sos"
output_file = Path(__file__).parent / "file.gpkg"

convert_sosi_to_gpkg(str(input_file), str(output_file))
```