import pandas as pd

from multiversum import Universe

settings = {
    "dimensions": {
        "x": "A",
        "y": "B",
    }
}

universe = Universe(
    settings=settings,
)

# Get the parsed universe settings
dimensions = universe.dimensions

final_data = pd.DataFrame({"value": [ord(dimensions["x"]) + ord(dimensions["y"])]})

universe.save_data(final_data)
