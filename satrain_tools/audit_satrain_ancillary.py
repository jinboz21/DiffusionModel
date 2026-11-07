"""Audit raw ancillary availability in local SatRain validation scenes."""

import json
from pathlib import Path

import numpy as np
import xarray as xr
from satrain.data import get_files


data_path = Path(__file__).resolve().parents[1] / "data" / "satrain_store"
stats_path = data_path / "gmi_xs_training_stats.json"
stats = json.loads(stats_path.read_text(encoding="utf-8"))
variables = stats["ancillary_variables"]

files = get_files(
    base_sensor="gmi",
    split="validation",
    input_data=["ancillary"],
    geometry="gridded",
    subset="xs",
    data_path=data_path,
    download=False,
)["ancillary"]
if not files:
    raise FileNotFoundError("No local validation ancillary files found")

finite_counts = np.zeros(len(variables), dtype=np.int64)
scenes_with_values = np.zeros(len(variables), dtype=np.int64)
pixel_counts = np.zeros(len(variables), dtype=np.int64)

for path in files:
    with xr.open_dataset(path) as ds:
        for index, name in enumerate(variables):
            finite = np.isfinite(ds[name].values)
            finite_counts[index] += int(finite.sum())
            scenes_with_values[index] += bool(finite.any())
            pixel_counts[index] += finite.size

print("Validation scenes:", len(files))
for name, count, scenes, total in zip(variables, finite_counts, scenes_with_values, pixel_counts):
    print(f"{name}: finite={count / total:.2%}, scenes_with_values={scenes}/{len(files)}")
