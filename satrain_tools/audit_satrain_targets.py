"""Summarize locally downloaded SatRain xs training targets without training."""

from pathlib import Path

import numpy as np
import xarray as xr
from satrain.target import TargetConfig


data_path = Path(__file__).resolve().parents[1] / "data" / "satrain_store"
target_dir = data_path / "satrain" / "gmi" / "training" / "xs" / "gridded"
target_files = sorted(target_dir.glob("**/target_*.nc"))
if not target_files:
    raise FileNotFoundError(f"No target files found in {target_dir}")

quality = TargetConfig(min_rqi=0.5, min_valid_fraction=0.5)
total_pixels = 0
valid_pixels = 0
rain_pixels = 0
heavy_pixels = 0
rain_scenes = 0
heavy_scenes = 0
max_valid_rain = 0.0

for path in target_files:
    with xr.open_dataset(path) as ds:
        rain = ds["surface_precip"].values
        valid = ~quality.get_mask(ds)
        usable_rain = rain[valid]

    total_pixels += rain.size
    valid_pixels += usable_rain.size
    scene_rain_pixels = int(np.count_nonzero(usable_rain > 0.1))
    scene_heavy_pixels = int(np.count_nonzero(usable_rain > 10.0))
    rain_pixels += scene_rain_pixels
    heavy_pixels += scene_heavy_pixels
    rain_scenes += scene_rain_pixels > 0
    heavy_scenes += scene_heavy_pixels > 0
    if usable_rain.size:
        max_valid_rain = max(max_valid_rain, float(usable_rain.max()))

if valid_pixels == 0:
    raise RuntimeError("No valid radar pixels passed the quality filters")

print("Scenes:", len(target_files))
print("Valid radar pixels:", valid_pixels, f"({valid_pixels / total_pixels:.2%} of all pixels)")
print("Rain pixels > 0.1 mm/h:", rain_pixels, f"({rain_pixels / valid_pixels:.2%} of valid pixels)")
print("Heavy rain pixels > 10 mm/h:", heavy_pixels, f"({heavy_pixels / valid_pixels:.4%} of valid pixels)")
print("Scenes with rain:", rain_scenes)
print("Scenes with heavy rain:", heavy_scenes)
print("Maximum valid rain rate (mm/h):", max_valid_rain)
