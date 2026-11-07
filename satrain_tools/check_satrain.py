"""Inspect one locally downloaded, fully matched SatRain scene."""

import argparse
from pathlib import Path

import xarray as xr
from satrain.data import get_files


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--split", choices=("training", "validation"), default="training")
args = parser.parse_args()

data_path = Path(__file__).resolve().parents[1] / "data" / "satrain_store"
if not data_path.is_dir():
    raise FileNotFoundError(f"SatRain data directory not found: {data_path}")

# Read only the already downloaded GMI xs subset.
files = get_files(
    base_sensor="gmi",
    split=args.split,
    input_data=["gmi", "geo", "ancillary"],
    geometry="gridded",
    subset="xs",
    data_path=data_path,
    download=False,
)

sources = ["gmi", "geo", "ancillary", "target"]
file_maps = {}
for source in sources:
    paths = files[source]
    file_maps[source] = {Path(path).stem.split("_")[-1]: path for path in paths}
    if len(file_maps[source]) != len(paths):
        raise RuntimeError(f"Duplicate timestamps found in {source} files")
    print(f"{source}: {len(paths)} files")

target_times = set(file_maps["target"])
if not target_times:
    raise RuntimeError("No local SatRain target files found")

for source in sources:
    source_times = set(file_maps[source])
    missing = target_times - source_times
    extra = source_times - target_times
    if missing or extra:
        raise RuntimeError(
            f"{source} timestamps do not match target: "
            f"{len(missing)} missing, {len(extra)} extra"
        )

print("Split:", args.split)
print("Fully matched scenes:", len(target_times))
timestamp = sorted(target_times)[0]
print("Matched sample:", timestamp)

for source in sources:
    path = file_maps[source][timestamp]
    with xr.open_dataset(path) as ds:
        print(f"\n=== {source.upper()} ===")
        print("Path:", path)
        print("Dimensions:", dict(ds.sizes))
        print("Variables:", list(ds.data_vars))

        if source == "target":
            rain = ds["surface_precip"]
            print("Valid fraction:", float(rain.notnull().mean()))
            print("Max rain rate (mm/h):", float(rain.max(skipna=True)))
