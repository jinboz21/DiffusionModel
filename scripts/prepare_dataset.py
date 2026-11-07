"""Create chronological HR/LR splits from ERA5-Land NetCDF files."""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def fill_nan_nearest(field):
    invalid = ~np.isfinite(field)
    if not invalid.any():
        return field.copy()
    if invalid.all():
        raise ValueError("A temperature field has no valid grid points")
    indices = distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return field[tuple(indices)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", dest="inputs",
                        help="NetCDF file; repeat for multiple files. Defaults to monthly downloads.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "era5_downscaling")
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    args = parser.parse_args()
    if min(args.train, args.val, args.test) <= 0:
        parser.error("All split sizes must be positive")
    paths = args.inputs or sorted((PROJECT_ROOT / "data" / "raw").glob("era5_land_t2m_20??_??_*.nc"))
    if not paths:
        parser.error("No monthly NetCDF files found; run download_era5.py first or pass --input")
    for path in paths:
        if not path.is_file():
            parser.error(f"Missing input file: {path}")

    total = args.train + args.val + args.test
    # Validate the time axis and grid before writing any output.
    entries = []
    grid = None
    for path in paths:
        with xr.open_dataset(path) as ds:
            if "t2m" not in ds or "valid_time" not in ds:
                raise ValueError(f"Missing t2m or valid_time in {path}")
            if ds["t2m"].dims != ("valid_time", "latitude", "longitude"):
                raise ValueError(f"Unexpected t2m dimensions in {path}: {ds['t2m'].dims}")
            if ds.sizes["latitude"] != 256 or ds.sizes["longitude"] != 256:
                raise ValueError(f"Expected a 256x256 grid in {path}, got {ds['t2m'].shape[1:]}")
            coords = (ds["latitude"].values, ds["longitude"].values)
            if grid is not None and not all(np.array_equal(a, b) for a, b in zip(grid, coords)):
                raise ValueError(f"Grid coordinates differ in {path}")
            grid = coords
            entries.extend((np.datetime64(t, "ns"), path, i)
                           for i, t in enumerate(ds["valid_time"].values))
    entries.sort(key=lambda item: item[0])
    if len(entries) != total:
        raise ValueError(f"Expected exactly {total} hours, found {len(entries)}. "
                         "Adjust --train/--val/--test or the requested date range.")
    for previous, current in zip(entries, entries[1:]):
        if current[0] - previous[0] != np.timedelta64(1, "h"):
            raise ValueError(f"Times are duplicated or not hourly: {previous[0]} -> {current[0]}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    split_sizes = {"train": args.train, "val": args.val, "test": args.test}
    for split in split_sizes:
        (args.output_dir / split).mkdir(parents=True, exist_ok=True)

    count = 0
    mean = 0.0
    m2 = 0.0
    low, high = float("inf"), float("-inf")
    opened_path = None
    ds = None
    try:
        for index, (time, path, source_index) in enumerate(entries):
            if path != opened_path:
                if ds is not None:
                    ds.close()
                ds = xr.open_dataset(path)
                opened_path = path
            raw = ds["t2m"].isel(valid_time=source_index).values.astype(np.float32)
            mask = np.isfinite(raw)
            hr = fill_nan_nearest(raw).astype(np.float32)
            lr = hr.reshape(32, 8, 32, 8).mean(axis=(1, 3)).astype(np.float32)
            split = "train" if index < args.train else "val" if index < args.train + args.val else "test"
            np.savez_compressed(args.output_dir / split / f"sample_{index:04d}.npz",
                                hr=hr, lr=lr, mask=mask.astype(np.float32), time=str(time))
            if split == "train":
                values = raw[mask].astype(np.float64)
                n = values.size
                batch_mean = float(values.mean())
                delta = batch_mean - mean
                new_count = count + n
                m2 += float(((values - batch_mean) ** 2).sum()) + delta ** 2 * count * n / new_count
                mean += delta * n / new_count
                count = new_count
                low = min(low, float(values.min()))
                high = max(high, float(values.max()))
            if (index + 1) % 100 == 0 or index + 1 == total:
                print(f"Prepared {index + 1}/{total} samples")
    finally:
        if ds is not None:
            ds.close()
    if count == 0 or m2 <= 0:
        raise ValueError("Training data has no valid temperature variation")
    stats = {
        "mean": mean, "std": float(np.sqrt(m2 / count)),
        "min": low, "max": high, "valid_pixel_count": count,
        "splits": split_sizes, "first_time": str(entries[0][0]),
        "last_time": str(entries[-1][0]),
        "source_files": [str(p) for p in paths],
    }
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
