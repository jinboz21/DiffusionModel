"""Download hourly ERA5-Land 2 m temperature in monthly NetCDF chunks.

Defaults cover 2,400 hours (2025-01-01 through 2025-04-10 inclusive).
Requires a configured CDS API account.
"""

import argparse
import calendar
from datetime import date, timedelta
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AREA = [55.0, -120.0, 29.5, -94.5]  # North, west, south, east


def validate_file(path, request):
    """Reject truncated or mismatched files before they are reused."""
    expected_hours = len(request["day"]) * 24
    with xr.open_dataset(path) as ds:
        if "t2m" not in ds or ds["t2m"].dims != ("valid_time", "latitude", "longitude"):
            raise ValueError(f"Unexpected variables or dimensions in {path}")
        if ds["t2m"].shape != (expected_hours, 256, 256):
            raise ValueError(f"Unexpected grid or hour count in {path}: {ds['t2m'].shape}")
        first = np.datetime64(f"{request['year'][0]}-{request['month'][0]}-{request['day'][0]}T00:00")
        last = np.datetime64(f"{request['year'][0]}-{request['month'][0]}-{request['day'][-1]}T23:00")
        times = ds["valid_time"].values
        if times[0] != first or times[-1] != last:
            raise ValueError(f"Unexpected time range in {path}: {times[0]} to {times[-1]}")


def month_requests(start: date, end: date):
    current = start.replace(day=1)
    while current <= end:
        last_day = calendar.monthrange(current.year, current.month)[1]
        days = [
            f"{day:02d}"
            for day in range(1, last_day + 1)
            if start <= date(current.year, current.month, day) <= end
        ]
        if days:
            yield current, {
                "variable": ["2m_temperature"],
                "year": [str(current.year)],
                "month": [f"{current.month:02d}"],
                "day": days,
                "time": [f"{hour:02d}:00" for hour in range(24)],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 4, 10))
    parser.add_argument("--area", nargs=4, type=float, default=DEFAULT_AREA,
                        metavar=("NORTH", "WEST", "SOUTH", "EAST"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("--end must be on or after --start")
    north, west, south, east = args.area
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        parser.error("--area must be NORTH WEST SOUTH EAST within geographic bounds")

    pending = []
    for month, request in month_requests(args.start, args.end):
        request["area"] = args.area
        # Include requested days so a partial-month request cannot reuse
        # a file created for a different range.
        first, last = request["day"][0], request["day"][-1]
        target = args.output_dir / f"era5_land_t2m_{month:%Y_%m}_{first}-{last}.nc"
        if target.exists():
            if target.stat().st_size == 0:
                raise RuntimeError(f"Empty download target: {target}")
            validate_file(target, request)
            print(f"Exists, skipping: {target}")
        else:
            print(f"Request: {target} ({len(request['day']) * 24} hours)")
            pending.append((request, target))

    if args.dry_run or not pending:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    for request, target in pending:
        client.retrieve("reanalysis-era5-land", request, str(target))
        validate_file(target, request)
        print(f"Downloaded: {target}")


if __name__ == "__main__":
    main()
