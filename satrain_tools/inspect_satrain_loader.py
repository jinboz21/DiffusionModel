"""Inspect local SatRain training or validation batches without downloading."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from satrain.data import get_files
from satrain.target import TargetConfig
from torch.utils.data import DataLoader, Dataset


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "satrain_store"
STATS_PATH = DATA_PATH / "gmi_xs_training_stats.json"
SOURCES = ("gmi", "geo", "ancillary", "target")
EXCLUDED_ANCILLARY = {
    "total_precipitation",
    "convective_precipitation",
    "surface_type",
    "sea_ice_concentration",
    "sea_surface_temperature",
}


class SatRainLocalDataset(Dataset):
    """Load paired 256x256 scenes with a separate radar quality mask."""

    def __init__(self, split: str = "training", stats_path: Path | None = None) -> None:
        if split not in ("training", "validation"):
            raise ValueError(f"Unsupported split: {split}")
        self.split = split
        files = get_files(
            base_sensor="gmi",
            split=split,
            input_data=["gmi", "geo", "ancillary"],
            geometry="gridded",
            subset="xs",
            data_path=DATA_PATH,
            download=False,
        )
        self.file_maps = {
            source: {Path(path).stem.rsplit("_", 1)[-1]: path for path in files[source]}
            for source in SOURCES
        }
        timestamps = set(self.file_maps["target"])
        if not timestamps or any(set(self.file_maps[source]) != timestamps for source in SOURCES):
            raise RuntimeError("Local SatRain modalities are not fully paired")
        self.timestamps = sorted(timestamps)
        self.target_config = TargetConfig(min_rqi=0.5, min_valid_fraction=0.5)

        first = self.file_maps["ancillary"][self.timestamps[0]]
        with xr.open_dataset(first) as ds:
            self.ancillary_variables = [
                name for name in ds.data_vars if name not in EXCLUDED_ANCILLARY
            ]

        self.stats = None
        if stats_path is not None:
            stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
            if (
                stats.get("base_sensor") != "gmi"
                or stats.get("geometry") != "gridded"
                or stats.get("subset") != "xs"
                or stats.get("statistics_split") != "training"
                or stats.get("ancillary_variables") != self.ancillary_variables
            ):
                raise ValueError("Statistics do not match this SatRain dataset")
            self.stats = {}
            for source, channels in (("gmi", 13), ("geo", 16), ("ancillary", len(self.ancillary_variables))):
                source_stats = stats["sources"][source]
                mean = np.asarray(source_stats["mean"], dtype=np.float32)
                std = np.asarray(source_stats["std"], dtype=np.float32)
                if mean.shape != (channels,) or std.shape != (channels,) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
                    raise ValueError(f"Invalid normalization statistics for {source}")
                self.stats[source] = (mean[:, None, None], std[:, None, None])

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, index: int):
        timestamp = self.timestamps[index]
        paths = {source: self.file_maps[source][timestamp] for source in SOURCES}

        with xr.open_dataset(paths["gmi"]) as ds:
            gmi = ds["observations"].transpose("channel", "latitude", "longitude").values
        with xr.open_dataset(paths["geo"]) as ds:
            geo = ds["observations"].transpose("channel", "latitude", "longitude").values
        with xr.open_dataset(paths["ancillary"]) as ds:
            ancillary = np.stack([ds[name].values for name in self.ancillary_variables])
        with xr.open_dataset(paths["target"]) as ds:
            rain = ds["surface_precip"].values.astype(np.float32)
            valid = ~self.target_config.get_mask(ds)

        arrays = {"gmi": gmi, "geo": geo, "ancillary": ancillary}
        inputs = {}
        for source, array in arrays.items():
            array = np.asarray(array, dtype=np.float32)
            if self.stats is not None:
                observed = np.isfinite(array)
                mean, std = self.stats[source]
                array = (np.where(observed, array, mean) - mean) / std
                inputs[f"{source}_observed"] = torch.from_numpy(
                    np.ascontiguousarray(observed)
                )
            inputs[source] = torch.from_numpy(np.ascontiguousarray(array))
        targets = {
            "rain": torch.from_numpy(
                np.ascontiguousarray(np.where(valid, rain, 0.0), dtype=np.float32)
            ),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
        }
        return inputs, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    dataset = SatRainLocalDataset(
        split=args.split,
        stats_path=STATS_PATH if args.normalize else None,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    inputs, targets = next(iter(loader))

    print("Split:", args.split)
    print("Paired scenes:", len(dataset))
    print("Ancillary variables:", dataset.ancillary_variables)
    for name in ("gmi", "geo", "ancillary"):
        tensor = inputs[name]
        print(f"{name}: shape={tuple(tensor.shape)}, finite_fraction={torch.isfinite(tensor).float().mean().item():.4f}")
        if args.normalize:
            print(f"{name} observed fraction: {inputs[f'{name}_observed'].float().mean().item():.4f}")
    print("rain shape:", tuple(targets["rain"].shape))
    print("valid mask shape:", tuple(targets["valid"].shape))
    print("valid radar fraction:", targets["valid"].float().mean().item())
