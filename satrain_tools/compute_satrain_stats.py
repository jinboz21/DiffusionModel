"""Compute per-channel input normalization statistics from local xs training scenes."""

import json
from pathlib import Path

import numpy as np

from satrain_tools.inspect_satrain_loader import SatRainLocalDataset

output_path = Path(__file__).resolve().parents[1] / "data" / "satrain_store" / "gmi_xs_training_stats.json"
dataset = SatRainLocalDataset()
accumulators = {}

for index in range(len(dataset)):
    inputs, _ = dataset[index]
    for source, tensor in inputs.items():
        values = tensor.numpy().astype(np.float64)
        finite = np.isfinite(values)
        safe = np.where(finite, values, 0.0)
        if source not in accumulators:
            channels = values.shape[0]
            accumulators[source] = {
                "count": np.zeros(channels, dtype=np.int64),
                "sum": np.zeros(channels, dtype=np.float64),
                "sum_sq": np.zeros(channels, dtype=np.float64),
            }
        accumulators[source]["count"] += finite.sum(axis=(1, 2))
        accumulators[source]["sum"] += safe.sum(axis=(1, 2))
        accumulators[source]["sum_sq"] += np.square(safe).sum(axis=(1, 2))
    if (index + 1) % 100 == 0 or index + 1 == len(dataset):
        print(f"Processed {index + 1}/{len(dataset)} training scenes")

stats = {
    "base_sensor": "gmi",
    "geometry": "gridded",
    "subset": "xs",
    "statistics_split": "training",
    "scene_count": len(dataset),
    "ancillary_variables": dataset.ancillary_variables,
    "sources": {},
}
for source, acc in accumulators.items():
    count = acc["count"]
    if np.any(count == 0):
        raise RuntimeError(f"{source} has channels with no finite training values")
    mean = acc["sum"] / count
    variance = np.maximum(acc["sum_sq"] / count - np.square(mean), 0.0)
    raw_std = np.sqrt(variance)
    std = np.where(raw_std > 1e-6, raw_std, 1.0)
    stats["sources"][source] = {
        "count": count.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "constant_channels": np.flatnonzero(raw_std <= 1e-6).tolist(),
    }
    valid_fraction = count / (len(dataset) * 256 * 256)
    print(
        f"{source}: {len(count)} channels, "
        f"finite fraction {valid_fraction.min():.2%} to {valid_fraction.max():.2%}, "
        f"constant channels {stats['sources'][source]['constant_channels']}"
    )

output_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
print("Wrote:", output_path)
