"""Check spatial alignment and plot one locally downloaded SatRain scene."""

from contextlib import ExitStack
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from satrain.data import get_files


project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "satrain_store"
output_path = project_root / "experiments" / "satrain_first_scene.png"

files = get_files(
    base_sensor="gmi",
    split="training",
    input_data=["gmi", "geo", "ancillary"],
    geometry="gridded",
    subset="xs",
    data_path=data_path,
    download=False,
)
sources = ("gmi", "geo", "ancillary", "target")
file_maps = {
    source: {Path(path).stem.rsplit("_", 1)[-1]: path for path in files[source]}
    for source in sources
}
target_times = set(file_maps["target"])
if not target_times or any(set(file_maps[source]) != target_times for source in sources):
    raise RuntimeError("GMI, GEO, ancillary, and target files are not fully paired")

timestamp = sorted(target_times)[0]
with ExitStack() as stack:
    datasets = {
        source: stack.enter_context(xr.open_dataset(file_maps[source][timestamp]))
        for source in sources
    }
    target = datasets["target"]
    for source, ds in datasets.items():
        if ds.sizes.get("latitude") != 256 or ds.sizes.get("longitude") != 256:
            raise RuntimeError(f"Unexpected spatial shape in {source}: {dict(ds.sizes)}")
        for coord in ("latitude", "longitude"):
            if not np.allclose(ds[coord].values, target[coord].values, rtol=0, atol=1e-4):
                raise RuntimeError(f"{source} {coord} coordinates do not match target")

    gmi = datasets["gmi"]["observations"].isel(channel=0).values
    geo_data = datasets["geo"]["observations"]
    geo_channel = int(np.isfinite(geo_data.values).sum(axis=(0, 1)).argmax())
    geo = geo_data.isel(channel=geo_channel).values
    rain = target["surface_precip"].values
    rqi = target["radar_quality_index"].values
    quality_mask = np.isfinite(rain) & np.isfinite(rqi) & (rqi >= 0.5)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
panels = (
    (gmi, "GMI channel 0", "gray"),
    (geo, f"GEO channel {geo_channel}", "gray"),
    (np.where(quality_mask, rain, np.nan), "Radar rain (RQI >= 0.5)", "turbo"),
    (rqi, "Radar quality index", "viridis"),
)
for ax, (array, title, cmap) in zip(axes, panels):
    image = ax.imshow(array, origin="upper", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Longitude pixel")
    ax.set_ylabel("Latitude pixel")
    fig.colorbar(image, ax=ax, shrink=0.8)

fig.suptitle(f"SatRain GMI training scene {timestamp}")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
plt.close(fig)

print("Matched timestamp:", timestamp)
print("Spatial coordinates: matched across all four files")
print("Radar valid + RQI >= 0.5 fraction:", float(quality_mask.mean()))
print("Figure:", output_path)
