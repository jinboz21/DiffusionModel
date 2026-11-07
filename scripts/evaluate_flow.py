"""Evaluate conditional flow matching against LR interpolation and ERA5 truth."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "edm"))

from scripts.climate_dataset import ClimateDataset
from scripts.flow_model import flow_sampler, make_model


def metrics(prediction, truth, mask, data_range):
    error = prediction[mask] - truth[mask]
    mse = float(np.mean(error ** 2))
    psnr = float(10 * np.log10(data_range ** 2 / mse)) if mse > 0 else float("inf")
    x, y = prediction.astype(np.float64), truth.astype(np.float64)
    mx, my = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
    vx = np.maximum(gaussian_filter(x * x, 1.5) - mx * mx, 0)
    vy = np.maximum(gaussian_filter(y * y, 1.5) - my * my, 0)
    covariance = gaussian_filter(x * y, 1.5) - mx * my
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim_map = ((2 * mx * my + c1) * (2 * covariance + c2)
                / ((mx * mx + my * my + c1) * (vx + vy + c2)))
    return {"mse_k2": mse, "psnr_db": psnr, "ssim": float(ssim_map[mask].mean())}


def radial_spectrum(field, mask):
    field = field.astype(np.float64).copy()
    field[~mask] = float(field[mask].mean())
    field -= field.mean()
    window = np.outer(np.hanning(field.shape[0]), np.hanning(field.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(field * window))) ** 2
    yy, xx = np.indices(field.shape)
    radius = np.rint(np.hypot(yy - field.shape[0] // 2,
                             xx - field.shape[1] // 2)).astype(int)
    maximum = min(field.shape) // 2
    sums = np.bincount(radius.ravel(), weights=power.ravel(), minlength=maximum + 1)
    counts = np.bincount(radius.ravel(), minlength=maximum + 1)
    return sums[1:maximum + 1] / counts[1:maximum + 1]


def save_example(output, baseline, generated, truth, mask):
    lower, upper = np.percentile(truth[mask], [1, 99])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, title, field in zip(axes, ["Upsampled LR", "Flow matching", "ERA5 HR"],
                                [baseline, generated, truth]):
        image = ax.imshow(np.where(mask, field, np.nan), vmin=lower, vmax=upper,
                          cmap="coolwarm")
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(image, ax=axes, label="Temperature (K)", shrink=0.8)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "era5_downscaling")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "experiments" / "flow_best.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "flow_evaluation")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Evaluate only the first N test samples")
    parser.add_argument("--steps", type=int, default=20, help="Heun integration steps")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.steps < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("--steps and --max-samples must be positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but this PyTorch environment has no CUDA support")
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))

    stats_path = args.data_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    data = ClimateDataset(args.data_dir / "test", stats_path)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint["stats"]["mean"] != stats["mean"] or checkpoint["stats"]["std"] != stats["std"]:
        raise ValueError("Checkpoint statistics do not match this dataset")
    if checkpoint.get("training_method") != "flow_matching":
        raise ValueError("Checkpoint does not identify a flow matching model")
    net = make_model(device, checkpoint["model_config"])
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(data), args.max_samples or len(data))
    data_range = stats["max"] - stats["min"]
    edges = np.linspace(stats["min"] - 5 * stats["std"],
                        stats["max"] + 5 * stats["std"], 101)
    hist = {name: np.zeros(100) for name in ("LR", "Flow", "ERA5")}
    spectra = {name: np.zeros(128) for name in hist}
    rows = []
    for index in range(n):
        sample = data[index]
        mask = sample["mask"][0].numpy().astype(bool)
        if not mask.any():
            raise ValueError(f"Test sample {index} has no valid grid cells")
        truth = sample["hr"][0].numpy() * stats["std"] + stats["mean"]
        baseline = sample["lr_up"][0].numpy() * stats["std"] + stats["mean"]
        torch.manual_seed(2025 + index)
        condition = sample["lr_up"].unsqueeze(0).to(device)
        generated = flow_sampler(net, condition, num_steps=args.steps)[0, 0].cpu().numpy()
        generated = generated * stats["std"] + stats["mean"]
        row = {"sample": data.files[index].stem}
        for name, field in (("LR", baseline), ("Flow", generated), ("ERA5", truth)):
            hist[name] += np.histogram(field[mask], bins=edges)[0]
            spectra[name] += radial_spectrum(field, mask)
            if name != "ERA5":
                row.update({f"{name.lower()}_{key}": value
                            for key, value in metrics(field, truth, mask, data_range).items()})
        rows.append(row)
        if index == 0:
            save_example(args.output_dir / "example.png", baseline, generated, truth, mask)
        print(f"Evaluated {index + 1}/{n}", flush=True)

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "sample"}
    summary.update({"samples": n, "steps": args.steps, "psnr_data_range_k": data_range})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in hist.items():
        density = values / max(values.sum(), 1) / np.diff(edges)
        ax.plot((edges[:-1] + edges[1:]) / 2, density, label=name)
    ax.set(xlabel="Temperature (K)", ylabel="Density", title="Valid-pixel temperature distributions")
    ax.legend()
    fig.savefig(args.output_dir / "histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in spectra.items():
        ax.loglog(np.arange(1, len(values) + 1), values / n, label=name)
    ax.set(xlabel="Radial spatial frequency bin", ylabel="Mean power", title="Spatial power spectra")
    ax.legend()
    fig.savefig(args.output_dir / "power_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
