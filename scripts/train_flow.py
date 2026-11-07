"""Train a conditional flow matching model on chronological ERA5-Land splits."""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "edm"))

from scripts.climate_dataset import ClimateDataset
from scripts.flow_model import MODEL_CONFIG, make_model
from training.flow_loss import FlowEDMLoss


def epoch_loss(net, loader, loss_fn, device, optimizer=None):
    net.train(optimizer is not None)
    total = 0.0
    count = 0
    for batch in loader:
        hr = batch["hr"].to(device)
        t = torch.rand(hr.shape[0], device=device)
        condition = batch["lr_up"].to(device)
        mask = batch["mask"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            loss_map = loss_fn(net, hr, t, condition, mask)
            loss = loss_map.sum() / mask.sum().clamp_min(1)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += loss.item() * hr.shape[0]
        count += hr.shape[0]
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "era5_downscaling")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / "flow_best.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Small-run diagnostic; omit for the full training split")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but this PyTorch environment has no CUDA support")
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(42)

    stats_path = args.data_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    train_data = ClimateDataset(args.data_dir / "train", stats_path)
    val_data = ClimateDataset(args.data_dir / "val", stats_path)
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            parser.error("--max-train-samples must be positive")
        train_data = Subset(train_data, range(min(args.max_train_samples, len(train_data))))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    print(f"Device: {device}; training: {len(train_data)}; validation: {len(val_data)}", flush=True)

    net = make_model(device)
    loss_fn = FlowEDMLoss(sigma_min=0.01)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    best = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = epoch_loss(net, train_loader, loss_fn, device, optimizer)
        # Use identical noise for each validation pass, without resetting the
        # training RNG outside this context.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(12345)
            val_loss = epoch_loss(net, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs}: train={train_loss:.5f}, val={val_loss:.5f}", flush=True)
        if val_loss < best:
            best = val_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": net.state_dict(), "model_config": MODEL_CONFIG,
                        "training_method": "flow_matching", "sigma_min": loss_fn.sigma_min,
                        "stats": stats, "epoch": epoch, "val_loss": val_loss}, args.output)
            print(f"Saved best checkpoint: {args.output}", flush=True)


if __name__ == "__main__":
    main()
