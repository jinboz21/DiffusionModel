"""Run the official SatRain GMI CNN with local paths and device settings."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader
from satrain.input import GMI, calculate_input_features
from satrain.pytorch.datasets import SatRainSpatial
from satrain.target import TargetConfig

from .official_model import IPWGUNet


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "satrain_store"
INTERNAL_FEATURES = [32, 64, 128, 256, 512]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Inspect the official data/model on CPU without training")
    modes.add_argument("--smoke", action="store_true", help="One official training batch and one validation batch")
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.threads) < 1 or args.workers < 0:
        parser.error("Batch size, epochs and threads must be positive; workers must be nonnegative")
    if version("satrain") != "1.2.1":
        raise RuntimeError("This saved example is paired with satrain==1.2.1")
    source_info = json.loads((HERE / "source_info.json").read_text(encoding="utf-8"))
    source_hash = hashlib.sha256((HERE / "official_model.py").read_bytes()).hexdigest()
    if source_hash.lower() != source_info["model_sha256"].lower():
        raise RuntimeError("official_model.py differs from the extracted upstream source")

    L.seed_everything(args.seed, workers=True)
    torch.set_num_threads(args.threads)
    # These input settings and model widths are the settings in the official notebook.
    inputs = [GMI(normalize="minmax", nan=-1.5, include_angles=False)]
    target_config = TargetConfig(min_rqi=0.5)
    common = dict(
        base_sensor="gmi", geometry="gridded", subset="xs", retrieval_input=inputs,
        target_config=target_config, stack=True, download=False, data_path=args.data_path,
    )
    training = SatRainSpatial(split="training", augment=not (args.no_augment or args.check), **common)
    validation = SatRainSpatial(split="validation", augment=False, **common)
    training_loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        worker_init_fn=training.worker_init_fn,
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        worker_init_fn=validation.worker_init_fn,
    )
    input_features = calculate_input_features(inputs)
    model = IPWGUNet(
        input_features=input_features, internal_features=INTERNAL_FEATURES, n_epochs=args.epochs,
    )
    print(f"Official source: {source_info['commit']}", flush=True)
    print(f"Official SatRainSpatial: train={len(training)}, validation={len(validation)}", flush=True)
    print(f"GMI input channels: {input_features}; model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    if args.check:
        model.eval()
        with torch.no_grad():
            for split, dataset in (("training", training), ("validation", validation)):
                batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
                x, target = batch
                assert x.shape == (1, 13, 256, 256) and torch.isfinite(x).all()
                valid = torch.isfinite(target["surface_precip"])
                assert valid.any(), f"First {split} scene has no valid radar labels"
                prediction = model(x)
                for name, tensor in prediction.items():
                    assert tensor.shape == (1, 1, 256, 256) and torch.isfinite(tensor).all()
                    print(f"{split}: {name} shape={tuple(tensor.shape)}", flush=True)
                loss = model.training_step(batch, 0)
                assert torch.isfinite(loss), f"Nonfinite official loss for {split}"
                print(f"{split}: valid radar fraction={valid.float().mean().item():.4f}; official combined loss={loss.item():.5f}", flush=True)
        print("Official data/model check finished. No optimizer steps or model files were created.")
        return

    use_gpu = args.accelerator == "gpu" or (args.accelerator == "auto" and torch.cuda.is_available())
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --accelerator cpu")
    accelerator, precision = ("gpu", "16-mixed") if use_gpu else ("cpu", "32-true")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    run_dir = args.output_dir or HERE / "runs" / (("smoke_" if args.smoke else "gmi_") + stamp)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Output directory already has content: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": source_info, "input_features": input_features, "internal_features": INTERNAL_FEATURES,
        "n_epochs": args.epochs, "batch_size": args.batch_size, "workers": args.workers,
        "training_augmentation": training.augment, "accelerator": accelerator, "precision": precision,
        "data_path": str(args.data_path.resolve()), "seed": args.seed, "smoke": args.smoke,
        "input_config": {"name": "gmi", "normalize": "minmax", "nan": -1.5, "include_angles": False},
        "versions": {name: version(name) for name in ("satrain", "lightning", "torch", "torchvision")},
    }
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", filename="best-{epoch:02d}",
        monitor="val_loss", mode="min", save_top_k=1, save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=model.n_epochs, accelerator=accelerator, devices=1, precision=precision,
        fast_dev_run=args.smoke, default_root_dir=str(run_dir),
        callbacks=[] if args.smoke else [checkpoint], enable_checkpointing=not args.smoke,
        logger=False if args.smoke else CSVLogger(save_dir=str(run_dir), name="logs"),
        log_every_n_steps=1,
    )
    trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=validation_loader)
    if args.smoke:
        print("Official one-batch training/validation smoke run finished; no trained checkpoint saved.")
    else:
        print(f"Best official checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
