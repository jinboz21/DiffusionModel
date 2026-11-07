from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


TEMP_MEAN = 268.6354034347734
TEMP_STD = 9.706773711322965


class ClimateDataset(Dataset):
    def __init__(self, data_dir, stats_path=None):#执行 ClimateDataset(...)、创建数据集对象时调用
        self.data_dir = Path(data_dir)

        # Dataset splits share training-only statistics in their parent folder.
        # Keep the legacy constants for the existing 24-sample prototype.
        if stats_path is None:
            candidate = self.data_dir.parent / "stats.json"
            stats_path = candidate if candidate.is_file() else None
        if stats_path is not None:
            stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
            self.mean = float(stats["mean"])
            self.std = float(stats["std"])
        else:
            self.mean = TEMP_MEAN
            self.std = TEMP_STD
        if self.std <= 0:
            raise ValueError("Temperature standard deviation must be positive")

        self.files = sorted(
            self.data_dir.glob("sample_*.npz")
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No sample_*.npz files found in {self.data_dir}"
            )

    def __len__(self):#执行 len(train_data) 时调用
        return len(self.files)

    def __getitem__(self, index):#执行 train_data[0] 时调用

        file = self.files[index]

        with np.load(file) as data:
            hr = data["hr"].astype(np.float32)
            lr = data["lr"].astype(np.float32)
            mask = data["mask"].astype(np.float32)

        # ----------------------------------------
        # Normalize HR and LR using same statistics
        # ----------------------------------------

        hr = (hr - self.mean) / self.std
        lr = (lr - self.mean) / self.std

        # ----------------------------------------
        # Convert numpy -> torch
        # Add channel dimension:
        #
        # (H, W)
        #   ->
        # (1, H, W)
        # ----------------------------------------

        hr = torch.from_numpy(hr).unsqueeze(0)

        lr = torch.from_numpy(lr).unsqueeze(0)

        mask = torch.from_numpy(mask).unsqueeze(0)

        # ----------------------------------------
        # Upsample LR:
        #
        # (1, 32, 32)
        #     ->
        # (1, 256, 256)
        # ----------------------------------------

        lr_up = F.interpolate(
            lr.unsqueeze(0),
            size=hr.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return {
            "hr": hr,
            "lr": lr,
            "lr_up": lr_up,
            "mask": mask,
        }


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    data_dir = (
        project_root
        / "data"
        / "processed"
    )

    dataset = ClimateDataset(
        data_dir
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print()
    print("Single sample:")

    print(
        "HR:",
        sample["hr"].shape
    )

    print(
        "LR:",
        sample["lr"].shape
    )

    print(
        "LR up:",
        sample["lr_up"].shape
    )

    print(
        "Mask:",
        sample["mask"].shape
    )

    print()

    print(
        "HR mean:",
        sample["hr"].mean().item()
    )

    print(
        "HR std:",
        sample["hr"].std().item()
    )

    # ----------------------------------------
    # DataLoader test
    # ----------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    batch = next(iter(loader))

    print()
    print("Batch:")

    print(
        "HR:",
        batch["hr"].shape
    )

    print(
        "LR:",
        batch["lr"].shape
    )

    print(
        "LR up:",
        batch["lr_up"].shape
    )

    print(
        "Mask:",
        batch["mask"].shape
    )
