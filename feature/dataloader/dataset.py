from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class SwissReferendumDataset(Dataset):
    """
    Loads Swiss referendum input features and vote-share targets from CSV,
    applies per-column min-max normalization (mirrors KNNexe.py preprocessing).
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
        data_dir = Path(data_dir)

        X_raw = np.loadtxt(
            data_dir / "grosserDatensatzEingabe.csv", delimiter=";", dtype=np.float32
        )
        Y_raw = np.loadtxt(
            data_dir / "grosserDatensatzAusgabe.csv", delimiter=";", dtype=np.float32
        )
        if Y_raw.ndim == 1:
            Y_raw = Y_raw[:, None]

        # Min-max normalization — same logic as the original KNNexe.py
        self.x_min = X_raw.min(axis=0)
        self.x_max = X_raw.max(axis=0)
        x_denom = np.where(self.x_max - self.x_min == 0, 1.0, self.x_max - self.x_min)
        X_norm = (X_raw - self.x_min) / x_denom

        self.y_min = Y_raw.min(axis=0)
        self.y_max = Y_raw.max(axis=0)
        y_denom = np.where(self.y_max - self.y_min == 0, 1.0, self.y_max - self.y_min)
        Y_norm = (Y_raw - self.y_min) / y_denom

        self.X = torch.from_numpy(X_norm)
        self.Y = torch.from_numpy(Y_norm)
        self.n_features: int = self.X.shape[1]
        self.n_outputs: int = self.Y.shape[1]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]

    def inverse_transform_y(self, Y_norm: torch.Tensor) -> torch.Tensor:
        """Converts normalized predictions back to original vote-share percentages."""
        y_min = torch.tensor(self.y_min, dtype=Y_norm.dtype, device=Y_norm.device)
        y_range = torch.tensor(
            self.y_max - self.y_min, dtype=Y_norm.dtype, device=Y_norm.device
        )
        return Y_norm * y_range + y_min


def build_dataloaders(
    data_dir: str | Path | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, SwissReferendumDataset]:
    """
    Splits the full dataset into train / val / test loaders.
    The dataset object is also returned so callers can call inverse_transform_y.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    dataset = SwissReferendumDataset(data_dir)
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset
