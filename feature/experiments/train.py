"""
Training experiment for Swiss referendum vote-share prediction.

Reproduces KNN.py / KNNexe.py in PyTorch and adds:
  - Automatic GPU / MPS / CPU selection
  - Autograd  (no hand-written backprop)
  - ReduceLROnPlateau learning-rate scheduling
  - Gradient clipping
  - Optional batch / layer normalisation
  - Model checkpointing (best validation weights auto-saved)
  - Extended metrics: MSE, RMSE, MAE, R2
  - Denormalized RMSE / MAE in original % units for interpretability
  - CLI interface matching the original's parameter names where possible

Usage:
    python -m feature.experiments.train
    python -m feature.experiments.train --optimizer adam --hidden 50 50 50 50 50 --max_epochs 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adagrad, Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature.dataloader.dataset import SwissReferendumDataset, build_dataloaders
from feature.model.mlp import MLP


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1.0 - ss_res / (ss_tot + 1e-12)).item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    loss_sum = 0.0
    preds_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        loss_sum += criterion(pred, Y).item() * len(X)
        preds_list.append(pred.cpu())
        targets_list.append(Y.cpu())

    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)
    n = len(targets)
    mse = loss_sum / n

    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": (preds - targets).abs().mean().item(),
        "r2": _r2(targets, preds),
        "preds": preds,
        "targets": targets,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    hidden_layers: tuple[int, ...] = (50, 50, 50, 50),
    activation: str = "relu",
    dropout: float = 0.0,
    norm: str = "none",
    optimizer_name: str = "adam",
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 200,
    patience: int = 25,
    grad_clip: float = 1.0,
    loss_fn: str = "mse",
    lr_schedule: bool = True,
    data_dir: str | Path | None = None,
    save_dir: str | Path | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[MLP, dict, dict]:
    """
    Trains an MLP on the Swiss referendum dataset and returns
    (trained_model, training_history, test_metrics).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device selection: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"Device: {device}")

    # ---- Data ---------------------------------------------------------------
    train_loader, val_loader, test_loader, dataset = build_dataloaders(
        data_dir=data_dir, batch_size=batch_size, seed=seed
    )
    if verbose:
        print(
            f"Dataset: {len(dataset)} samples  "
            f"(train {len(train_loader.dataset)} / "
            f"val {len(val_loader.dataset)} / "
            f"test {len(test_loader.dataset)})"
        )

    # ---- Model --------------------------------------------------------------
    model = MLP(
        n_features=dataset.n_features,
        n_outputs=dataset.n_outputs,
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout,
        norm=norm,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        arch = [dataset.n_features, *hidden_layers, dataset.n_outputs]
        print(f"Architecture: {arch}  |  Parameters: {n_params:,}")

    # ---- Loss ---------------------------------------------------------------
    loss_map = {"mse": nn.MSELoss(), "huber": nn.HuberLoss(), "mae": nn.L1Loss()}
    if loss_fn not in loss_map:
        raise ValueError(f"loss_fn must be one of {list(loss_map)}, got '{loss_fn}'")
    criterion = loss_map[loss_fn]

    # ---- Optimizer ----------------------------------------------------------
    opt_map = {
        "adam": lambda: Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
        "rmsprop": lambda: RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay),
        # SGD with Nesterov momentum mirrors the original's best SGD variant
        "sgd": lambda: SGD(
            model.parameters(), lr=lr, momentum=0.9,
            weight_decay=weight_decay, nesterov=True
        ),
        "adagrad": lambda: Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay),
    }
    if optimizer_name not in opt_map:
        raise ValueError(f"optimizer must be one of {list(opt_map)}, got '{optimizer_name}'")
    optimizer = opt_map[optimizer_name]()

    scheduler: ReduceLROnPlateau | None = None
    if lr_schedule:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )

    # ---- Checkpointing ------------------------------------------------------
    save_dir = Path(save_dir) if save_dir else Path(__file__).parent / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "best_model.pt"

    # ---- Training loop ------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_r2": []}
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), Y)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss_sum += loss.item() * len(X)

        train_loss = train_loss_sum / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["mse"]

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_metrics["r2"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if verbose and (epoch == 1 or epoch % 10 == 0):
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d}/{max_epochs}  "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"R2={val_metrics['r2']:.4f}  lr={lr_now:.2e}  patience={patience_counter}"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch} (patience={patience})")
            break

    elapsed = time.time() - t0
    if verbose:
        print(f"\nTraining complete in {elapsed:.1f}s")

    # ---- Test evaluation ----------------------------------------------------
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device)

    if verbose:
        print("\n--- Test Set Results (normalized) ---")
        print(f"  MSE:  {test_metrics['mse']:.6f}")
        print(f"  RMSE: {test_metrics['rmse']:.6f}")
        print(f"  MAE:  {test_metrics['mae']:.6f}")
        print(f"  R2:   {test_metrics['r2']:.4f}")

        preds_raw = dataset.inverse_transform_y(test_metrics["preds"])
        targets_raw = dataset.inverse_transform_y(test_metrics["targets"])
        rmse_raw = ((targets_raw - preds_raw) ** 2).mean().item() ** 0.5
        mae_raw = (targets_raw - preds_raw).abs().mean().item()
        print(f"\n--- Test Set Results (original % scale) ---")
        print(f"  RMSE: {rmse_raw:.3f}%")
        print(f"  MAE:  {mae_raw:.3f}%")

    _plot_results(history, test_metrics, dataset, save_dir)

    return model, history, test_metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(
    history: dict,
    test_metrics: dict,
    dataset: SwissReferendumDataset,
    save_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Loss curves
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train MSE")
    ax.plot(history["val_loss"], label="Val MSE")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Curves")
    ax.legend()

    # 2. Validation R2 over epochs
    ax = axes[1]
    ax.plot(history["val_r2"], color="tab:orange")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R2")
    ax.set_title("Validation R2")

    # 3. Scatter: predictions vs ground truth (denormalized to %)
    ax = axes[2]
    preds = dataset.inverse_transform_y(test_metrics["preds"]).numpy().ravel()
    targets = dataset.inverse_transform_y(test_metrics["targets"]).numpy().ravel()
    ax.scatter(targets, preds, alpha=0.35, s=12, color="tab:blue")
    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("True vote share (%)")
    ax.set_ylabel("Predicted vote share (%)")
    ax.set_title(f"Test Predictions  R2={test_metrics['r2']:.3f}")
    ax.legend()

    plt.tight_layout()
    out_path = save_dir / "results.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MLP on Swiss referendum data")
    p.add_argument("--hidden", type=int, nargs="+", default=[50, 50, 50, 50],
                   metavar="N", help="Hidden layer sizes (default: 50 50 50 50)")
    p.add_argument("--activation", default="relu",
                   choices=["relu", "leaky_relu", "sigmoid", "tanh"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--norm", default="none", choices=["none", "batch", "layer"])
    p.add_argument("--optimizer", default="adam",
                   choices=["adam", "rmsprop", "sgd", "adagrad"])
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss", default="mse", choices=["mse", "huber", "mae"])
    p.add_argument("--no_schedule", action="store_true",
                   help="Disable ReduceLROnPlateau scheduler")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        hidden_layers=tuple(args.hidden),
        activation=args.activation,
        dropout=args.dropout,
        norm=args.norm,
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        grad_clip=args.grad_clip,
        loss_fn=args.loss,
        lr_schedule=not args.no_schedule,
        seed=args.seed,
    )
