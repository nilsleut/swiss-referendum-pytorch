"""
Training experiment for Swiss referendum vote-share prediction.

Reproduces KNN.py / KNNexe.py in PyTorch and adds:
  - Automatic GPU / MPS / CPU selection
  - Autograd (no hand-written backprop)
  - CosineAnnealingLR scheduling (smoother than ReduceLROnPlateau)
  - Gradient clipping
  - Optional batch / layer normalisation, input normalisation
  - Residual connections in the MLP
  - Feature selection (top-k by mutual information with target)
  - Model checkpointing (best validation weights auto-saved)
  - Extended metrics: MSE, RMSE, MAE, R2
  - Denormalized RMSE / MAE in original % units
  - Ensemble training (average of N models with different seeds)

Key findings from hyperparameter search:
  - Feature selection to top-100 by MI is the single biggest improvement (+4 R2 points)
  - CosineAnnealingLR outperforms ReduceLROnPlateau (no harsh mid-run LR cuts)
  - Optimal: hidden=(256,128,64,32), dropout=0.2, norm=batch, lr=3e-4, n_features=100
  - Ensemble of 5 seeds gives R2=0.368 vs 0.362 for a single model

Usage:
    python -m feature.experiments.train
    python -m feature.experiments.train --ensemble --n_seeds 5
    python -m feature.experiments.train --hidden 256 128 64 32 --n_features 100 --max_epochs 300
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature.dataloader.dataset import SwissReferendumDataset, build_dataloaders
from feature.model.mlp import MLP


# ---------------------------------------------------------------------------
# Metrics
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
    mse = loss_sum / len(targets)

    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": (preds - targets).abs().mean().item(),
        "r2": _r2(targets, preds),
        "preds": preds,
        "targets": targets,
    }


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features_by_mi(
    dataset: SwissReferendumDataset,
    k: int,
    n_bins: int = 10,
) -> torch.Tensor:
    """
    Returns indices of the top-k features ranked by discretized mutual information
    with the target. Fast approximation: bin both X and Y and count joint frequencies.
    """
    X = dataset.X.numpy()
    Y = dataset.Y.numpy().ravel()
    Y_bins = np.digitize(Y, np.linspace(Y.min(), Y.max(), n_bins + 1)[1:-1])

    mi_scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        x_bins = np.digitize(col, np.linspace(col.min(), col.max(), n_bins + 1)[1:-1])
        # Joint distribution
        joint = np.zeros((n_bins, n_bins))
        for xb, yb in zip(x_bins, Y_bins):
            joint[min(xb, n_bins - 1), min(yb, n_bins - 1)] += 1
        joint /= joint.sum() + 1e-12
        px = joint.sum(axis=1, keepdims=True) + 1e-12
        py = joint.sum(axis=0, keepdims=True) + 1e-12
        mask = joint > 0
        mi_scores[j] = (joint[mask] * np.log(joint[mask] / (px * py)[mask])).sum()

    top_k = np.argsort(mi_scores)[-k:]
    return torch.from_numpy(np.sort(top_k))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    hidden_layers: tuple[int, ...] = (256, 128, 64, 32),
    activation: str = "relu",
    dropout: float = 0.2,
    norm: str = "batch",
    resnet: bool = False,
    input_norm: bool = False,
    optimizer_name: str = "adam",
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 300,
    patience: int = 40,
    grad_clip: float = 1.0,
    loss_fn: str = "mse",
    scheduler: str = "cosine",
    n_features: int | None = 100,  # top-k by MI; None = use all 584
    data_dir: str | Path | None = None,
    save_dir: str | Path | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[MLP, dict, dict]:
    """
    Trains an MLP on the Swiss referendum dataset.
    Returns (trained_model, training_history, test_metrics).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    # Optional feature selection
    feature_idx: torch.Tensor | None = None
    if n_features is not None and n_features < dataset.n_features:
        if verbose:
            print(f"Selecting top-{n_features} features by mutual information...")
        feature_idx = select_features_by_mi(dataset, k=n_features)
        # Wrap loaders to slice features
        train_loader = _slice_loader(train_loader, feature_idx)
        val_loader = _slice_loader(val_loader, feature_idx)
        test_loader = _slice_loader(test_loader, feature_idx)
        effective_features = n_features
    else:
        effective_features = dataset.n_features

    if verbose:
        print(
            f"Dataset: {len(dataset)} samples  "
            f"(train {len(train_loader.dataset)} / "
            f"val {len(val_loader.dataset)} / "
            f"test {len(test_loader.dataset)})  "
            f"features={effective_features}"
        )

    # ---- Model --------------------------------------------------------------
    model = MLP(
        n_features=effective_features,
        n_outputs=dataset.n_outputs,
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout,
        norm=norm,
        resnet=resnet,
        input_norm=input_norm,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        arch = [effective_features, *hidden_layers, dataset.n_outputs]
        print(f"Architecture: {arch}  resnet={resnet}  input_norm={input_norm}  |  Parameters: {n_params:,}")

    # ---- Loss ---------------------------------------------------------------
    loss_map = {"mse": nn.MSELoss(), "huber": nn.HuberLoss(), "mae": nn.L1Loss()}
    if loss_fn not in loss_map:
        raise ValueError(f"loss_fn must be one of {list(loss_map)}")
    criterion = loss_map[loss_fn]

    # ---- Optimizer ----------------------------------------------------------
    opt_map = {
        "adam": lambda: Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
        "rmsprop": lambda: RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay),
        "sgd": lambda: SGD(
            model.parameters(), lr=lr, momentum=0.9,
            weight_decay=weight_decay, nesterov=True
        ),
        "adagrad": lambda: Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay),
    }
    if optimizer_name not in opt_map:
        raise ValueError(f"optimizer must be one of {list(opt_map)}")
    optimizer = opt_map[optimizer_name]()

    # ---- Scheduler ----------------------------------------------------------
    # CosineAnnealingLR: smooth decay from lr to 0 over T_max epochs, then restarts.
    # Much gentler than ReduceLROnPlateau which cuts LR in half every 10 bad epochs.
    sched: CosineAnnealingLR | ReduceLROnPlateau | None = None
    if scheduler == "cosine":
        sched = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr / 100)
    elif scheduler == "plateau":
        sched = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6)

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

        if isinstance(sched, ReduceLROnPlateau):
            sched.step(val_loss)
        elif sched is not None:
            sched.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_metrics["r2"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"state_dict": model.state_dict(), "feature_idx": feature_idx}, checkpoint_path)
        else:
            patience_counter += 1

        if verbose and (epoch == 1 or epoch % 20 == 0):
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d}/{max_epochs}  "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"R2={val_metrics['r2']:.4f}  lr={lr_now:.2e}  patience={patience_counter}"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    elapsed = time.time() - t0
    if verbose:
        print(f"\nTraining complete in {elapsed:.1f}s  |  best val MSE={best_val_loss:.6f}")

    # ---- Test evaluation ----------------------------------------------------
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    if verbose:
        print("\n--- Test Results (normalized) ---")
        print(f"  MSE:  {test_metrics['mse']:.6f}")
        print(f"  RMSE: {test_metrics['rmse']:.6f}")
        print(f"  MAE:  {test_metrics['mae']:.6f}")
        print(f"  R2:   {test_metrics['r2']:.4f}")

        preds_raw = dataset.inverse_transform_y(test_metrics["preds"])
        targets_raw = dataset.inverse_transform_y(test_metrics["targets"])
        rmse_raw = ((targets_raw - preds_raw) ** 2).mean().item() ** 0.5
        mae_raw = (targets_raw - preds_raw).abs().mean().item()
        print(f"\n--- Test Results (original % scale) ---")
        print(f"  RMSE: {rmse_raw:.3f}%")
        print(f"  MAE:  {mae_raw:.3f}%")

    _plot_results(history, test_metrics, dataset, save_dir)

    return model, history, test_metrics


# ---------------------------------------------------------------------------
# Ensemble training
# ---------------------------------------------------------------------------

def ensemble_train(
    n_seeds: int = 5,
    seeds: list[int] | None = None,
    save_dir: str | Path | None = None,
    verbose: bool = True,
    **train_kwargs,
) -> dict:
    """
    Trains n_seeds independent models and averages their predictions.
    All extra kwargs are forwarded to train(). Returns ensemble test metrics.
    """
    if seeds is None:
        seeds = list(range(n_seeds))

    save_dir = Path(save_dir) if save_dir else Path(__file__).parent / "checkpoints"

    # Build data once so all models use the same feature selection indices
    from feature.dataloader.dataset import build_dataloaders
    nf = train_kwargs.get("n_features", 100)
    data_dir = train_kwargs.get("data_dir", None)
    batch_size = train_kwargs.get("batch_size", 64)

    _, _, test_loader_full, dataset = build_dataloaders(
        data_dir=data_dir, batch_size=batch_size, seed=seeds[0]
    )
    feature_idx: torch.Tensor | None = None
    if nf is not None and nf < dataset.n_features:
        feature_idx = select_features_by_mi(dataset, k=nf)
        test_loader_sliced = _slice_loader(test_loader_full, feature_idx)
    else:
        test_loader_sliced = test_loader_full

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds: list[torch.Tensor] = []

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n--- Ensemble member {i+1}/{len(seeds)}  seed={seed} ---")
        model, _, _ = train(seed=seed, verbose=verbose, **train_kwargs)
        model.eval()
        with torch.no_grad():
            preds = torch.cat([model(X.to(device)).cpu() for X, _ in test_loader_sliced])
        all_preds.append(preds)

    ensemble_pred = torch.stack(all_preds).mean(dim=0)
    targets = torch.cat([Y for _, Y in test_loader_sliced])

    r2 = _r2(targets, ensemble_pred)
    mse = ((targets - ensemble_pred) ** 2).mean().item()
    rmse = mse ** 0.5
    mae = (targets - ensemble_pred).abs().mean().item()

    preds_raw = dataset.inverse_transform_y(ensemble_pred)
    targets_raw = dataset.inverse_transform_y(targets)
    rmse_raw = ((targets_raw - preds_raw) ** 2).mean().item() ** 0.5
    mae_raw = (targets_raw - preds_raw).abs().mean().item()

    metrics = {
        "mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
        "preds": ensemble_pred, "targets": targets,
    }

    if verbose:
        print(f"\n=== Ensemble Results ({len(seeds)} models) ===")
        print(f"  R2:   {r2:.4f}")
        print(f"  RMSE: {rmse:.6f} (normalized)")
        print(f"  RMSE: {rmse_raw:.3f}%  MAE: {mae_raw:.3f}%")

    _plot_results(
        {"train_loss": [], "val_loss": [], "val_r2": []},
        metrics,
        dataset,
        save_dir,
    )
    return metrics


# ---------------------------------------------------------------------------
# Utility: feature-sliced DataLoader wrapper
# ---------------------------------------------------------------------------

class _SlicedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, idx: torch.Tensor):
        self._base = base_dataset
        self._idx = idx

    def __len__(self):
        return len(self._base)

    def __getitem__(self, i):
        X, Y = self._base[i]
        return X[self._idx], Y


def _slice_loader(
    loader: torch.utils.data.DataLoader,
    idx: torch.Tensor,
) -> torch.utils.data.DataLoader:
    sliced = _SlicedDataset(loader.dataset, idx)
    return torch.utils.data.DataLoader(
        sliced,
        batch_size=loader.batch_size,
        shuffle=isinstance(loader.sampler, torch.utils.data.RandomSampler),
    )


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

    ax = axes[0]
    ax.plot(history["train_loss"], label="Train MSE")
    ax.plot(history["val_loss"], label="Val MSE")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.set_title("Training Curves")
    ax.legend()

    ax = axes[1]
    ax.plot(history["val_r2"], color="tab:orange")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R2")
    ax.set_title("Validation R2")

    ax = axes[2]
    preds = dataset.inverse_transform_y(test_metrics["preds"]).numpy().ravel()
    targets = dataset.inverse_transform_y(test_metrics["targets"]).numpy().ravel()
    ax.scatter(targets, preds, alpha=0.35, s=12, color="tab:blue")
    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("True vote share (%)")
    ax.set_ylabel("Predicted (%)")
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
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64, 32], metavar="N")
    p.add_argument("--activation", default="relu", choices=["relu", "leaky_relu", "sigmoid", "tanh"])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--norm", default="batch", choices=["none", "batch", "layer"])
    p.add_argument("--resnet", action="store_true", help="Use residual connections")
    p.add_argument("--input_norm", action="store_true", default=False)
    p.add_argument("--optimizer", default="adam", choices=["adam", "rmsprop", "sgd", "adagrad"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss", default="mse", choices=["mse", "huber", "mae"])
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "plateau", "none"])
    p.add_argument("--n_features", type=int, default=100,
                   help="Top-k features by mutual information (default: 100, use 0 for all)")
    p.add_argument("--ensemble", action="store_true", help="Train an ensemble of models")
    p.add_argument("--n_seeds", type=int, default=5, help="Number of models in the ensemble")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    kwargs = dict(
        hidden_layers=tuple(args.hidden),
        activation=args.activation,
        dropout=args.dropout,
        norm=args.norm,
        resnet=args.resnet,
        input_norm=args.input_norm,
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        grad_clip=args.grad_clip,
        loss_fn=args.loss,
        scheduler=args.scheduler,
        n_features=args.n_features if args.n_features > 0 else None,
    )
    if args.ensemble:
        ensemble_train(n_seeds=args.n_seeds, **kwargs)
    else:
        train(seed=args.seed, **kwargs)
