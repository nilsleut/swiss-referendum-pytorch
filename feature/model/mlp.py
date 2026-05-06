from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

ActivationType = Literal["relu", "leaky_relu", "sigmoid", "tanh"]
NormType = Literal["none", "batch", "layer"]


class MLP(nn.Module):
    """
    Fully-connected MLP for regression on the Swiss referendum dataset.

    PyTorch re-implementation of KNN.py with these additions:
    - leaky_relu activation
    - optional batch / layer normalization between hidden layers
    - He / Xavier weight init matched to activation (same logic as original)
    - linear output layer (no final activation) — same as original
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_layers: tuple[int, ...] = (50, 50, 50, 50),
        activation: ActivationType = "relu",
        dropout: float = 0.0,
        norm: NormType = "none",
    ) -> None:
        super().__init__()

        act_fn: dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.1),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }
        if activation not in act_fn:
            raise ValueError(f"activation must be one of {list(act_fn)}, got '{activation}'")

        layer_sizes = [n_features, *hidden_layers, n_outputs]
        layers: list[nn.Module] = []

        for i in range(len(layer_sizes) - 1):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
            is_output = i == len(layer_sizes) - 2

            layers.append(nn.Linear(in_dim, out_dim))

            if not is_output:
                if norm == "batch":
                    layers.append(nn.BatchNorm1d(out_dim))
                elif norm == "layer":
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(act_fn[activation])
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))

        self.net = nn.Sequential(*layers)
        self._init_weights(activation)

    def _init_weights(self, activation: str) -> None:
        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue
            if activation in ("relu", "leaky_relu"):
                # He initialization — matches the original's np.sqrt(2 / input_size)
                nn.init.kaiming_normal_(m.weight, nonlinearity=activation.replace("_", ""))
            else:
                # Xavier initialization — matches the original's np.sqrt(1 / input_size)
                nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
