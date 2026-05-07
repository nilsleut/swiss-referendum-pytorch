from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

ActivationType = Literal["relu", "leaky_relu", "sigmoid", "tanh"]
NormType = Literal["none", "batch", "layer"]


class _ResidualBlock(nn.Module):
    """
    Pre-activation residual block for tabular data.
    Follows the pattern: Norm -> Act -> Linear -> Norm -> Act -> Linear,
    with a learned projection on the skip path when dimensions change.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: nn.Module,
        norm: NormType,
        dropout: float,
    ) -> None:
        super().__init__()

        def make_norm(d: int) -> nn.Module:
            if norm == "batch":
                return nn.BatchNorm1d(d)
            if norm == "layer":
                return nn.LayerNorm(d)
            return nn.Identity()

        self.block = nn.Sequential(
            make_norm(in_dim),
            activation,
            nn.Linear(in_dim, out_dim),
            make_norm(out_dim),
            activation,
            nn.Linear(out_dim, out_dim),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class MLP(nn.Module):
    """
    Fully-connected MLP for regression on the Swiss referendum dataset.

    Compared to the original KNN.py this adds:
    - Optional residual connections (resnet=True)
    - Optional BatchNorm1d on the raw input (input_norm=True)
    - leaky_relu activation
    - batch / layer normalization between hidden layers
    - He / Xavier weight init matched to activation
    - Linear output layer (no final activation) — same as original
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_layers: tuple[int, ...] = (50, 50, 50, 50),
        activation: ActivationType = "relu",
        dropout: float = 0.0,
        norm: NormType = "none",
        resnet: bool = False,
        input_norm: bool = False,
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

        # Input normalization — helps stabilize training on mixed sparse/dense features
        self.input_bn = nn.BatchNorm1d(n_features) if input_norm else nn.Identity()

        if resnet:
            self._build_resnet(n_features, n_outputs, hidden_layers, act_fn[activation], norm, dropout)
        else:
            self._build_plain(n_features, n_outputs, hidden_layers, act_fn[activation], norm, dropout)

        self._init_weights(activation)

    def _build_plain(self, n_features, n_outputs, hidden_layers, act, norm, dropout):
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
                layers.append(act)
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def _build_resnet(self, n_features, n_outputs, hidden_layers, act, norm, dropout):
        # First projection: input -> first hidden dim
        layers: list[nn.Module] = [nn.Linear(n_features, hidden_layers[0])]
        # Residual blocks for pairs of hidden layers
        sizes = list(hidden_layers)
        for i in range(len(sizes) - 1):
            layers.append(_ResidualBlock(sizes[i], sizes[i + 1], act, norm, dropout))
        # Output head
        layers.append(nn.Linear(sizes[-1], n_outputs))
        self.net = nn.Sequential(*layers)

    def _init_weights(self, activation: str) -> None:
        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue
            if activation in ("relu", "leaky_relu"):
                nn.init.kaiming_normal_(m.weight, nonlinearity=activation.replace("_", ""))
            else:
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_bn(x))
