from __future__ import annotations

from typing import Any

import torch
from torch import nn


class BiddingPolicyNetwork(nn.Module):
    """Feedforward ReLU policy network for one physical generator."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if any(width <= 0 for width in hidden_layers):
            raise ValueError("All hidden layer widths must be positive.")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = list(hidden_layers)
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def export_layers_to_json(self) -> list[dict[str, Any]]:
        """Return Linear/ReLU layers in a Gurobi-friendly JSON structure."""
        exported_layers: list[dict[str, Any]] = []
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                exported_layers.append(
                    {
                        "type": "linear",
                        "weight": layer.weight.detach().cpu().tolist(),
                        "bias": layer.bias.detach().cpu().tolist(),
                    }
                )
            elif isinstance(layer, nn.ReLU):
                exported_layers.append({"type": "relu"})
            else:
                raise TypeError(f"Unsupported layer type for export: {type(layer).__name__}")
        return exported_layers
