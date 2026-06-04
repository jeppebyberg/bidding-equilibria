from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from torch import nn

from models.neural_network.training.dataset import (
    BiddingPolicyData,
    load_generator_policy_data,
)
from models.neural_network.training.model import BiddingPolicyNetwork

@dataclass(frozen=True)
class BiddingPolicyTrainingConfig:
    hidden_layers: list[int]
    learning_rate: float
    batch_size: int
    num_epochs: int
    weight_decay: float
    val_size: float   # fraction of data for early-stopping validation
    test_size: float  # fraction of data for held-out evaluation
    random_state: int
    patience: int | None
    min_delta: float
    device: str | None = None
    final_activation: str = "linear"
    # ReduceLROnPlateau scheduler on the validation loss. Set
    # use_lr_scheduler=False to keep a constant learning rate.
    use_lr_scheduler: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 20
    lr_scheduler_min_lr: float = 1e-6

def train_generator_policy(
    csv_path: str | Path,
    model_dir: str | Path,
    result_dir: str | Path,
    config: BiddingPolicyTrainingConfig,
) -> dict[str, Any]:
    """Train, save, and export a bidding policy network for one generator."""
    policy_data = load_generator_policy_data(
        csv_path=csv_path,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state,
        batch_size=config.batch_size,
        shuffle_train=True,
    )
    if policy_data.output_dim != len(policy_data.target_columns):
        raise ValueError(
            "output_dim must equal the number of target columns; "
            f"got {policy_data.output_dim} and {len(policy_data.target_columns)}."
        )

    selected_device = torch.device(
        config.device if config.device is not None else _default_device()
    )
    model = BiddingPolicyNetwork(
        input_dim=policy_data.input_dim,
        output_dim=policy_data.output_dim,
        hidden_layers=config.hidden_layers,
        final_activation=config.final_activation,
    ).to(selected_device)
    _initialize_final_relu_bias_from_targets(model, policy_data, selected_device)

    print(
        f"Training model for {policy_data.generator_name} on {selected_device} with "
        f"hidden layers {config.hidden_layers}, learning rate {config.learning_rate}, "
        f"batch size {config.batch_size}, and {config.num_epochs} epochs."
    )

    history = _fit_model(model, policy_data, config, selected_device)

    model_dir_path = Path(model_dir)
    result_dir_path = Path(result_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    result_dir_path.mkdir(parents=True, exist_ok=True)

    generator_name = policy_data.generator_name
    model_path = model_dir_path / f"{generator_name}_policy.pt"
    metadata_path = model_dir_path / f"{generator_name}_policy_metadata.json"
    weights_path = model_dir_path / f"{generator_name}_policy_weights.json"
    history_path = result_dir_path / f"{generator_name}_training_history.json"

    torch.save(model.state_dict(), model_path)
    _write_json(history_path, history)
    _write_json(
        metadata_path,
        _build_metadata(
            policy_data=policy_data,
            config=config,
            history=history,
        ),
    )
    _write_json(
        weights_path,
        _build_weight_export(
            model=model,
            policy_data=policy_data,
        ),
    )

    summary_entry = {
        "generator_name": generator_name,
        "input_dim": policy_data.input_dim,
        "output_dim": policy_data.output_dim,
        "number_of_target_bidding_blocks": len(policy_data.target_columns),
        "train_size": policy_data.train_size,
        "val_size": policy_data.val_size,
        "test_size": policy_data.test_size,
        "best_val_loss": history["best_val_loss"],
        "final_test_loss": history["final_test_loss"],
        "best_epoch": history["best_epoch"],
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "history_path": str(history_path),
    }
    return {
        "summary": summary_entry,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "weights_path": weights_path,
        "history_path": history_path,
        "policy_data": policy_data,
        "history": history,
    }

def _fit_model(
    model: BiddingPolicyNetwork,
    policy_data: BiddingPolicyData,
    config: BiddingPolicyTrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
            min_lr=config.lr_scheduler_min_lr,
        )

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    lr_history: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.num_epochs + 1):
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{config.num_epochs}...")
        train_loss = _run_epoch(
            model=model,
            data_loader=policy_data.train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
        )
        val_loss = _evaluate(
            model=model,
            data_loader=policy_data.val_loader,
            loss_function=loss_function,
            device=device,
        )
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Record the LR that produced this epoch, then let the scheduler react.
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)
        if scheduler is not None:
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < current_lr:
                print(f"  LR reduced: {current_lr:.3g} -> {new_lr:.3g} at epoch {epoch}")

        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            config.patience is not None
            and epochs_without_improvement >= config.patience
        ):
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Evaluate once on the true held-out test set after checkpoint selection.
    final_test_loss = _evaluate(
        model=model,
        data_loader=policy_data.test_loader,
        loss_function=loss_function,
        device=device,
    )

    return {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "lr": lr_history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_test_loss": final_test_loss,
    }

def _initialize_final_relu_bias_from_targets(
    model: BiddingPolicyNetwork,
    policy_data: BiddingPolicyData,
    device: torch.device,
) -> None:
    if getattr(model, "final_activation", None) != "relu":
        return

    final_linear = None
    for layer in reversed(model.network):
        if isinstance(layer, nn.Linear):
            final_linear = layer
            break
    if final_linear is None:
        return

    targets = policy_data.train_loader.dataset.tensors[1].to(device)
    target_means = targets.mean(dim=0).clamp_min(1e-6)
    with torch.no_grad():
        final_linear.bias.copy_(target_means)

def _run_epoch(
    model: BiddingPolicyNetwork,
    data_loader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = features.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / total_samples

def _evaluate(
    model: BiddingPolicyNetwork,
    data_loader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            loss = loss_function(predictions, targets)

            batch_size = features.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / total_samples

def _build_metadata(
    policy_data: BiddingPolicyData,
    config: BiddingPolicyTrainingConfig,
    history: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generator_name": policy_data.generator_name,
        "input_dim": policy_data.input_dim,
        "output_dim": policy_data.output_dim,
        "feature_columns": policy_data.feature_columns,
        "target_columns": policy_data.target_columns,
        "hidden_layers": config.hidden_layers,
        "activation": "relu",
        "final_activation": config.final_activation,
        "train_size": policy_data.train_size,
        "val_size": policy_data.val_size,
        "val_fraction": config.val_size,
        "test_size": policy_data.test_size,
        "test_fraction": config.test_size,
        "num_epochs_trained": len(history["train_loss"]),
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],
        "final_test_loss": history["final_test_loss"],
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "weight_decay": config.weight_decay,
        "random_state": config.random_state,
        "use_lr_scheduler": config.use_lr_scheduler,
        "lr_scheduler_factor": config.lr_scheduler_factor,
        "lr_scheduler_patience": config.lr_scheduler_patience,
        "lr_scheduler_min_lr": config.lr_scheduler_min_lr,
        "final_learning_rate": history["lr"][-1] if history.get("lr") else config.learning_rate,
    }

def _build_weight_export(
    model: BiddingPolicyNetwork,
    policy_data: BiddingPolicyData,
) -> dict[str, Any]:
    return {
        "generator_name": policy_data.generator_name,
        "activation": "relu",
        "final_activation": model.final_activation,
        "layers": model.export_layers_to_json(),
        "feature_columns": policy_data.feature_columns,
        "target_columns": policy_data.target_columns,
    }

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)

def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
