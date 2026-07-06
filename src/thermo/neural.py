from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class NeuralConfig:
    n_train: int
    width: int
    lr: float
    weight_decay: float
    seed: int
    max_steps: int = 180
    batch_size: int = 128


def checkpoint_steps(max_steps: int) -> tuple[int, ...]:
    raw = [max(1, round(max_steps * frac)) for frac in (0.33, 0.67, 1.0)]
    raw.extend([40, 90, 180])
    return tuple(sorted({step for step in raw if step <= max_steps}))


class MLP(nn.Module):
    def __init__(self, input_dim: int, width: int, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def l2_norm_sq(model: nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for param in model.parameters():
            total += float(torch.sum(param.detach() ** 2).cpu())
    return total


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 2048,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            logits = model(xb)
            losses.append(float(loss_fn(logits, yb).detach().cpu()))
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().detach().cpu())
            total += int(yb.numel())
    return sum(losses) / total, correct / total


def train_mlp_config(
    config: NeuralConfig,
    arrays: dict[str, np.ndarray | str],
    device: str | torch.device,
) -> tuple[list[dict[str, float | int | str]], MLP]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(device)

    x_train_np = arrays["x_train"]
    y_train_np = arrays["y_train"]
    x_val_np = arrays["x_val"]
    y_val_np = arrays["y_val"]
    assert isinstance(x_train_np, np.ndarray)
    assert isinstance(y_train_np, np.ndarray)
    assert isinstance(x_val_np, np.ndarray)
    assert isinstance(y_val_np, np.ndarray)

    x_train = torch.from_numpy(x_train_np[: config.n_train]).to(device)
    y_train = torch.from_numpy(y_train_np[: config.n_train]).to(device)
    x_val = torch.from_numpy(x_val_np).to(device)
    y_val = torch.from_numpy(y_val_np).to(device)

    model = MLP(input_dim=x_train.shape[1], width=config.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    records: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    checkpoints = checkpoint_steps(config.max_steps)

    for step in range(1, config.max_steps + 1):
        model.train()
        idx = torch.randint(0, config.n_train, (config.batch_size,), device=device)
        logits = model(x_train[idx])
        loss = loss_fn(logits, y_train[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step in checkpoints:
            train_loss, train_acc = evaluate(model, x_train, y_train)
            val_loss, val_acc = evaluate(model, x_val, y_val)
            norm_sq = l2_norm_sq(model)
            reg_loss = 0.5 * config.weight_decay * norm_sq
            records.append(
                {
                    "family": "mnist_mlp",
                    "strategy": "scratch",
                    "seed": config.seed,
                    "n_train": config.n_train,
                    "width": config.width,
                    "params": parameter_count(model),
                    "lr": config.lr,
                    "weight_decay": config.weight_decay,
                    "step": step,
                    "epochs": step * config.batch_size / config.n_train,
                    "examples_seen": step * config.batch_size,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "l2_norm_sq": norm_sq,
                    "regularizer_loss": reg_loss,
                    "train_total_loss_reported": train_loss + reg_loss,
                    "val_total_loss_reported": val_loss + reg_loss,
                    "elapsed_s": time.perf_counter() - started,
                }
            )

    return records, model


def _train_existing(
    model: MLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float,
    weight_decay: float,
    seed: int,
    steps: int,
    batch_size: int,
) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    n_train = x_train.shape[0]
    for _ in range(steps):
        model.train()
        idx = torch.randint(0, n_train, (batch_size,), device=x_train.device)
        loss = loss_fn(model(x_train[idx]), y_train[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def graft_width(old: MLP, new_width: int, seed: int) -> MLP:
    torch.manual_seed(seed)
    first_old: nn.Linear = old.net[0]  # type: ignore[assignment]
    second_old: nn.Linear = old.net[2]  # type: ignore[assignment]
    new = MLP(first_old.in_features, new_width, second_old.out_features).to(first_old.weight.device)
    first_new: nn.Linear = new.net[0]  # type: ignore[assignment]
    second_new: nn.Linear = new.net[2]  # type: ignore[assignment]
    old_width = first_old.out_features
    with torch.no_grad():
        first_new.weight[:old_width].copy_(first_old.weight)
        first_new.bias[:old_width].copy_(first_old.bias)
        second_new.weight[:, :old_width].copy_(second_old.weight)
        second_new.bias.copy_(second_old.bias)
    return new


def run_hysteresis_probe(
    arrays: dict[str, np.ndarray | str],
    device: str | torch.device,
    n_train: int = 2048,
    lr: float = 0.004,
    weight_decay: float = 1e-3,
    seed: int = 100,
    half_steps: int = 90,
    batch_size: int = 128,
) -> list[dict[str, float | int | str]]:
    device = torch.device(device)
    x_train_np = arrays["x_train"]
    y_train_np = arrays["y_train"]
    x_val_np = arrays["x_val"]
    y_val_np = arrays["y_val"]
    assert isinstance(x_train_np, np.ndarray)
    assert isinstance(y_train_np, np.ndarray)
    assert isinstance(x_val_np, np.ndarray)
    assert isinstance(y_val_np, np.ndarray)

    x_train = torch.from_numpy(x_train_np[:n_train]).to(device)
    y_train = torch.from_numpy(y_train_np[:n_train]).to(device)
    x_val = torch.from_numpy(x_val_np).to(device)
    y_val = torch.from_numpy(y_val_np).to(device)

    records: list[dict[str, float | int | str]] = []

    for strategy, width, first_steps, second_width in [
        ("scratch_64", 64, 2 * half_steps, None),
        ("scratch_128", 128, 2 * half_steps, None),
        ("graft_64_to_128", 64, half_steps, 128),
    ]:
        torch.manual_seed(seed)
        model = MLP(x_train.shape[1], width).to(device)
        started = time.perf_counter()
        _train_existing(
            model, x_train, y_train, x_val, y_val, lr, weight_decay, seed, first_steps, batch_size
        )
        if second_width is not None:
            model = graft_width(model, second_width, seed + 1)
            _train_existing(
                model,
                x_train,
                y_train,
                x_val,
                y_val,
                lr,
                weight_decay,
                seed + 2,
                half_steps,
                batch_size,
            )
            final_width = second_width
        else:
            final_width = width
        train_loss, train_acc = evaluate(model, x_train, y_train)
        val_loss, val_acc = evaluate(model, x_val, y_val)
        norm_sq = l2_norm_sq(model)
        reg_loss = 0.5 * weight_decay * norm_sq
        records.append(
            {
                "family": "mnist_mlp",
                "strategy": strategy,
                "seed": seed,
                "n_train": n_train,
                "width": final_width,
                "params": parameter_count(model),
                "lr": lr,
                "weight_decay": weight_decay,
                "step": 2 * half_steps,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "l2_norm_sq": norm_sq,
                "regularizer_loss": reg_loss,
                "train_total_loss_reported": train_loss + reg_loss,
                "val_total_loss_reported": val_loss + reg_loss,
                "elapsed_s": time.perf_counter() - started,
            }
        )

    return records


def default_neural_configs(seed: int = 0) -> list[NeuralConfig]:
    configs: list[NeuralConfig] = []
    for n_train in (512, 2048, 8192):
        for width in (16, 64, 128):
            for lr in (0.001, 0.004):
                for weight_decay in (0.0, 1e-3):
                    configs.append(
                        NeuralConfig(
                            n_train=n_train,
                            width=width,
                            lr=lr,
                            weight_decay=weight_decay,
                            seed=seed,
                        )
                    )
    return configs


def dense_local_neural_configs(seed: int = 0) -> list[NeuralConfig]:
    """Clustered, log-spaced retrain grid for local susceptibility curves.

    This is deliberately still a local pilot, not the 1e5-run remote grid. It
    puts more resolution where finite differences are estimated while staying
    cheap enough to rerun during report iteration.
    """

    n_trains = (8192, 11585, 16384, 23170, 32768, 46341, 50000)
    widths = (32, 45, 64, 91, 128, 181, 256)
    weight_decays = (
        0.0,
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
    )
    lr = 0.003
    batch_size = 256
    epochs = 3.0

    configs: list[NeuralConfig] = []
    for n_train in n_trains:
        max_steps = max(1, round(epochs * n_train / batch_size))
        for width in widths:
            for weight_decay in weight_decays:
                configs.append(
                    NeuralConfig(
                        n_train=n_train,
                        width=width,
                        lr=lr,
                        weight_decay=weight_decay,
                        seed=seed,
                        max_steps=max_steps,
                        batch_size=batch_size,
                    )
                )
    return configs
