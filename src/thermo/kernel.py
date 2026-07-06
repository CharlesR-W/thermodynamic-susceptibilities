from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KernelConfig:
    target_kind: str
    target_param: float
    smoothness: float
    bandwidth: float
    ridge: float
    n_train: int
    seed: int = 0
    kmax: int = 96
    n_val: int = 2048
    noise: float = 0.05


def fourier_features(x: np.ndarray, kmax: int) -> np.ndarray:
    x = x.reshape(-1, 1)
    k = np.arange(1, kmax + 1, dtype=np.float64).reshape(1, -1)
    angles = 2.0 * np.pi * x * k
    return np.sqrt(2.0) * np.concatenate([np.cos(angles), np.sin(angles)], axis=1)


def target_amplitudes(kind: str, param: float, kmax: int) -> np.ndarray:
    k = np.arange(1, kmax + 1, dtype=np.float64)
    if kind == "power_law":
        amp = k ** (-param)
    elif kind == "exponential":
        amp = np.exp(-k / param)
    else:
        raise ValueError(f"Unknown target kind {kind}")
    return amp / np.sqrt(np.sum(amp**2))


def kernel_weights(smoothness: float, bandwidth: float, kmax: int) -> np.ndarray:
    k = np.arange(1, kmax + 1, dtype=np.float64)
    if np.isinf(smoothness):
        q = np.exp(-((k / bandwidth) ** 2))
    else:
        q = (1.0 + (k / bandwidth) ** 2) ** (-smoothness)
    q = np.maximum(q, 1e-12)
    return q / q.max()


def make_fourier_task(config: KernelConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    x_train = rng.uniform(0.0, 1.0, size=max(1024, config.n_train))
    x_val = rng.uniform(0.0, 1.0, size=config.n_val)

    amp = target_amplitudes(config.target_kind, config.target_param, config.kmax)
    coeff_cos = rng.normal(size=config.kmax) * amp
    coeff_sin = rng.normal(size=config.kmax) * amp

    phi_train = fourier_features(x_train, config.kmax)
    phi_val = fourier_features(x_val, config.kmax)
    coeff = np.concatenate([coeff_cos, coeff_sin])
    y_train_clean = phi_train @ coeff
    y_val_clean = phi_val @ coeff

    scale = np.std(y_train_clean)
    y_train_clean = y_train_clean / scale
    y_val_clean = y_val_clean / scale
    y_train = y_train_clean + config.noise * rng.normal(size=y_train_clean.shape)

    target_power = coeff_cos**2 + coeff_sin**2
    target_power = target_power / target_power.sum()

    return {
        "x_train": x_train,
        "x_val": x_val,
        "y_train": y_train,
        "y_val": y_val_clean,
        "target_power": target_power,
    }


def spectral_alignment(target_power: np.ndarray, q: np.ndarray) -> float:
    a = target_power / np.linalg.norm(target_power)
    b = q / np.linalg.norm(q)
    return float(np.dot(a, b))


def kernel_ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    q: np.ndarray,
    ridge: float,
    kmax: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    phi_train = fourier_features(x_train, kmax)
    phi_val = fourier_features(x_val, kmax)
    q_feat = np.concatenate([q, q])
    z_train = phi_train * np.sqrt(q_feat.reshape(1, -1))
    z_val = phi_val * np.sqrt(q_feat.reshape(1, -1))
    n = z_train.shape[0]
    gram = z_train.T @ z_train
    mat = gram + (n * ridge) * np.eye(gram.shape[0])
    rhs = z_train.T @ y_train
    try:
        beta = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(mat, rhs, rcond=None)[0]
    pred_train = z_train @ beta
    pred_val = z_val @ beta
    eigvals = np.linalg.eigvalsh(gram)
    effective_dim = float(np.sum(eigvals / (eigvals + n * ridge)))
    return pred_train, pred_val, effective_dim


def run_kernel_config(config: KernelConfig) -> dict[str, float | int | str]:
    task = make_fourier_task(config)
    n = config.n_train
    q = kernel_weights(config.smoothness, config.bandwidth, config.kmax)
    pred_train, pred_val, effective_dim = kernel_ridge_fit_predict(
        task["x_train"][:n],
        task["y_train"][:n],
        task["x_val"],
        q,
        config.ridge,
        config.kmax,
    )
    train_mse = float(np.mean((pred_train - task["y_train"][:n]) ** 2))
    val_mse = float(np.mean((pred_val - task["y_val"]) ** 2))
    smooth_label = "rbf" if np.isinf(config.smoothness) else f"nu={config.smoothness:g}"
    return {
        "family": "fourier_kernel",
        "target_kind": config.target_kind,
        "target_param": config.target_param,
        "smoothness": config.smoothness,
        "smoothness_label": smooth_label,
        "bandwidth": config.bandwidth,
        "ridge": config.ridge,
        "n_train": config.n_train,
        "seed": config.seed,
        "kmax": config.kmax,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "effective_dim": effective_dim,
        "spectral_alignment": spectral_alignment(task["target_power"], q),
    }


def default_kernel_configs(seed: int = 0) -> list[KernelConfig]:
    configs: list[KernelConfig] = []
    targets = [("power_law", 1.0), ("exponential", 8.0)]
    smoothnesses = [0.75, 1.5, 3.0, np.inf]
    bandwidths = [4.0, 12.0, 32.0]
    ridges = [1e-4, 1e-2]
    n_trains = [64, 256, 1024]
    for target_kind, target_param in targets:
        for smoothness in smoothnesses:
            for bandwidth in bandwidths:
                for ridge in ridges:
                    for n_train in n_trains:
                        configs.append(
                            KernelConfig(
                                target_kind=target_kind,
                                target_param=target_param,
                                smoothness=smoothness,
                                bandwidth=bandwidth,
                                ridge=ridge,
                                n_train=n_train,
                                seed=seed,
                            )
                        )
    return configs
