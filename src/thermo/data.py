from __future__ import annotations

import gzip
import os
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def default_mnist_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_root = os.environ.get("MNIST_RAW_DIR")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(ROOT / "data" / "MNIST" / "raw")
    candidates.extend(sorted(ROOT.parent.glob("*/data/MNIST/raw")))
    return candidates


def _open_idx(path: Path):
    if path.exists():
        return path.open("rb")
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists():
        return gzip.open(gz_path, "rb")
    raise FileNotFoundError(path)


def _read_idx_images(path: Path) -> np.ndarray:
    with _open_idx(path) as f:
        magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} has IDX magic {magic}, expected 2051")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_images, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with _open_idx(path) as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} has IDX magic {magic}, expected 2049")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_labels)


def find_mnist_raw(extra_root: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if extra_root is not None:
        root = Path(extra_root)
        candidates.extend([root, root / "MNIST" / "raw", root / "data" / "MNIST" / "raw"])
    candidates.extend(default_mnist_candidates())
    for root in candidates:
        if (root / "train-images-idx3-ubyte").exists() or (
            root / "train-images-idx3-ubyte.gz"
        ).exists():
            if (root / "train-labels-idx1-ubyte").exists() or (
                root / "train-labels-idx1-ubyte.gz"
            ).exists():
                return root
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find cached MNIST IDX files. Searched:\n{searched}")


def downsample_28_to_14(images: np.ndarray) -> np.ndarray:
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"Expected images with shape (n, 28, 28), got {images.shape}")
    return images.reshape(images.shape[0], 14, 2, 14, 2).mean(axis=(2, 4))


def balanced_indices(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = n // len(classes)
    remainder = n % len(classes)
    chosen: list[np.ndarray] = []
    for i, cls in enumerate(classes):
        cls_idx = np.flatnonzero(labels == cls)
        take = per_class + (1 if i < remainder else 0)
        if take > len(cls_idx):
            raise ValueError(f"Requested {take} examples for class {cls}, only have {len(cls_idx)}")
        chosen.append(rng.choice(cls_idx, size=take, replace=False))
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


def load_mnist_arrays(
    max_train: int = 8192,
    n_val: int = 2000,
    seed: int = 0,
    raw_root: str | Path | None = None,
) -> dict[str, np.ndarray | str]:
    """Load a balanced, nested MNIST subset from cached IDX files."""

    root = find_mnist_raw(raw_root)
    train_images = _read_idx_images(root / "train-images-idx3-ubyte")
    train_labels = _read_idx_labels(root / "train-labels-idx1-ubyte")
    val_images = _read_idx_images(root / "t10k-images-idx3-ubyte")
    val_labels = _read_idx_labels(root / "t10k-labels-idx1-ubyte")

    train_idx = balanced_indices(train_labels, max_train, seed)
    val_idx = balanced_indices(val_labels, n_val, seed + 1)

    x_train = downsample_28_to_14(train_images[train_idx]).astype(np.float32) / 255.0
    x_val = downsample_28_to_14(val_images[val_idx]).astype(np.float32) / 255.0
    y_train = train_labels[train_idx].astype(np.int64)
    y_val = val_labels[val_idx].astype(np.int64)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-4, 1.0, std)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    return {
        "x_train": x_train.astype(np.float32),
        "y_train": y_train,
        "x_val": x_val.astype(np.float32),
        "y_val": y_val,
        "source": str(root),
        "max_train": max_train,
        "n_val": n_val,
    }
