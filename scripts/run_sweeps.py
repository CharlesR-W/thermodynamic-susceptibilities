from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermo.analysis import summarize_kernel, summarize_neural
from thermo.data import load_mnist_arrays
from thermo.kernel import default_kernel_configs, run_kernel_config
from thermo.neural import (
    default_neural_configs,
    dense_local_neural_configs,
    run_hysteresis_probe,
    train_mlp_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--profile", default="dense-local", choices=["pilot", "dense-local"])
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--include-hysteresis", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--skip-kernel", action="store_true")
    return parser.parse_args()


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false")
    return name


def write_tables(tables: dict[str, pd.DataFrame]) -> None:
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    if args.seeds is None:
        seeds = [0, 1, 2] if args.profile == "dense-local" else [args.seed]
    else:
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    started = time.perf_counter()
    tables: dict[str, pd.DataFrame] = {}
    metadata: dict[str, object] = {
        "seed": args.seed,
        "seeds": seeds,
        "data_seed": args.data_seed,
        "device": device,
        "profile": args.profile,
        "torch_version": torch.__version__,
        "started_unix": time.time(),
    }

    if not args.skip_neural:
        if args.profile == "dense-local":
            max_train = 50000
            n_val = 5000
            seed_configs = [dense_local_neural_configs(seed=seed) for seed in seeds]
            metadata["neural_control_protocol"] = (
                "scratch retrains; clustered log grid; fixed 3.0 epoch exposure; "
                "local derivatives in log2(N), log2(params), log10(weight_decay)"
            )
        else:
            max_train = 8192
            n_val = 2000
            seed_configs = [default_neural_configs(seed=seed) for seed in seeds]
            metadata["neural_control_protocol"] = (
                "scratch retrains; small pilot grid; fixed optimizer steps"
            )
        configs = [config for configs_for_seed in seed_configs for config in configs_for_seed]
        if args.max_configs is not None:
            configs = configs[: args.max_configs]
            metadata["max_configs"] = args.max_configs
        arrays = load_mnist_arrays(max_train=max_train, n_val=n_val, seed=args.data_seed)
        metadata["mnist_source"] = arrays["source"]
        metadata["mnist_max_train"] = max_train
        metadata["mnist_n_val"] = n_val
        metadata["neural_config_count"] = len(configs)
        metadata["neural_condition_count"] = len(configs) // max(1, len(seeds))
        metadata["neural_seed_count"] = len(seeds)
        neural_records = []
        for i, config in enumerate(configs, start=1):
            print(f"[neural {i:02d}/{len(configs)}] {config}")
            records, _ = train_mlp_config(config, arrays, device=device)
            neural_records.extend(records)
        if args.include_hysteresis:
            neural_records.extend(run_hysteresis_probe(arrays, device=device, seed=args.seed + 100))
        neural_df = pd.DataFrame(neural_records)
        tables["neural_raw"] = neural_df
        tables.update(summarize_neural(neural_df))

    if not args.skip_kernel:
        kernel_records = []
        configs = default_kernel_configs(seed=args.seed)
        for i, config in enumerate(configs, start=1):
            if i == 1 or i % 24 == 0:
                print(f"[kernel {i:03d}/{len(configs)}] {config}")
            kernel_records.append(run_kernel_config(config))
        kernel_df = pd.DataFrame(kernel_records)
        tables["kernel_raw"] = kernel_df
        tables.update(summarize_kernel(kernel_df))

    metadata["elapsed_s"] = time.perf_counter() - started
    metadata["tables"] = sorted(tables)
    write_tables(tables)
    (ROOT / "results" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "raw" / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
