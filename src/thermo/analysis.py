from __future__ import annotations

import numpy as np
import pandas as pd


def _local_response(
    df: pd.DataFrame,
    group_cols: list[str],
    x_col: str,
    y_col: str,
    x_transform,
) -> tuple[pd.Series, pd.Series]:
    """Return h=-dR/dtheta and C=|dh/dtheta| on nonuniform grids."""

    field = pd.Series(np.nan, index=df.index, dtype=float)
    heat = pd.Series(np.nan, index=df.index, dtype=float)
    for _, group in df.groupby(group_cols, dropna=False):
        group = group.sort_values(x_col)
        if group[x_col].nunique() < 2:
            continue
        idx = group.index
        x_raw = group[x_col].astype(float).to_numpy()
        y = group[y_col].astype(float).to_numpy()
        x = x_transform(x_raw)
        if np.unique(x).size < 2:
            continue
        edge_order = 2 if len(x) >= 3 else 1
        h = -np.gradient(y, x, edge_order=edge_order)
        c = (
            np.abs(np.gradient(h, x, edge_order=edge_order))
            if np.unique(x).size >= 3
            else np.full_like(h, np.nan, dtype=float)
        )
        field.loc[idx] = h
        heat.loc[idx] = c
    return field, heat


def _endpoint_pressure(
    df: pd.DataFrame,
    group_cols: list[str],
    x_col: str,
    y_col: str,
    out_name: str,
    log_x: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        group = group.sort_values(x_col)
        if group[x_col].nunique() < 2:
            continue
        lo = group.iloc[0]
        hi = group.iloc[-1]
        x0 = float(lo[x_col])
        x1 = float(hi[x_col])
        if log_x:
            x0 = np.log(x0)
            x1 = np.log(x1)
        denom = x1 - x0
        if abs(denom) < 1e-12:
            continue
        pressure = -float(hi[y_col] - lo[y_col]) / denom
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "x_low": float(lo[x_col]),
                "x_high": float(hi[x_col]),
                out_name: pressure,
                "loss_low": float(lo[y_col]),
                "loss_high": float(hi[y_col]),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[*group_cols, "x_low", "x_high", out_name, "loss_low", "loss_high"]
        )
    return pd.DataFrame(rows)


def summarize_neural(neural_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scratch = neural_df[neural_df["strategy"] == "scratch"].copy()
    final_by_seed = (
        scratch.sort_values("step")
        .groupby(["n_train", "width", "lr", "weight_decay", "seed"], as_index=False)
        .tail(1)
        .copy()
    )

    condition_cols = [
        "family",
        "strategy",
        "n_train",
        "width",
        "params",
        "lr",
        "weight_decay",
        "step",
        "epochs",
        "examples_seen",
    ]
    metric_cols = [
        col
        for col in [
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "l2_norm_sq",
            "regularizer_loss",
            "train_total_loss_reported",
            "val_total_loss_reported",
        ]
        if col in scratch.columns
    ]
    named_aggs = {
        col: pd.NamedAgg(column=col, aggfunc="mean")
        for col in metric_cols
    }
    named_aggs.update(
        {
            f"{col}_std": pd.NamedAgg(column=col, aggfunc="std")
            for col in metric_cols
        }
    )
    named_aggs["seed_count"] = pd.NamedAgg(column="seed", aggfunc="nunique")
    final = (
        final_by_seed.groupby(condition_cols, dropna=False, as_index=False)
        .agg(**named_aggs)
        .copy()
    )
    scratch_mean = (
        scratch.groupby(condition_cols, dropna=False, as_index=False)
        .agg(**named_aggs)
        .copy()
    )

    data_pressure = _endpoint_pressure(
        final,
        ["width", "lr", "weight_decay"],
        "n_train",
        "val_loss",
        "data_pressure",
    )
    capacity_pressure = _endpoint_pressure(
        final,
        ["n_train", "lr", "weight_decay"],
        "params",
        "val_loss",
        "capacity_pressure",
    )
    time_pressure = _endpoint_pressure(
        scratch_mean,
        ["n_train", "width", "lr", "weight_decay"],
        "step",
        "val_loss",
        "time_pressure",
    )

    reg_rows: list[dict[str, float | int | str]] = []
    for keys, group in final.groupby(["n_train", "width", "lr"]):
        if group["weight_decay"].nunique() < 2:
            continue
        low = group.sort_values("weight_decay").iloc[0]
        high = group.sort_values("weight_decay").iloc[-1]
        n_train, width, lr = keys
        reg_rows.append(
            {
                "n_train": n_train,
                "width": width,
                "lr": lr,
                "regularization_effect": float(high["val_loss"] - low["val_loss"]),
                "acc_effect": float(high["val_acc"] - low["val_acc"]),
            }
        )
    regularization = pd.DataFrame(reg_rows)
    if regularization.empty:
        regularization = pd.DataFrame(
            columns=[
                "n_train",
                "width",
                "lr",
                "regularization_effect",
                "acc_effect",
            ]
        )

    local = final.copy()
    local["theta_N_log2"] = np.log2(local["n_train"].astype(float))
    local["theta_P_log2"] = np.log2(local["params"].astype(float))
    local["theta_lambda_log10"] = np.nan
    positive_lambda_mask = local["weight_decay"] > 0
    local.loc[positive_lambda_mask, "theta_lambda_log10"] = np.log10(
        local.loc[positive_lambda_mask, "weight_decay"].astype(float)
    )

    h_data, c_data = _local_response(
        local,
        ["width", "lr", "weight_decay"],
        "n_train",
        "val_loss",
        np.log2,
    )
    h_cap, c_cap = _local_response(
        local,
        ["n_train", "lr", "weight_decay"],
        "params",
        "val_loss",
        np.log2,
    )
    h_reg = pd.Series(np.nan, index=local.index, dtype=float)
    c_reg = pd.Series(np.nan, index=local.index, dtype=float)
    pos_lambda = local[local["weight_decay"] > 0].copy()
    if not pos_lambda.empty:
        h_reg_pos, c_reg_pos = _local_response(
            pos_lambda,
            ["n_train", "width", "lr"],
            "weight_decay",
            "val_loss",
            np.log10,
        )
        h_reg.loc[h_reg_pos.index] = h_reg_pos
        c_reg.loc[c_reg_pos.index] = c_reg_pos

    local["h_data_per_doubling"] = h_data
    local["C_data_specific_heat"] = c_data
    local["h_capacity_per_doubling"] = h_cap
    local["C_capacity_specific_heat"] = c_cap
    local["h_regularization_per_decade"] = h_reg
    local["C_regularization_specific_heat"] = c_reg

    response_cols = [
        "h_data_per_doubling",
        "h_capacity_per_doubling",
        "h_regularization_per_decade",
    ]
    dominant: list[str] = []
    for _, row in local.iterrows():
        vals = row[response_cols].astype(float).abs().dropna()
        if vals.empty or vals.max() < 1e-4:
            dominant.append("flat")
            continue
        winner = vals.idxmax()
        if winner == "h_data_per_doubling":
            dominant.append("data-limited")
        elif winner == "h_capacity_per_doubling":
            dominant.append("capacity-limited")
        elif float(row[winner]) > 0:
            dominant.append("under-regularized")
        else:
            dominant.append("over-regularized")
    local["dominant_response"] = dominant

    if local["weight_decay"].gt(0).any():
        specific_heat_summary = (
            local[local["weight_decay"] > 0]
            .groupby(["weight_decay"], as_index=False)
            .agg(
                median_C_regularization=("C_regularization_specific_heat", "median"),
                median_h_regularization=("h_regularization_per_decade", "median"),
                median_val_loss=("val_loss", "median"),
            )
        )
    else:
        specific_heat_summary = pd.DataFrame(
            columns=[
                "weight_decay",
                "median_C_regularization",
                "median_h_regularization",
                "median_val_loss",
            ]
        )

    pressure_means = pd.DataFrame(
        [
            {"quantity": "data_pressure", "mean": data_pressure["data_pressure"].mean()},
            {
                "quantity": "capacity_pressure",
                "mean": capacity_pressure["capacity_pressure"].mean(),
            },
            {"quantity": "time_pressure", "mean": time_pressure["time_pressure"].mean()},
            {
                "quantity": "regularization_effect",
                "mean": regularization["regularization_effect"].mean(),
            },
        ]
    )

    return {
        "neural_final_by_seed": final_by_seed,
        "neural_final": final,
        "neural_data_pressure": data_pressure,
        "neural_capacity_pressure": capacity_pressure,
        "neural_time_pressure": time_pressure,
        "neural_regularization": regularization,
        "neural_local_derivatives": local,
        "neural_specific_heat_summary": specific_heat_summary,
        "neural_pressure_means": pressure_means,
    }


def summarize_kernel(kernel_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    best_ridge = (
        kernel_df.sort_values("val_mse")
        .groupby(["target_kind", "target_param", "smoothness_label", "bandwidth", "n_train"])
        .head(1)
        .reset_index(drop=True)
    )

    data_pressure = _endpoint_pressure(
        best_ridge,
        ["target_kind", "target_param", "smoothness_label", "bandwidth"],
        "n_train",
        "val_mse",
        "data_pressure",
    )

    reg_rows: list[dict[str, float | int | str]] = []
    for keys, group in kernel_df.groupby(
        ["target_kind", "target_param", "smoothness_label", "bandwidth", "n_train"]
    ):
        if group["ridge"].nunique() < 2:
            continue
        low = group.sort_values("ridge").iloc[0]
        high = group.sort_values("ridge").iloc[-1]
        target_kind, target_param, smoothness_label, bandwidth, n_train = keys
        reg_rows.append(
            {
                "target_kind": target_kind,
                "target_param": target_param,
                "smoothness_label": smoothness_label,
                "bandwidth": bandwidth,
                "n_train": n_train,
                "regularization_effect": float(high["val_mse"] - low["val_mse"]),
            }
        )
    regularization = pd.DataFrame(reg_rows)

    opt = (
        best_ridge.sort_values("val_mse")
        .groupby(["target_kind", "target_param", "n_train"])
        .head(1)
        .reset_index(drop=True)
    )

    pressure_means = pd.DataFrame(
        [
            {"quantity": "kernel_data_pressure", "mean": data_pressure["data_pressure"].mean()},
            {
                "quantity": "kernel_regularization_effect",
                "mean": regularization["regularization_effect"].mean(),
            },
        ]
    )

    return {
        "kernel_best_ridge": best_ridge,
        "kernel_data_pressure": data_pressure,
        "kernel_regularization": regularization,
        "kernel_optima": opt,
        "kernel_pressure_means": pressure_means,
    }
