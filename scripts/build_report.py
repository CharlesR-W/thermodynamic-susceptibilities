from __future__ import annotations

import base64
import html
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(ROOT / "src"))


FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
REPORT_PATH = ROOT / "reports" / "thermo_phase_report.html"


def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / f"{name}.csv")


def savefig(name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def save_html(name: str, content: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    path.write_text(content, encoding="utf-8")
    return path


def image_data(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def relative_report_path(path: Path) -> str:
    return os.path.relpath(path, REPORT_PATH.parent).replace(os.sep, "/")


def fmt_float(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return f"{x:.0e}"
    return f"{x:g}"


def heatmap(
    ax,
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = False,
):
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    if annotate and matrix.size <= 80:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3g}", ha="center", va="center", fontsize=7)
    return im


def closest_available(values: list[float], targets: list[float]) -> list[float]:
    out: list[float] = []
    arr = np.array(sorted(values), dtype=float)
    for target in targets:
        value = float(arr[np.argmin(np.abs(arr - target))])
        if value not in out:
            out.append(value)
    return out


def plot_neural_phase() -> Path:
    df = load_table("neural_final")
    df = df[df["strategy"] == "scratch"].copy()
    best = (
        df.sort_values("val_loss")
        .groupby(["n_train", "width", "weight_decay"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    n_vals = sorted(best["n_train"].unique())
    widths = sorted(best["width"].unique())
    selected_wd = closest_available(
        sorted(best["weight_decay"].unique()), [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2]
    )
    vmin = float(best["val_loss"].quantile(0.02))
    vmax = float(best["val_loss"].quantile(0.98))

    ncols = min(3, len(selected_wd))
    nrows = int(np.ceil(len(selected_wd) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)
    for ax, wd in zip(axes.flat, selected_wd):
        sub = best[np.isclose(best["weight_decay"], wd)]
        pivot = (
            sub.pivot_table(index="width", columns="n_train", values="val_loss", aggfunc="min")
            .reindex(widths)
            .loc[:, n_vals]
        )
        im = heatmap(
            ax,
            pivot.to_numpy(),
            [str(n) for n in n_vals],
            [str(w) for w in widths],
            rf"$R(N,P;\lambda={fmt_float(wd)})$ validation CE",
            "viridis_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(r"data size $N$")
        ax.set_ylabel(r"hidden width $W$")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.flat[len(selected_wd) :]:
        ax.axis("off")
    fig.suptitle(
        r"Phase surface: empirical potential $R=$ validation cross-entropy", y=1.01, fontsize=14
    )
    return savefig("neural_phase_dense.png")


def interp_along_axis(
    values: np.ndarray, old_coords: np.ndarray, new_coords: np.ndarray, axis: int
) -> np.ndarray:
    moved = np.moveaxis(values, axis, 0)
    interpolated = np.empty((len(new_coords),) + moved.shape[1:], dtype=float)
    for index in np.ndindex(moved.shape[1:]):
        interpolated[(slice(None),) + index] = np.interp(
            new_coords, old_coords, moved[(slice(None),) + index]
        )
    return np.moveaxis(interpolated, 0, axis)


def smooth_axis(values: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(values, axis, 0)
    pad_width = [(0, 0)] * moved.ndim
    pad_width[0] = (1, 1)
    padded = np.pad(moved, pad_width, mode="edge")
    smoothed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    return np.moveaxis(smoothed, 0, axis)


def smooth_volume(values: np.ndarray, passes: int = 2) -> np.ndarray:
    out = values.copy()
    for _ in range(passes):
        for axis in range(out.ndim):
            out = smooth_axis(out, axis)
    return out


def normalize_to_unit(values: np.ndarray | pd.Series, coords: np.ndarray) -> np.ndarray:
    span = float(coords.max() - coords.min())
    if span == 0:
        return np.zeros_like(np.asarray(values, dtype=float))
    return (np.asarray(values, dtype=float) - float(coords.min())) / span


def tick_indices(count: int, max_ticks: int = 7) -> list[int]:
    if count <= max_ticks:
        return list(range(count))
    return sorted(set(np.round(np.linspace(0, count - 1, max_ticks)).astype(int).tolist()))


def plot_neural_phase_3d_isosurfaces() -> Path:
    import plotly.graph_objects as go
    import plotly.io as pio

    df = load_table("neural_final")
    df = df[(df["strategy"] == "scratch") & (df["weight_decay"] > 0)].copy()
    df["theta_N"] = np.log2(df["n_train"].astype(float))
    df["theta_W"] = np.log2(df["width"].astype(float))
    df["theta_lambda"] = np.log10(df["weight_decay"].astype(float))

    n_coords = np.array(sorted(df["theta_N"].unique()), dtype=float)
    w_coords = np.array(sorted(df["theta_W"].unique()), dtype=float)
    lambda_coords = np.array(sorted(df["theta_lambda"].unique()), dtype=float)

    loss = np.full((len(n_coords), len(w_coords), len(lambda_coords)), np.nan, dtype=float)
    n_index = {value: idx for idx, value in enumerate(n_coords)}
    w_index = {value: idx for idx, value in enumerate(w_coords)}
    lambda_index = {value: idx for idx, value in enumerate(lambda_coords)}
    for row in df.itertuples(index=False):
        loss[
            n_index[row.theta_N],
            w_index[row.theta_W],
            lambda_index[row.theta_lambda],
        ] = row.val_loss
    if np.isnan(loss).any():
        raise ValueError("MNIST 3D phase grid is incomplete; cannot build iso-loss surface.")

    grand = float(loss.mean())
    additive_main_effects = (
        grand
        + (loss.mean(axis=(1, 2), keepdims=True) - grand)
        + (loss.mean(axis=(0, 2), keepdims=True) - grand)
        + (loss.mean(axis=(0, 1), keepdims=True) - grand)
    )
    residual = loss - additive_main_effects
    df["axis_N"] = normalize_to_unit(df["theta_N"], n_coords)
    df["axis_W"] = normalize_to_unit(df["theta_W"], w_coords)
    df["axis_lambda"] = normalize_to_unit(df["theta_lambda"], lambda_coords)
    df["loss_residual"] = [
        residual[n_index[theta_n], w_index[theta_w], lambda_index[theta_lambda]]
        for theta_n, theta_w, theta_lambda in zip(
            df["theta_N"], df["theta_W"], df["theta_lambda"]
        )
    ]
    df["loss_residual_mce"] = 1000.0 * df["loss_residual"]

    dense_n = np.linspace(n_coords.min(), n_coords.max(), 52)
    dense_w = np.linspace(w_coords.min(), w_coords.max(), 52)
    dense_lambda = np.linspace(lambda_coords.min(), lambda_coords.max(), 72)
    dense = interp_along_axis(loss, n_coords, dense_n, axis=0)
    dense = interp_along_axis(dense, w_coords, dense_w, axis=1)
    dense = interp_along_axis(dense, lambda_coords, dense_lambda, axis=2)
    dense = smooth_volume(dense, passes=2)
    dense_residual = interp_along_axis(residual, n_coords, dense_n, axis=0)
    dense_residual = interp_along_axis(dense_residual, w_coords, dense_w, axis=1)
    dense_residual = interp_along_axis(
        dense_residual, lambda_coords, dense_lambda, axis=2
    )
    dense_residual = smooth_volume(dense_residual, passes=2)
    dense_residual_mce = 1000.0 * dense_residual

    grid_n, grid_w, grid_lambda = np.meshgrid(
        normalize_to_unit(dense_n, n_coords),
        normalize_to_unit(dense_w, w_coords),
        normalize_to_unit(dense_lambda, lambda_coords),
        indexing="ij",
    )
    isomin = float(np.quantile(dense, 0.08))
    isomax = float(np.quantile(dense, 0.92))
    residual_span = float(np.quantile(np.abs(dense_residual_mce), 0.96))
    residual_span = max(residual_span, 1e-3)
    measured_min = float(df["val_loss"].min())
    measured_max = float(df["val_loss"].max())

    fig = go.Figure()
    fig.add_trace(
        go.Isosurface(
            x=grid_n.ravel(),
            y=grid_w.ravel(),
            z=grid_lambda.ravel(),
            value=dense.ravel(),
            isomin=isomin,
            isomax=isomax,
            surface_count=7,
            opacity=0.34,
            colorscale="Viridis",
            colorbar=dict(title="validation CE<br>R"),
            caps=dict(
                x=dict(show=False),
                y=dict(show=False),
                z=dict(show=False),
            ),
            hovertemplate=(
                "u_N=%{x:.2f}<br>"
                "u_W=%{y:.2f}<br>"
                "u_lambda=%{z:.2f}<br>"
                "interpolated R=%{value:.4f}<extra>iso-loss</extra>"
            ),
            name="raw iso-loss shells",
            visible=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=df["axis_N"],
            y=df["axis_W"],
            z=df["axis_lambda"],
            mode="markers",
            marker=dict(
                size=3.0,
                color=df["val_loss"],
                colorscale="Viridis",
                cmin=measured_min,
                cmax=measured_max,
                opacity=0.72,
                line=dict(width=0),
            ),
            text=[
                f"N={int(n)}, W={int(w)}, lambda={wd:g}, R={loss_value:.4f}"
                for n, w, wd, loss_value in zip(
                    df["n_train"], df["width"], df["weight_decay"], df["val_loss"]
                )
            ],
            hovertemplate="%{text}<extra>measured cell</extra>",
            name="measured 3-seed cells",
            visible=False,
        )
    )
    fig.add_trace(
        go.Isosurface(
            x=grid_n.ravel(),
            y=grid_w.ravel(),
            z=grid_lambda.ravel(),
            value=dense_residual_mce.ravel(),
            isomin=-residual_span,
            isomax=residual_span,
            surface_count=7,
            opacity=0.40,
            colorscale="RdBu_r",
            colorbar=dict(title="residual<br>mCE"),
            caps=dict(
                x=dict(show=False),
                y=dict(show=False),
                z=dict(show=False),
            ),
            hovertemplate=(
                "u_N=%{x:.2f}<br>"
                "u_W=%{y:.2f}<br>"
                "u_lambda=%{z:.2f}<br>"
                "balanced residual=%{value:.3f} mCE"
                "<extra>balanced curvature</extra>"
            ),
            name="balanced residual shells",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=df["axis_N"],
            y=df["axis_W"],
            z=df["axis_lambda"],
            mode="markers",
            marker=dict(
                size=3.0,
                color=df["loss_residual_mce"],
                colorscale="RdBu_r",
                cmin=-residual_span,
                cmax=residual_span,
                opacity=0.78,
                line=dict(width=0),
            ),
            text=[
                f"N={int(n)}, W={int(w)}, lambda={wd:g}, residual={resid:.3f} mCE"
                for n, w, wd, resid in zip(
                    df["n_train"],
                    df["width"],
                    df["weight_decay"],
                    df["loss_residual_mce"],
                )
            ],
            hovertemplate="%{text}<extra>measured residual cell</extra>",
            name="measured residual cells",
        )
    )

    n_ticks = tick_indices(len(n_coords))
    w_ticks = tick_indices(len(w_coords))
    lambda_ticks = tick_indices(len(lambda_coords), max_ticks=6)
    raw_title = "MNIST 3D view: raw validation iso-loss shells"
    residual_title = (
        "MNIST 3D diagnostic: balanced residual after additive axis trends are removed"
    )

    fig.update_layout(
        title=dict(text=residual_title, x=0.55, y=0.97, xanchor="center"),
        height=860,
        margin=dict(l=0, r=0, t=92, b=0),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=1,
                x=0.02,
                y=1.14,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Raw iso-loss R",
                        method="update",
                        args=[
                            {"visible": [True, True, False, False]},
                            {"title": dict(text=raw_title, x=0.55, y=0.97, xanchor="center")},
                        ],
                    ),
                    dict(
                        label="Balanced curvature residual",
                        method="update",
                        args=[
                            {"visible": [False, False, True, True]},
                            {
                                "title": dict(
                                    text=residual_title,
                                    x=0.55,
                                    y=0.97,
                                    xanchor="center",
                                )
                            },
                        ],
                    ),
                ],
            )
        ],
        scene=dict(
            domain=dict(x=[0.02, 0.88], y=[0.02, 0.98]),
            xaxis=dict(
                title=r"data axis u_N, ticks show N",
                tickvals=normalize_to_unit(n_coords[n_ticks], n_coords),
                ticktext=[str(int(round(2**value))) for value in n_coords[n_ticks]],
                gridcolor="#d9dee5",
                backgroundcolor="#ffffff",
                range=[0, 1],
            ),
            yaxis=dict(
                title=r"capacity axis u_W, ticks show W",
                tickvals=normalize_to_unit(w_coords[w_ticks], w_coords),
                ticktext=[str(int(round(2**value))) for value in w_coords[w_ticks]],
                gridcolor="#d9dee5",
                backgroundcolor="#ffffff",
                range=[0, 1],
            ),
            zaxis=dict(
                title=r"regularization axis u_lambda, ticks show lambda",
                tickvals=normalize_to_unit(lambda_coords[lambda_ticks], lambda_coords),
                ticktext=[fmt_float(float(10**value)) for value in lambda_coords[lambda_ticks]],
                gridcolor="#d9dee5",
                backgroundcolor="#ffffff",
                range=[0, 1],
            ),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.45, y=-1.62, z=1.18)),
        ),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.72)"),
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, Arial, sans-serif", size=13),
    )

    plot_div = pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
    )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MNIST 3D Iso-Loss Cutouts</title>
<style>
body {{
  margin: 0;
  font-family: Inter, Arial, sans-serif;
  color: #18202a;
  background: #ffffff;
}}
.wrap {{ padding: 14px 16px 8px; }}
h1 {{ margin: 0 0 6px; font-size: 20px; }}
p {{ margin: 0 0 10px; max-width: 1120px; color: #5d6875; line-height: 1.42; }}
.plot {{ height: 900px; }}
.plot .plotly-graph-div {{ height: 860px !important; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>MNIST 3D Iso-Loss Cutouts</h1>
  <p>Axes are normalized log coordinates for data size N, hidden width W, and positive weight decay
  lambda, with tick labels in the original units. Use Raw iso-loss R for the faithful smoothed
  validation-CE surface. The default balanced residual mode subtracts additive N, W, and lambda
  main effects to make small interactions visible; it is a diagnostic visualization, not a
  separate loss or phase diagram. Orthogonal slice planes are intentionally disabled.</p>
</div>
<div class="plot">{plot_div}</div>
</body>
</html>"""
    return save_html("neural_phase_3d_isosurfaces.html", html_doc)


def plot_neural_pressure_fields() -> Path:
    df = load_table("neural_local_derivatives")
    df = df[df["strategy"] == "scratch"].copy()
    positive = df[df["weight_decay"] > 0]
    if positive.empty:
        positive = df
    selected_wd = closest_available(sorted(positive["weight_decay"].unique()), [1e-5, 1e-3, 1e-2])
    n_vals = sorted(df["n_train"].unique())
    widths = sorted(df["width"].unique())

    fig, axes = plt.subplots(1, len(selected_wd), figsize=(4.7 * len(selected_wd), 3.8), sharey=True)
    if len(selected_wd) == 1:
        axes = [axes]
    vals = df["h_regularization_per_decade"].replace([np.inf, -np.inf], np.nan)
    vmax = float(np.nanquantile(np.abs(vals), 0.95)) if vals.notna().any() else 1.0
    vmax = max(vmax, 1e-3)
    for ax, wd in zip(axes, selected_wd):
        sub = df[np.isclose(df["weight_decay"], wd)]
        pivot = (
            sub.pivot_table(
                index="width", columns="n_train", values="h_regularization_per_decade", aggfunc="median"
            )
            .reindex(widths)
            .loc[:, n_vals]
        )
        im = heatmap(
            ax,
            pivot.to_numpy(),
            [str(n) for n in n_vals],
            [str(w) for w in widths],
            rf"$h_\lambda=-\partial R/\partial\log_{{10}}\lambda$, $\lambda={fmt_float(wd)}$",
            "coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlabel(r"data size $N$")
        ax.set_ylabel(r"hidden width $W$")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        r"Regularization field: positive means stronger weight decay helps locally",
        y=1.03,
        fontsize=14,
    )
    return savefig("neural_regularization_field.png")


def neural_lower_envelope() -> tuple[pd.DataFrame, list[int], list[int], np.ndarray, np.ndarray, np.ndarray]:
    df = load_table("neural_final")
    df = df[df["strategy"] == "scratch"].copy()
    best = (
        df.sort_values("val_loss")
        .groupby(["n_train", "width"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    n_vals = sorted(best["n_train"].unique())
    widths = sorted(best["width"].unique())
    risk = (
        best.pivot_table(index="width", columns="n_train", values="val_loss", aggfunc="min")
        .reindex(widths)
        .loc[:, n_vals]
        .to_numpy()
    )
    params_by_width = (
        best.groupby("width")["params"].median().reindex(widths).astype(float).to_numpy()
    )
    theta_n = np.log2(np.array(n_vals, dtype=float))
    theta_p = np.log2(params_by_width)
    return best, n_vals, widths, risk, theta_n, theta_p


def plot_neural_envelope_susceptibilities() -> Path:
    best, n_vals, widths, risk, theta_n, theta_p = neural_lower_envelope()
    h_data = -np.gradient(risk, theta_n, axis=1, edge_order=2)
    h_capacity = -np.gradient(risk, theta_p, axis=0, edge_order=2)
    sem = (
        best.assign(sem=best["val_loss_std"] / np.sqrt(best["seed_count"].clip(lower=1)))
        .pivot_table(index="width", columns="n_train", values="sem", aggfunc="median")
        .reindex(widths)
        .loc[:, n_vals]
        .to_numpy()
    )
    response_floor = max(float(np.nanmedian(sem) * 2.0), 0.006)

    categories = ["unresolved", "data response", "capacity response", "mixed/noisy"]
    colors = ["#d9dee5", "#4c78a8", "#59a14f", "#e15759"]
    codes = np.zeros_like(risk, dtype=int)
    strongest_positive = np.maximum(h_data, h_capacity)
    nonmonotone = (np.maximum(np.abs(h_data), np.abs(h_capacity)) >= response_floor) & (
        strongest_positive < response_floor
    )
    data_wins = (h_data >= h_capacity) & (h_data >= response_floor)
    cap_wins = (h_capacity > h_data) & (h_capacity >= response_floor)
    codes[data_wins] = 1
    codes[cap_wins] = 2
    codes[nonmonotone] = 3

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), squeeze=False)
    xlabels = [str(n) for n in n_vals]
    ylabels = [str(w) for w in widths]

    im0 = heatmap(
        axes[0, 0],
        risk,
        xlabels,
        ylabels,
        r"Lower envelope: best validation CE over $\lambda$",
        "viridis_r",
    )
    axes[0, 0].set_xlabel(r"data size $N$")
    axes[0, 0].set_ylabel(r"hidden width $W$")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, label="validation CE")

    vmax_h = float(np.nanquantile(np.abs([h_data, h_capacity]), 0.95))
    vmax_h = max(vmax_h, response_floor * 2)
    im1 = heatmap(
        axes[0, 1],
        h_data,
        xlabels,
        ylabels,
        r"$h_N^*=-\partial R^*/\partial\log_2 N$",
        "coolwarm",
        vmin=-vmax_h,
        vmax=vmax_h,
    )
    axes[0, 1].set_xlabel(r"data size $N$")
    axes[0, 1].set_ylabel(r"hidden width $W$")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, label="risk decrease per data doubling")

    im2 = heatmap(
        axes[1, 0],
        h_capacity,
        xlabels,
        ylabels,
        r"$h_P^*=-\partial R^*/\partial\log_2 P$",
        "coolwarm",
        vmin=-vmax_h,
        vmax=vmax_h,
    )
    axes[1, 0].set_xlabel(r"data size $N$")
    axes[1, 0].set_ylabel(r"hidden width $W$")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, label="risk decrease per parameter doubling")

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(categories) + 1) - 0.5, len(categories))
    im3 = axes[1, 1].imshow(codes, aspect="auto", cmap=cmap, norm=norm)
    axes[1, 1].set_xticks(range(len(n_vals)))
    axes[1, 1].set_xticklabels(xlabels, rotation=35, ha="right")
    axes[1, 1].set_yticks(range(len(widths)))
    axes[1, 1].set_yticklabels(ylabels)
    axes[1, 1].set_xlabel(r"data size $N$")
    axes[1, 1].set_ylabel(r"hidden width $W$")
    axes[1, 1].set_title(r"Largest lower-envelope response (thresholded)")
    handles = [Patch(facecolor=color, edgecolor="none", label=cat) for color, cat in zip(colors, categories)]
    axes[1, 1].legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(
        "Post-hoc lower-envelope finite differences: descriptive capacity/data diagnostic",
        y=1.02,
        fontsize=15,
    )
    return savefig("neural_envelope_susceptibilities.png")


def plot_neural_regularization_audit() -> Path:
    df = load_table("neural_final")
    df = df[df["strategy"] == "scratch"].copy()
    n_vals = sorted(df["n_train"].unique())
    widths = sorted(df["width"].unique())
    wd_vals = sorted(df["weight_decay"].unique())

    best = (
        df.sort_values("val_loss")
        .groupby(["n_train", "width"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    zero = df[np.isclose(df["weight_decay"], 0.0)][["n_train", "width", "val_loss"]].rename(
        columns={"val_loss": "zero_loss"}
    )
    span = (
        df.groupby(["n_train", "width"], as_index=False)
        .agg(
            lambda_span=("val_loss", lambda values: float(values.max() - values.min())),
            median_sem=(
                "val_loss_std",
                lambda values: float(np.nanmedian(values) / np.sqrt(3.0)),
            ),
        )
    )
    audit = best.merge(zero, on=["n_train", "width"], how="left").merge(
        span, on=["n_train", "width"], how="left"
    )
    audit["gain_vs_zero"] = audit["zero_loss"] - audit["val_loss"]
    audit["gain_snr"] = audit["gain_vs_zero"] / audit["median_sem"].replace(0, np.nan)

    best_wd_code = {wd: idx for idx, wd in enumerate(wd_vals)}
    best_wd = (
        audit.assign(wd_code=audit["weight_decay"].map(best_wd_code))
        .pivot_table(index="width", columns="n_train", values="wd_code", aggfunc="first")
        .reindex(widths)
        .loc[:, n_vals]
        .to_numpy()
    )
    gain = (
        1000
        * audit.pivot_table(index="width", columns="n_train", values="gain_vs_zero", aggfunc="first")
        .reindex(widths)
        .loc[:, n_vals]
        .to_numpy()
    )
    snr = (
        audit.pivot_table(index="width", columns="n_train", values="gain_snr", aggfunc="first")
        .reindex(widths)
        .loc[:, n_vals]
        .to_numpy()
    )
    lambda_span = (
        1000
        * audit.pivot_table(index="width", columns="n_train", values="lambda_span", aggfunc="first")
        .reindex(widths)
        .loc[:, n_vals]
        .to_numpy()
    )

    local = load_table("neural_local_derivatives")
    pos = local[(local["strategy"] == "scratch") & (local["weight_decay"] > 0)].copy()
    magnitudes = [
        pos["h_data_per_doubling"].abs().dropna().to_numpy(),
        pos["h_capacity_per_doubling"].abs().dropna().to_numpy(),
        pos["h_regularization_per_decade"].abs().dropna().to_numpy(),
    ]
    labels = [r"$|h_N|$", r"$|h_P|$", r"$|h_\lambda|$"]
    medians = [float(np.nanmedian(values)) for values in magnitudes]
    lo = [float(np.nanquantile(values, 0.25)) for values in magnitudes]
    hi = [float(np.nanquantile(values, 0.75)) for values in magnitudes]
    yerr = np.array([[m - l for m, l in zip(medians, lo)], [h - m for m, h in zip(medians, hi)]])

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), squeeze=False)
    xlabels = [str(n) for n in n_vals]
    ylabels = [str(w) for w in widths]

    wd_cmap = ListedColormap(plt.cm.viridis(np.linspace(0.05, 0.95, len(wd_vals))))
    wd_norm = BoundaryNorm(np.arange(len(wd_vals) + 1) - 0.5, len(wd_vals))
    im0 = axes[0, 0].imshow(best_wd, aspect="auto", cmap=wd_cmap, norm=wd_norm)
    axes[0, 0].set_xticks(range(len(n_vals)))
    axes[0, 0].set_xticklabels(xlabels, rotation=35, ha="right")
    axes[0, 0].set_yticks(range(len(widths)))
    axes[0, 0].set_yticklabels(ylabels)
    axes[0, 0].set_title(r"Best $\lambda$ on lower envelope")
    axes[0, 0].set_xlabel(r"data size $N$")
    axes[0, 0].set_ylabel(r"hidden width $W$")
    cbar = fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, ticks=range(len(wd_vals)))
    cbar.ax.set_yticklabels([fmt_float(float(wd)) for wd in wd_vals])

    vmax_gain = max(float(np.nanquantile(np.abs(gain), 0.95)), 1.0)
    im1 = heatmap(
        axes[0, 1],
        gain,
        xlabels,
        ylabels,
        r"Tuning gain vs $\lambda=0$ (mCE)",
        "coolwarm",
        vmin=-vmax_gain,
        vmax=vmax_gain,
    )
    axes[0, 1].set_xlabel(r"data size $N$")
    axes[0, 1].set_ylabel(r"hidden width $W$")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, label="milli-cross-entropy")

    vmax_snr = max(float(np.nanquantile(np.abs(snr), 0.95)), 2.0)
    im2 = heatmap(
        axes[1, 0],
        snr,
        xlabels,
        ylabels,
        r"Gain / seed standard error",
        "coolwarm",
        vmin=-vmax_snr,
        vmax=vmax_snr,
    )
    axes[1, 0].set_xlabel(r"data size $N$")
    axes[1, 0].set_ylabel(r"hidden width $W$")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, label="signal/noise")

    ax = axes[1, 1]
    ax.bar(labels, medians, color=["#4c78a8", "#59a14f", "#f28e2b"], alpha=0.88)
    ax.errorbar(labels, medians, yerr=yerr, fmt="none", ecolor="#20242b", capsize=5, lw=1.3)
    ax.set_yscale("log")
    ax.set_ylabel("median absolute response")
    ax.set_title("Response scale audit: λ signal is tiny")
    ax.grid(axis="y", alpha=0.25)
    span_label = f"median λ-span = {np.nanmedian(lambda_span):.2f} mCE"
    frac_snr = float(np.nanmean(np.abs(snr) > 2.0))
    ax.text(
        0.03,
        0.95,
        f"{span_label}\n|gain/SE|>2 in {100 * frac_snr:.1f}% of cells",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="#4f5b68",
    )

    fig.suptitle(
        "Regularization audit: tune it, but do not infer a clean λ phase boundary here",
        y=1.02,
        fontsize=15,
    )
    return savefig("neural_regularization_audit.png")


def plot_neural_specific_heat_curves() -> Path:
    df = load_table("neural_local_derivatives")
    df = df[(df["strategy"] == "scratch") & (df["weight_decay"] > 0)].copy()
    n_vals = closest_available(sorted(df["n_train"].unique()), [8192, 16384, 32768, 50000])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharex=True)
    for n_train in n_vals:
        sub = df[df["n_train"] == n_train]
        grouped = (
            sub.groupby("weight_decay", as_index=False)
            .agg(
                h_lambda=("h_regularization_per_decade", "median"),
                C_lambda=("C_regularization_specific_heat", "median"),
            )
            .sort_values("weight_decay")
        )
        axes[0].plot(
            grouped["weight_decay"],
            grouped["h_lambda"],
            marker="o",
            lw=1.8,
            label=f"N={int(n_train)}",
        )
        axes[1].plot(
            grouped["weight_decay"],
            grouped["C_lambda"],
            marker="o",
            lw=1.8,
            label=f"N={int(n_train)}",
        )
    axes[0].axhline(0, color="#20242b", lw=0.9)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylabel(r"$h_\lambda=-\partial R/\partial\log_{10}\lambda$")
    axes[1].set_ylabel(r"$C_\lambda=|\partial h_\lambda/\partial\log_{10}\lambda|$")
    for ax in axes:
        ax.set_xlabel(r"weight decay $\lambda$")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_title("Raw lambda response: weak and sign-sensitive")
    axes[1].set_title("Second difference: low-confidence diagnostic")
    fig.suptitle(
        r"Raw regularization finite differences: do not read these as discovered phase boundaries",
        y=1.03,
        fontsize=14,
    )
    return savefig("neural_specific_heat_curves.png")


def plot_neural_loss_decomposition() -> Path:
    df = load_table("neural_final")
    df = df[(df["strategy"] == "scratch") & (df["n_train"] == df["n_train"].max())].copy()
    best = df.sort_values("val_loss").groupby("weight_decay", as_index=False).head(1)
    best = best.sort_values("weight_decay")
    labels = [fmt_float(float(v)) for v in best["weight_decay"]]
    x = np.arange(len(best))
    ce = best["val_loss"].to_numpy()
    reg = best["regularizer_loss"].to_numpy() if "regularizer_loss" in best else np.zeros_like(ce)
    total = ce + reg

    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    ax.bar(x, ce, color="#4c78a8", label=r"real loss $R_{\rm CE}$")
    ax.bar(x, reg, bottom=ce, color="#f28e2b", label=r"diagnostic L2 term $\lambda\|w\|^2/2$")
    ax.plot(x, total, color="#20242b", marker="o", lw=1.4, label="reported total")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel(r"weight decay $\lambda$")
    ax.set_ylabel("validation objective components")
    ax.set_title(
        r"Loss decomposition at largest $N$: CE plus measured L2 penalty of best-width cells"
    )
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.98,
        "AdamW is decoupled; orange bars are a diagnostic additive penalty, not the exact optimized loss.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#4f5b68",
    )
    return savefig("neural_loss_decomposition.png")


def inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def markdown_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    in_list = False
    in_code = False
    in_math = False
    block: list[str] = []
    paragraph: list[str] = []
    list_item: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(f"<p>{inline_markdown(' '.join(paragraph))}</p>")
            paragraph = []

    def flush_list_item() -> None:
        nonlocal list_item
        if list_item:
            out.append(f"<li>{inline_markdown(' '.join(list_item))}</li>")
            list_item = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            flush_list_item()
            out.append("</ul>")
            in_list = False

    for line in lines:
        stripped = line.strip()
        if line.strip() == "$$":
            if in_math:
                out.append('<div class="math-block">$$\n' + html.escape("\n".join(block)) + "\n$$</div>")
                block = []
                in_math = False
            else:
                flush_paragraph()
                close_list()
                in_math = True
            continue
        if in_math:
            block.append(line)
            continue
        if line.startswith("```"):
            if in_code:
                out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
                block = []
                in_code = False
            else:
                flush_paragraph()
                close_list()
                in_code = True
            continue
        if in_code:
            block.append(line)
            continue
        if line.startswith("- "):
            flush_paragraph()
            if not in_list:
                out.append("<ul>")
                in_list = True
            flush_list_item()
            list_item = [line[2:].strip()]
            continue
        if in_list and (line.startswith("  ") or line.startswith("\t")) and stripped:
            list_item.append(stripped)
            continue
        if in_list:
            close_list()
        if line.startswith("# "):
            flush_paragraph()
            out.append(f"<h1>{inline_markdown(line[2:])}</h1>")
        elif line.startswith("## "):
            flush_paragraph()
            out.append(f"<h2>{inline_markdown(line[3:])}</h2>")
        elif stripped:
            paragraph.append(stripped)
        else:
            flush_paragraph()
    flush_paragraph()
    close_list()
    return "\n".join(out)


def small_table(df: pd.DataFrame, cols: list[str], n: int = 8) -> str:
    present = [col for col in cols if col in df.columns]
    return df[present].head(n).to_html(
        index=False, classes="data-table", border=0, float_format="%.4g"
    )


def metadata_summary_html(metadata: dict[str, object]) -> str:
    seeds = metadata.get("seeds", [])
    if isinstance(seeds, list):
        seed_text = ", ".join(str(seed) for seed in seeds)
    else:
        seed_text = str(seeds)

    elapsed = metadata.get("elapsed_s")
    if isinstance(elapsed, (int, float)):
        elapsed_text = f"{elapsed / 60:.1f} min"
    else:
        elapsed_text = None

    rows = [
        ("profile", metadata.get("profile")),
        ("seeds", seed_text),
        ("scratch retrains", metadata.get("neural_config_count")),
        ("unique conditions", metadata.get("neural_condition_count")),
        ("validation examples", metadata.get("mnist_n_val")),
        ("device", metadata.get("device")),
        ("runtime", elapsed_text),
    ]
    items = [
        f"<li><b>{html.escape(label)}</b>: {html.escape(str(value))}</li>"
        for label, value in rows
        if value not in {None, ""}
    ]
    protocol = metadata.get("neural_control_protocol")
    if protocol:
        items.append(
            "<li><b>protocol</b>: "
            f"{html.escape(str(protocol))}</li>"
        )
    items.append("<li><b>data</b>: cached MNIST IDX files</li>")
    return "".join(items)


def regularization_audit_summary_html(
    neural_final: pd.DataFrame, neural_local: pd.DataFrame
) -> str:
    df = neural_final[neural_final["strategy"] == "scratch"].copy()
    best = (
        df.sort_values("val_loss")
        .groupby(["n_train", "width"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    zero = df[np.isclose(df["weight_decay"], 0.0)][["n_train", "width", "val_loss"]].rename(
        columns={"val_loss": "zero_loss"}
    )
    span = (
        df.groupby(["n_train", "width"], as_index=False)
        .agg(
            lambda_span=("val_loss", lambda values: float(values.max() - values.min())),
            median_sem=("val_loss_std", lambda values: float(np.nanmedian(values) / np.sqrt(3.0))),
        )
    )
    audit = best.merge(zero, on=["n_train", "width"], how="left").merge(
        span, on=["n_train", "width"], how="left"
    )
    audit["gain_vs_zero"] = audit["zero_loss"] - audit["val_loss"]
    audit["gain_snr"] = audit["gain_vs_zero"] / audit["median_sem"].replace(0, np.nan)
    frac_snr = float(np.nanmean(np.abs(audit["gain_snr"]) > 2.0))
    median_gain_mce = float(1000.0 * np.nanmedian(audit["gain_vs_zero"]))
    median_span_mce = float(1000.0 * np.nanmedian(audit["lambda_span"]))

    pos = neural_local[
        (neural_local["strategy"] == "scratch") & (neural_local["weight_decay"] > 0)
    ].copy()
    h_n = float(np.nanmedian(pos["h_data_per_doubling"].abs()))
    h_p = float(np.nanmedian(pos["h_capacity_per_doubling"].abs()))
    h_l = float(np.nanmedian(pos["h_regularization_per_decade"].abs()))
    wd_values = sorted(float(v) for v in df["weight_decay"].unique())
    positive_wd = [v for v in wd_values if v > 0]
    wd_range = f"{fmt_float(min(positive_wd))} to {fmt_float(max(positive_wd))}"

    rows = [
        ("positive weight-decay range", wd_range),
        ("median tuning gain vs lambda=0", f"{median_gain_mce:.2f} mCE"),
        ("median full lambda span", f"{median_span_mce:.2f} mCE"),
        ("cells with |gain/SE| > 2", f"{100 * frac_snr:.1f}%"),
        ("median |h_N|, |h_P|, |h_lambda|", f"{h_n:.3g}, {h_p:.3g}, {h_l:.3g}"),
    ]
    return "".join(
        f"<li><b>{html.escape(label)}</b>: {html.escape(value)}</li>"
        for label, value in rows
    )


def build_html(figures: dict[str, Path]) -> str:
    spec_html = markdown_to_html((ROOT / "docs" / "spec.md").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "results" / "raw" / "run_metadata.json").read_text())
    neural_final = load_table("neural_final").sort_values("val_loss")
    neural_local = load_table("neural_local_derivatives").sort_values("val_loss")
    fig_html = {
        name: image_data(path)
        for name, path in figures.items()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    neural_phase_3d_src = html.escape(relative_report_path(figures["neural_phase_3d"]))
    metadata_bits = metadata_summary_html(metadata)
    reg_audit_bits = regularization_audit_summary_html(neural_final, neural_local)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thermodynamic Susceptibilities for Learning</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [["$", "$"], ["\\\\(", "\\\\)"]],
    displayMath: [["$$", "$$"], ["\\\\[", "\\\\]"]]
  }},
  options: {{ skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"] }}
}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
:root {{
  color-scheme: light;
  --ink: #18202a;
  --muted: #5d6875;
  --line: #d9dee5;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --accent: #365f91;
}}
body {{
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
}}
header {{
  padding: 28px min(5vw, 56px) 18px;
  background: #ffffff;
  border-bottom: 1px solid var(--line);
}}
h1, h2, h3 {{ letter-spacing: 0; }}
header h1 {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 46px); }}
header p {{ max-width: 1040px; margin: 0; color: var(--muted); font-size: 17px; line-height: 1.45; }}
nav {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  padding: 12px min(5vw, 56px);
  background: #eef1f5;
  position: sticky;
  top: 0;
  z-index: 3;
  border-bottom: 1px solid var(--line);
}}
nav button {{
  border: 1px solid var(--line);
  background: #ffffff;
  color: var(--ink);
  padding: 8px 11px;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 600;
}}
nav button.active {{ background: var(--accent); color: #ffffff; border-color: var(--accent); }}
main {{ padding: 22px min(5vw, 56px) 54px; }}
section.tab {{ display: none; }}
section.tab.active {{ display: block; }}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 360px), 1fr));
  gap: 18px;
  align-items: start;
}}
.wide {{ grid-column: 1 / -1; }}
.panel {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
}}
.panel h3 {{ margin-top: 0; }}
.figure {{ width: 100%; display: block; }}
.interactive-figure {{
  width: 100%;
  height: min(84vh, 980px);
  min-height: 760px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #ffffff;
}}
.note {{ color: var(--muted); line-height: 1.5; }}
.lead {{ font-size: 17px; line-height: 1.56; }}
.callout {{
  border-left: 4px solid var(--accent);
  padding: 10px 12px;
  background: #f2f5f9;
  color: var(--ink);
}}
.warning {{
  border-left-color: #b85b2b;
  background: #fff4ed;
}}
.formula {{
  overflow-x: auto;
  background: #f0f3f6;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px 12px;
  line-height: 1.45;
}}
.metric-list, .plot-list {{ padding-left: 20px; }}
.plot-list li, .metric-list li {{ margin-bottom: 8px; line-height: 1.45; }}
code {{
  background: #eef1f5;
  border: 1px solid #dce2e8;
  border-radius: 4px;
  padding: 0 4px;
}}
.data-table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
}}
.data-table th, .data-table td {{
  border-bottom: 1px solid var(--line);
  padding: 6px 7px;
  text-align: right;
}}
.data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
pre, .math-block {{
  overflow-x: auto;
  background: #f0f3f6;
  border: 1px solid var(--line);
  padding: 12px;
  border-radius: 6px;
  white-space: pre;
}}
#spec h1 {{ display: none; }}
#spec p, #spec li {{ max-width: 1020px; line-height: 1.58; }}
</style>
</head>
<body>
<header>
  <h1>Thermodynamic Susceptibilities for Learning</h1>
  <p>A finite MNIST test of whether Lagrange-multiplier-like responses can define a useful notion of neural-network capacity. The short version: data and width responses are real in this sweep; the direct L2/weight-decay signal is too weak here to support a clean capacity-collapse story.</p>
</header>
<nav>
  <button class="active" data-tab="overview">Overview</button>
  <button data-tab="spec">Spec</button>
  <button data-tab="mnist">MNIST Phase</button>
  <button data-tab="suscept">Susceptibilities</button>
  <button data-tab="scale">Scale</button>
  <button data-tab="caveats">Caveats</button>
</nav>
<main>
<section id="overview" class="tab active">
  <div class="grid">
    <div class="panel wide">
      <h3>What This Is Testing</h3>
      <p class="lead">The motivating idea is to define neural-network capacity by a constrained optimization problem, then read the Lagrange multiplier as a susceptibility or shadow price. The point is not that Lagrange multipliers are new; it is that they give a common currency for comparing many different regularizers. Weight norm, rank, activation sparsity, noise robustness, and architectural bottlenecks can all be phrased as constraints with conjugate fields.</p>
      <p class="lead">If a norm budget is a real capacity constraint, tightening that budget should eventually make the task impossible; in the extreme, very heavy L2 regularization should collapse the network toward a near-zero/constant predictor and drive task capacity to zero.</p>
      <p class="lead">This MNIST run is a small empirical probe of that idea. It asks whether validation risk has a stable response to data size, hidden width, and weight decay, and whether the weight-decay direction has enough signal to behave like a meaningful capacity field.</p>
    </div>
    <div class="panel">
      <h3>Mathematical Context</h3>
      <p class="note">A clean capacity story would start with a norm budget <code>B</code> and a constrained risk:</p>
      <div class="formula">$$R^*(B)=\\min_{{\\lVert w\\rVert_2^2\\le B}} R(w),\\qquad \\mu^*(B)=-\\frac{{dR^*}}{{dB}}.$$</div>
      <p class="note">Here <code>mu*</code> is the shadow price of capacity: how much validation risk would fall if the allowed norm budget were relaxed. A penalty form <code>R(w)+lambda ||w||^2/2</code> is the Lagrangian proxy, so the response to <code>lambda</code> should reveal whether norm is acting like a real capacity bottleneck.</p>
    </div>
    <div class="panel">
      <h3>Ensemble View</h3>
      <p class="note">The more thermodynamic version is not one trained network but an ensemble over networks. Hard constraints become expectation constraints, e.g. <code>E_p[||w||^2]=B</code>. A max-entropy problem then gives a Gibbs family:</p>
      <div class="formula">$$\\begin{{aligned}}p(w)&\\propto \\exp\\left[-\\beta L(w)-\\sum_a \\lambda_a q_a(w)\\right],\\\\\\mathbb{{E}}_p[q_a]&=B_a.\\end{{aligned}}$$</div>
      <p class="note">This is attractive because very different regularizers become observables <code>q_a</code> with comparable conjugate fields <code>lambda_a</code>. Rank constraints, spectral penalties, dropout/noise robustness, and input or weight noise can all be studied as softened ensemble constraints if the observable is chosen carefully.</p>
    </div>
    <div class="panel">
      <h3>Architectural Constraints</h3>
      <p class="note">Some capacity controls are not smooth scalar penalties at first. Fixing a matrix rank, width, sparsity pattern, or attention bottleneck is closer to changing the architecture. The possible bridge is to replace the hard constraint by a softened observable, such as nuclear norm, effective rank, spectral decay, mask entropy, or expected active units.</p>
      <p class="note">That softening is analytically useful because it may make the field differentiable, support local linearization, and expose universality classes: different microscopic regularizers could have the same coarse response law.</p>
    </div>
    <div class="panel">
      <h3>Empirical Proxy</h3>
      <p class="note">The actual sweep measures validation cross-entropy <code>R(N,W,lambda)</code> after scratch retraining. The plotted fields are finite differences in log coordinates:</p>
      <div class="formula">$$\\begin{{aligned}}h_N&=-\\frac{{\\partial R}}{{\\partial\\log_2 N}},\\\\h_P&=-\\frac{{\\partial R}}{{\\partial\\log_2 P}},\\\\h_\\lambda&=-\\frac{{\\partial R}}{{\\partial\\log_{{10}}\\lambda}}.\\end{{aligned}}$$</div>
      <p class="note">Positive <code>h_N</code> means another data doubling helps. Positive <code>h_P</code> means another parameter doubling helps. Positive <code>h_lambda</code> means stronger weight decay helps locally; negative <code>h_lambda</code> means it hurts.</p>
    </div>
    <div class="panel">
      <h3>Coordinate Caveat</h3>
      <p class="note">The current log-scale derivatives are a practical normalization, not a principled invariant. A cleaner comparison would use responses to constraint budgets themselves, for example <code>-dR*/dB</code> or the dimensionless elasticity <code>-dR*/d log B = B mu*</code>.</p>
      <p class="note">For ensembles, the natural local geometry is the covariance/Fisher matrix of observables, <code>Cov_p(q_a,q_b)</code>. For networks, parameterization symmetries make raw L2 especially suspect; path norms, spectral quantities, Fisher/KL distances in function space, or NTK-local coordinates may be better candidates.</p>
    </div>
    <div class="panel wide">
      <h3>Current Verdict</h3>
      <p class="callout warning">This sweep does not yet support the strong claim that weight decay provides a clean Lagrange-multiplier definition of capacity. Data and width effects are much larger than the direct <code>lambda</code> effect, and the regularization gains are often at the scale of seed noise.</p>
      <p class="note">That is still useful: it says the next experiment should test the capacity hypothesis more directly, with a wider norm/regularization path, an optimizer/objective that matches the Lagrangian being interpreted, and a predeclared rule for when a regularization ridge counts as real.</p>
    </div>
    <div class="panel">
      <h3>Regularization Audit Numbers</h3>
      <ul class="metric-list">{reg_audit_bits}</ul>
      <p class="note">The current maximum positive weight decay is not "super-heavy" L2. It is a local probe, so a missing capacity collapse here should be read as "not tested yet," not as evidence against the limiting idea.</p>
    </div>
    <div class="panel">
      <h3>Run Metadata</h3>
      <ul class="metric-list">{metadata_bits}</ul>
    </div>
    <div class="panel wide">
      <h3>How To Read The Plots</h3>
      <ul class="plot-list">
        <li><b>Dense MNIST phase surface</b>: the most literal plot. Each heatmap fixes weight decay and shows validation cross-entropy over data size and width. If the panels look similar across <code>lambda</code>, that is a result.</li>
        <li><b>Regularization signal audit</b>: the key honesty check for the L2-capacity hypothesis. It asks whether tuning weight decay beats <code>lambda=0</code> by more than seed noise and whether <code>h_lambda</code> is on the same scale as data/width responses.</li>
        <li><b>Envelope susceptibilities</b>: useful but post-hoc. It first chooses the best <code>lambda</code> for each <code>(N,W)</code>, then differentiates the lower envelope. This describes what data or width still buys after regularization tuning; it is not independent evidence for a regularization phase boundary.</li>
        <li><b>Raw lambda finite differences</b>: low-confidence diagnostic. They are included to show the failure mode: second derivatives of a weak signal can produce attractive-looking bumps that should not be promoted into discoveries.</li>
        <li><b>3D residual view</b>: visualization only. The default residual mode subtracts additive main effects so small interactions can be seen, but it is not a literal loss surface.</li>
      </ul>
    </div>
  </div>
</section>
<section id="spec" class="tab">
  <div class="panel">{spec_html}</div>
</section>
<section id="mnist" class="tab">
  <div class="grid">
    <div class="panel wide">
      <h3>Dense MNIST Phase Surface</h3>
      <p class="note">Each small heatmap fixes weight decay <code>lambda</code>. The horizontal axis is data size <code>N</code>, the vertical axis is hidden width <code>W</code>, and color is validation cross-entropy after scratch retraining. Lower values are better. The main visible pattern is ordinary learning: more data and more width usually reduce risk. The panels being very similar across <code>lambda</code> is exactly why the regularization-capacity claim is weak in this run.</p>
      <img class="figure" src="{fig_html['neural_phase']}">
    </div>
    <div class="panel wide">
      <h3>3D Iso-Loss Cutouts</h3>
      <p class="note">This interactive view puts <code>N</code>, <code>W</code>, and positive <code>lambda</code> on normalized log axes. Start with the <b>Raw iso-loss R</b> button if you want the faithful smoothed validation-CE surface. The default <b>Balanced curvature residual</b> mode subtracts additive axis trends, making small interactions visible only after the dominant data/width effects have been removed. Treat that residual as a visualization normalization, not as evidence for a new thermodynamic state variable.</p>
      <iframe class="interactive-figure" src="{neural_phase_3d_src}" title="MNIST 3D interpolated iso-loss surfaces"></iframe>
    </div>
    <div class="panel wide">
      <h3>Loss Decomposition</h3>
      <p class="note">The blue bar is the real validation cross-entropy at the largest data size. The orange bar is the measured diagnostic term <code>lambda ||w||^2 / 2</code> for the trained model. This is the plot closest to the intended capacity story: if L2 were made extremely strong, this term should force a small-norm/low-capacity model. But this run uses AdamW, where weight decay is decoupled from the gradient of an additive penalty, so the orange bar is a sanity check rather than the exact optimized Lagrangian.</p>
      <img class="figure" src="{fig_html['neural_loss_decomp']}">
    </div>
    <div class="panel">
      <h3>Best Neural Cells</h3>
      {small_table(neural_final, ['n_train','width','lr','weight_decay','seed_count','epochs','val_loss','val_loss_std','val_acc'], n=10)}
    </div>
    <div class="panel">
      <h3>Best Local Responses</h3>
      {small_table(neural_local, ['n_train','width','weight_decay','val_loss','h_data_per_doubling','h_capacity_per_doubling','h_regularization_per_decade'], n=10)}
    </div>
  </div>
</section>
<section id="suscept" class="tab">
  <div class="grid">
    <div class="panel wide">
      <h3>Current Read</h3>
      <p class="callout warning">The clean capacity result is not here yet. The lower envelope over <code>lambda</code> still has meaningful data/width responses, but the direct weight-decay response is too small and noisy to justify a raw regularization-specific-heat story.</p>
    </div>
    <div class="panel wide">
      <h3>Envelope Susceptibilities</h3>
      <p class="note">This plot uses <code>R*(N,W)=min_lambda R(N,W,lambda)</code>. Top left is the best observed validation loss after choosing the best weight decay at each <code>(N,W)</code>. Top right estimates the risk decrease per data doubling. Bottom left estimates the risk decrease per parameter/width doubling. Bottom right assigns a thresholded descriptive label. Because <code>lambda</code> is optimized before differentiating, this is a post-hoc summary of the best surface, not a proof that regularization defines capacity.</p>
      <img class="figure" src="{fig_html['neural_envelope_suscept']}">
    </div>
    <div class="panel wide">
      <h3>Regularization Signal Audit</h3>
      <p class="note">This is the main guardrail against cherry-picked structure. The panels ask four plain questions: which <code>lambda</code> wins, how much it beats <code>lambda=0</code>, whether that gain is bigger than seed standard error, and how the response scale compares with data and width. The answer is sobering: tuning weight decay helps a little in some cells, but the effect is usually tiny relative to the data/width directions.</p>
      <img class="figure" src="{fig_html['neural_reg_audit']}">
    </div>
    <div class="panel wide">
      <h3>Raw λ Finite Differences</h3>
      <p class="note">The left panel is the median local response to a decade change in weight decay. The right panel is the finite-difference second derivative of that response. These curves are useful mainly as a warning: the signal is weak, sign-sensitive, and edge-sensitive, so attractive-looking bumps should not be read as discovered phase boundaries.</p>
      <img class="figure" src="{fig_html['neural_specific_heat']}">
    </div>
  </div>
</section>
<section id="scale" class="tab">
  <div class="grid">
    <div class="panel">
      <h3>Toward 10^5 Runs</h3>
      <p class="note">The next experiment should be predeclared around the capacity hypothesis: choose whether <code>lambda</code> is an additive L2 penalty, decoupled AdamW decay, or an explicit norm constraint; then sweep far enough that the small-norm collapse is actually in range.</p>
      <p class="note">A useful remote run would be sharded scratch retraining with clustered log grids: 3-5 seeds, 20-30 data sizes, 20-30 widths, 15-25 regularization values, a few optimizer settings, and either sparse time checkpoints or an early-stop envelope. Start with 10^3-10^4 runs to verify derivative stability before spending the full 10^5-run budget.</p>
    </div>
    <div class="panel">
      <h3>Free Energy</h3>
      <p class="note">A literal free energy needs a measure over weights and an integral over basins. This report estimates an effective response potential, validation risk <code>R(theta)</code>. That is useful for susceptibilities, but it is not yet the partition-function free energy.</p>
    </div>
  </div>
</section>
<section id="caveats" class="tab">
  <div class="panel">
    <h3>What This Does and Does Not Show</h3>
    <p>The Legendre/susceptibility language is exact only after the coarse observable and conjugate family have been fixed. For neural networks, it is an empirical diagnostic: useful if it compresses training behavior and predicts transitions, but not evidence by itself that the network has a unique thermodynamic state variable.</p>
    <p>This run does not test literal "super-heavy L2 implies zero capacity." The largest positive weight decay is modest, training uses AdamW rather than an additive L2 objective, and there is no explicit norm-budget solve. The report should therefore be read as a failed/negative local signal for the regularization phase-boundary story, not as a final verdict on constrained-capacity definitions.</p>
    <p>The dense local run averages three scratch seeds per condition under fixed 3-epoch exposure. Treat any ridges as candidates. The next meaningful check is a predeclared constrained or additive-L2 sweep, quasi-equilibrium early stopping, and richer observables such as empirical NTK drift, representation CKA, or activation-spectrum order parameters.</p>
  </div>
</section>
</main>
<script>
const buttons = Array.from(document.querySelectorAll("nav button"));
const tabs = Array.from(document.querySelectorAll("section.tab"));
buttons.forEach(button => {{
  button.addEventListener("click", () => {{
    buttons.forEach(b => b.classList.remove("active"));
    tabs.forEach(t => t.classList.remove("active"));
    button.classList.add("active");
    document.getElementById(button.dataset.tab).classList.add("active");
  }});
}});
</script>
</body>
</html>"""


def main() -> None:
    figures = {
        "neural_phase": plot_neural_phase(),
        "neural_phase_3d": plot_neural_phase_3d_isosurfaces(),
        "neural_envelope_suscept": plot_neural_envelope_susceptibilities(),
        "neural_reg_audit": plot_neural_regularization_audit(),
        "neural_specific_heat": plot_neural_specific_heat_curves(),
        "neural_loss_decomp": plot_neural_loss_decomposition(),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_html(figures), encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
