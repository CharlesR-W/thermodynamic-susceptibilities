# Thermodynamic Susceptibilities for Learning

This repository contains a small MNIST pilot for asking whether neural-network
capacity can be read through Lagrange-multiplier-like response fields rather
than only through final validation loss.

The motivating hypothesis is that a norm or regularization constraint should
have a shadow price: relaxing the constraint should buy down risk, and extremely
heavy L2 regularization should eventually collapse task capacity. This run is a
local MNIST probe of that idea, not the final constrained-capacity experiment.
The broader goal is to compare different regularizers as softened constraints in
an ensemble/maxent picture, where norm, rank, sparsity, noise, and architecture
bottlenecks can all be assigned conjugate fields.

## Current Read

- Data and width responses are visible and interpretable in this sweep.
- The direct weight-decay response is much smaller and often comparable to seed
  noise, so the report treats raw regularization "specific heat" as a cautionary
  diagnostic rather than as evidence for a phase boundary.
- The lower envelope `R*(N,W)=min_lambda R(N,W,lambda)` is useful as a post-hoc
  descriptive surface, but not independent evidence that regularization defines
  capacity.
- The current result is therefore negative/inconclusive for the strong
  Lagrange-multiplier capacity story. A real test needs additive L2 or explicit
  norm constraints, a wider regularization path, and predeclared thresholds.

## Open The Report

The viewable GitHub Pages version is:

```text
https://charlesr-w.github.io/thermodynamic-susceptibilities/
```

The repository artifact is:

```text
reports/thermo_phase_report.html
```

Open it from the repository root so the embedded 3D phase view can load its
relative file at `results/figures/neural_phase_3d_isosurfaces.html`.

## Reproduce

```bash
python scripts/run_sweeps.py --profile dense-local --skip-kernel
python scripts/build_report.py
```

The command above uses cached MNIST when available, trains 539 MNIST conditions
with three scratch seeds per condition, skips the optional kernel calibration,
and then rebuilds the HTML report. The Fourier-kernel calibration code is still
available for follow-up checks, but the sendable report is intentionally
MNIST-focused and omits the old kernel results tab.

## Layout

- `docs/spec.md` - embedded theory/specification and interpretation rules.
- `scripts/run_sweeps.py` - data generation for neural and optional kernel sweeps.
- `scripts/build_report.py` - rebuilds the HTML report from saved tables.
- `results/tables/` - CSV summaries used by the report.
- `results/figures/` - generated figures, including the external 3D view.
- `reports/thermo_phase_report.html` - main report.
