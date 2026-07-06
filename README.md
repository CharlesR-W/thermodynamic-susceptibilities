# Thermodynamic Susceptibilities for Learning

This repository contains a small MNIST pilot for reading learning systems through
local response fields rather than only through final validation loss.

The current report asks a narrow question: if we perturb data size, hidden width,
or weight decay, where does validation risk move sharply? The analogy is to a
specific heat curve: the interesting signal is not just the loss surface, but
where the usefulness of a control changes quickly.

## Current Read

- The cleanest object in this run is the lower envelope over weight decay,
  `R*(N,W)=min_lambda R(N,W,lambda)`.
- Data and capacity responses remain visible on that envelope.
- The direct weight-decay response is much smaller and often comparable to seed
  noise, so the report treats raw regularization "specific heat" as a diagnostic
  audit rather than the main result.
- The result is a finite-sweep design probe, not a claim that the network has a
  literal thermodynamic state variable.

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
