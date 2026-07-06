# Thermodynamics Experiment Project

This repo is a fresh experiment scaffold for phenomenological thermodynamics of
deep learning. Keep the project report-driven: the durable artifact is
`reports/thermo_phase_report.html`, with `docs/spec.md` embedded into the report.

Local conventions:

- Use the cached raw MNIST IDX files from sibling projects when available; do not
  download datasets unless the user explicitly asks for that.
- Treat Legendre-dual language as exact only after a coarse observable and
  conjugate family have been fixed. For neural nets, report dual quantities as
  empirical finite-difference diagnostics.
- Generated result tables live in `results/tables/`; generated figures live in
  `results/figures/`.
- Keep sweeps CPU-friendly by default. Confirm before scaling to paid or long
  remote compute.
