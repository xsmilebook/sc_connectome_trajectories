# Plan

Implement tiered CLG-ODE training so subjects with 1/2/3+ sessions are all usable, persist run-level records for reproducibility, and generate a strict tier availability report (SC+morph file existence) with documentation updates.

## Scope
- In: Tiered objective (manifold/velocity/acceleration), default denoising, run directory naming (timestamp+jobid), strict tier stats + docs report, and documentation/progress/session updates.
- Out: Job submission actions, dataset regeneration, changes under `data/` or committed runtime artifacts under `outputs/`, and unrelated refactors.

## Action items
[x] Extend dataset handling to include subjects with 1 session (`min_length=1`) while still requiring strict SC+morph file existence.
[x] Implement tiered training objectives: `L_manifold` (Tier 3/2/1), `L_vel` (Tier 2/1), `L_acc` (Tier 1) with a warmup schedule and recommended default weights.
[x] Enable default denoising augmentation (morph noise + SC positive-edge dropout) and keep `s_mean` enabled by default.
[x] Add per-run directory naming under `--results_dir/runs/<timestamp>_job<jobid>/` and persist `args.json`, `run_meta.json`, and per-epoch `metrics.csv`.
[x] Add `python -m scripts.report_clg_ode_tiers` to compute strict tier availability and write a report to `docs/reports/`.
[x] Update documentation (`README.md`, `docs/workflow.md`, `docs/methods.md`) plus `PROGRESS.md` and `docs/sessions/` to reflect tiered training and run tracking.
[ ] Run a quick smoke validation in the Singularity container (single GPU and 4-GPU `torchrun`) without submitting jobs from the assistant.
[ ] Commit the change set with a clear message.

## Open questions
- None.
