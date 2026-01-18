# PROGRESS.md


## Current focus
- 2026-01-16: Validate CLG-ODE loss stabilization changes (topo normalization + GradNorm + warmup).

## Completed
- 2026-01-12: Drafted the repo-structure refactor plan in `PLAN.md`.
- 2026-01-12: Tagged and backed up the original `src/` repo as `src-pre-root-migration-20260112.bundle`.
- 2026-01-12: Initialized a new root Git repo and committed root docs + current `src/` snapshot.
- 2026-01-12: Removed nested `src/.git` so `src/` is now a normal directory in the root repo.
- 2026-01-12: Reorganized runtime data into the single-dataset layout (`data/interim/`, `data/processed/`) and moved figures/results under `outputs/`.
- 2026-01-12: Imported the full legacy `src` commit history into the root repo (rewritten under the `src/` prefix).
- 2026-01-12: Added root-level `scripts/` entrypoints and made `src/` importable as a package.
- 2026-01-12: Added `configs/paths.yaml` and switched defaults to config-managed paths; removed tracked runtime sublists from `src/`.
- 2026-01-13: Improved FreeSurfer rerun behavior (force rerun and incomplete-done detection).
- 2026-01-14: Updated the FreeSurfer rerun sublist to only include subjects without completion markers in logs.
- 2026-01-14: Required git commits for any code or documentation edits in `AGENTS.md`.
- 2026-01-14: Refined README and workflow documentation for morphology submission and summary tables.
- 2026-01-14: Locked CLG-ODE implementation details (topology conditioning, SC loss decomposition, covariates, and pair sampling).
- 2026-01-14: Locked ECC triangle counting and edge-score calibration for CLG-ODE.
- 2026-01-14: Linked cluster GPU usage guide across docs and agent rules.
- 2026-01-14: Replaced `PLAN.md` with the CLG-ODE implementation and execution plan.
- 2026-01-14: Implemented CLG-ODE spec updates (SC preprocessing, topology conditioning, delta-time sampling), added submission script, and refreshed docs/tests.
- 2026-01-14: Added the Singularity definition/build script and documented the container workflow.
- 2026-01-15: Updated the CLG-ODE submission script to enforce python3 and validate resolved paths.
- 2026-01-15: Added error handling around path resolution to avoid unbound variables in submission scripts.
- 2026-01-15: Updated CLG-ODE submission script to use absolute log paths and workdir.
- 2026-01-15: Moved `set -euo pipefail` below `#SBATCH` lines to keep log directives effective.
- 2026-01-15: Fixed `scripts.render_paths` usage to include all required path keys in one invocation.
- 2026-01-15: Added distributed training support (DDP) and updated the submission script to request 4 GPUs with a 48h limit.
- 2026-01-15: Enabled tiered training objectives for CLG-ODE so 1/2/3 timepoint subjects are all used (manifold/velocity/acceleration losses with warmup).
- 2026-01-15: Added per-run directory naming (`<timestamp>_job<jobid>`) and persisted `args.json`, `run_meta.json`, and `metrics.csv` for experiment tracking.
- 2026-01-15: Added a strict tier statistics script and generated an up-to-date dataset tier report under `docs/reports/`.
- 2026-01-15: Added test-only SC evaluation metrics (log-domain MSE/MAE/pearson + ECC similarity) and saved them as `test_sc_metrics.json`.
- 2026-01-15: Made test metrics deterministic (`t0→t1`) and aligned ECC with sparsity via top-k masking.
- 2026-01-15: Added `--cv_fold` support for per-fold CLG-ODE training as a single-GPU fallback.
- 2026-01-15: Updated submission script to default to single-GPU and support fold-wise Slurm array execution.
- 2026-01-15: Moved CLG-ODE logs into `outputs/logs/clg_ode/` and auto-select `master_port`.
- 2026-01-15: Grouped per-fold outputs under a shared `runs/<time>_job<array_job_id>/fold{0..4}` root.
- 2026-01-15: Updated CLG-ODE default training hyperparameters (longer epochs/patience, higher ODE steps, stronger vel/acc, small KL).
- 2026-01-15: Added an experimental topology loss based on Betti curves (β0/β1) to constrain connected components and cycles.
- 2026-01-15: Added masked and top-k Pearson correlations for SC evaluation to align sparse comparisons.
- 2026-01-15: Added a full-graph Pearson computed after top-k sparsification (`sc_log_pearson_sparse`).
- 2026-01-15: Applied top-k sparsification during training for `L_weight` and `L_topo`.
- 2026-01-16: Fixed test metric aggregation for new pearson metrics and added RUN_DATE/RUN_TIME for consistent run folders.
- 2026-01-16: Reduced default `lambda_topo` to 1e-3 to avoid loss explosion.
- 2026-01-16: Added fold length distribution logs and per-epoch vel/acc counts; randomize master_port to avoid collisions.
- 2026-01-16: Updated workflow/cluster docs with log directory, port conflict, and new metrics fields.
- 2026-01-16: Added topo loss normalization (quantile scale + log1p), GradNorm for manifold/topo, and 20% cosine warmup with new CLI flags and doc updates.
- 2026-01-16: Added a short CLG-ODE smoke submission script and documented the usage in workflow.
- 2026-01-16: Fixed GradNorm crash by ensuring topo sparsification preserves gradients and adding a requires-grad guard.
- 2026-01-16: Added a smoke submit wrapper and logged smoke outcomes in the model dev log.
- 2026-01-16: Removed time limit from full CLG-ODE submit script and logged the latest smoke result.
- 2026-01-16: Merged CLG-ODE 5-fold results across split runs and documented the analysis in `docs/reports/clg_ode_cv_merge_20260116.md`.
- 2026-01-16: Computed identity-mapping baseline metrics on the CLG-ODE test split and updated the report comparison.

## In progress

## Next up
- Push the new root repo to `origin` (may require forced push depending on the remote state).
- Decide whether to import full historical commits from the old `src/` repo into the root repo (optional; bundle available).

## Issues and solutions

| Issue | Solution | Date |
|-------|----------|------|

## Notes
