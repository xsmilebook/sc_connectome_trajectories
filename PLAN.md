# Plan

Draft a CLG-ODE implementation and execution plan that matches the locked specification, then define the exact steps for submitting a GPU job on the cluster. The plan focuses on model/data deltas, topology conditioning, and a Slurm+Singularity runbook.

## Scope
- In: CLG-ODE model/data/training updates, topology conditioning features, SC/morph preprocessing rules, Slurm submission script, and runbook docs.
- Out: New datasets, container build steps, changes under `data/` or `outputs/`, and unrelated refactors.

## Action items
[ ] Compare current CLG-ODE code with `docs/reports/implementation_specification.md` and enumerate required deltas (time definition, covariates, losses, topology usage).
[ ] Implement SC preprocessing in the data layer: symmetrize, zero diagonal, log1p transform; compute global strength covariates `s` (and optional `s_mean`) from raw weights.
[ ] Add train-only morphology Z-score normalization at ROI-metric granularity, with optional ICV/TIV adjustment when available.
[ ] Build topology conditioning features per spec (ECC vector from log1p weights, quantile thresholds, triangle counts) and inject into the ODE dynamics; keep topology out of the training loss.
[ ] Switch to delta-time integration (age - age0), inject age0/sex/site/strength/topology covariates, and implement multi-start forecasting pair sampling (70% adjacent, 30% random).
[ ] Replace decoder/loss with edge-existence BCE and positive-edge Huber on log1p weights; disable hard sparsification/topk pruning.
[ ] Add a Slurm submission script that runs `python -m scripts.train_clg_ode` on `q_ai4` with `--gres=gpu:1`, and reference the Singularity image under `data/external/containers/` via `configs/paths.yaml`.
[ ] Add validation steps (`python -m pytest`, CUDA availability check in-container) and document expected outputs in `outputs/logs/` and `outputs/results/`.
[ ] Update `README.md`, `docs/workflow.md`, `PROGRESS.md`, and `docs/sessions/` to reflect the CLG-ODE execution workflow and constraints.

## Open questions
- Confirm the exact Singularity image filename to store under `data/external/containers/`.
- None (s_mean defaults to enabled unless explicitly disabled).
