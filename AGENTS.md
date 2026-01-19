# AGENTS.md

Operational rules for AI-assisted work in `sc_connectome_trajectories`.
This repository treats `ARCHITECTURE.md` as the source of truth for the stable layout. Do not propose structural changes unless explicitly requested.

## Base workflow

1) Read `ARCHITECTURE.md` before planning or editing.
2) Execute changes only according to `PLAN.md` or explicit user instructions.
3) If a change affects usage or structure, update `README.md` and/or `docs/workflow.md`.
4) Update `PROGRESS.md` after completing a change set.
5) Log each session in `docs/sessions/` (date-stamped file).
6) If you modify code or documentation, commit the changes with git (write your own message).
7) If the `create-plan` skill is used, write the plan to the root `PLAN.md`.
8) All packages must be installed in a dedicated conda environment for this project; activate via `/GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate`.
9) Do not run deep learning training/inference on the host system (the OS/toolchain is too old and cannot reliably install PyTorch).
10) For deep learning training/inference, always use the container-based workflow documented in `docs/workflow.md` and `docs/cluster_gpu_usage.md` (Slurm + Singularity/Apptainer; no on-node env mutation).
11) Any `sbatch` submission or long-running container-based training job must be submitted by the user (the agent may only draft commands/scripts).

## Scope and constraints

- Default scope is documentation-only unless the user requests engineering changes.
- Do not modify runtime artifacts under `data/` or `outputs/`.
- Do not change directory names or structure unless explicitly requested.
- Keep diffs minimal; avoid refactors not tied to the request.

## Engineering conventions (when requested)

- For GPU work on the current cluster, follow `docs/cluster_gpu_usage.md` (Slurm + Singularity + no on-node env mutation).
- Use `python -m scripts.<entry>` for execution.
- Avoid ad-hoc `sys.path` hacks; if unavoidable, confine them to entry points and document why.
- All filesystem paths must come from `configs/`.

## Documentation conventions

- Use precise, scientific language.
- Write all `docs/` content in Chinese; keep filenames in English.
- Root-level Markdown files must be English only.
- Keep repository root folder names in English.
- sc_connectome_trajectories-specific notes live in `docs/workflow.md` and `configs/paths.yaml` (dataset section).
