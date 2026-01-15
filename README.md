# sc_connectome_trajectories

## Overview

This repository contains preprocessing and modeling workflows for ABCD structural connectome
trajectories. Paths are managed by `configs/paths.yaml`; FreeSurfer roots under `/GPFS/.../ABCD/...`
are intentionally kept as absolute paths.

## FreeSurfer (recon-all)

```bash
N=$(find /GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/raw_data/smri -type f -name "*_T1w.nii" | wc -l)
sbatch --array=1-${N} src/preprocess/freesurfer.sh
```

Rerun note:
- If a previous run was interrupted but a stale `scripts/recon-all.done` causes an incorrect skip,
  submit with `FREESURFER_FORCE=1` to force re-run.

## Morphology (Schaefer400)

Set atlas and output directories:

```bash
export ATLAS_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/raw/atlas"
export OUTPUT_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/processed/morphology/baselineYear1Arm1/SIEMENS"
```

Single site (array job):

```bash
sbatch --export=ALL,SUBJECTS_DIR="/GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/processed/freesurfer/baselineYear1Arm1/SIEMENS/site14",ATLAS_DIR="${ATLAS_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",SITE_NAME="site14" \
  --array=1-100 \
  src/preprocess/run_schaefer400_morphology.sh /path/to/sublist.txt
```

All sites (run from `src/preprocess` so relative paths resolve):

```bash
cd /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/src/preprocess
mkdir -p /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/processed/table/sublist_by_site/4YearFollowUpYArm1

bash submit_schaefer400_morphology_all_sites.sh \
  /GPFS/cuizaixu_lab_permanent/xuxiaoyu/ABCD/processed/freesurfer/4YearFollowUpYArm1/SIEMENS \
  /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/processed/table/sublist_by_site/4YearFollowUpYArm1 \
  ./run_schaefer400_morphology.sh
```

## Morphology summaries

```bash
python -m scripts.export_morphology_tables \
  --morph_root /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/processed/morphology \
  --subject_info_sc /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/processed/table/subject_info_sc.csv \
  --out_success /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/processed/table/subject_info_morphology_success.csv \
  --out_missing /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/processed/table/subject_info_sc_without_morphology.csv
```

## Model training

Vector LSTM baseline:

```bash
python -m scripts.train \
  --sc_dir /path/to/sc_connectome/schaefer400 \
  --results_dir /path/to/results
```

CLG-ODE:

```bash
python -m scripts.train_clg_ode \
  --sc_dir /path/to/sc_connectome/schaefer400 \
  --morph_root /path/to/morphology \
  --subject_info_csv /path/to/subject_info_sc.csv \
  --results_dir /path/to/results
```

Notes:
- CLG-ODE expects morphology CSVs named `Schaefer400_Morphology_<subid>.csv` under `--morph_root`.
- The trainer uses the `age`, `sex`, and `siteid` columns in `subject_info_sc.csv`, with delta-time integration and `age0` as a covariate.
- Topology features (ECC) and strength covariates (`s`, `s_mean`) are used for conditioning; topology is not part of the training loss.
- `s_mean` is enabled by default; disable via `--disable_s_mean` if needed.
- The default training objective supports subjects with 1/2/3 timepoints (tiered `L_manifold`, `L_vel`, `L_acc`) and writes a per-run directory under `--results_dir/runs/<timestamp>_job<jobid>/` containing `args.json`, `run_meta.json`, and `metrics.csv` for reproducibility.

Dataset tier report (strict SC+morph file existence, writes to `docs/reports/`):

```bash
python -m scripts.report_clg_ode_tiers
```

Cluster GPU usage:
- See `docs/cluster_gpu_usage.md` for the Slurm + Singularity GPU workflow and cluster-specific constraints.

CLG-ODE submission helper (uses paths from `configs/paths.yaml` and a Singularity image under `data/external/containers/`):

```bash
sbatch scripts/submit_clg_ode.sh
```

The submission script defaults to 4 GPUs on `q_ai4` and uses `torchrun` for distributed training.

Container build (torch+CUDA+GNN, remote Singularity build):

```bash
bash scripts/build_torch_gnn_container.sh
```
