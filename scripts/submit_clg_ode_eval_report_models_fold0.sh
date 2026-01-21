#!/bin/bash
set -euo pipefail

# Submit one evaluation job per model (fold0), producing `test_sc_metrics_ext.json`
# under each run directory. User must run this script on the login node.

models=(
  "clg_baseline_original"
  "clg_focal_ld01"
  "clg_fs_residual"
  "clg_fs_no_dt_gate"
  "clg_fs_no_Lsmall"
  "clg_fs_no_residual"
  "clg_fs_innov_default"
  "clg_d2prime_resume_c2_long_freeze"
)

for model in "${models[@]}"; do
  run_dir="outputs/results/clg_ode/runs/${model}/fold0"
  if [[ ! -d "$run_dir" ]]; then
    echo "Skip (missing): $run_dir" >&2
    continue
  fi
  echo "Submit eval: $run_dir"
  sbatch --export=ALL,EVAL_RUN_DIR="$run_dir" scripts/submit_clg_ode_eval.sh
done

