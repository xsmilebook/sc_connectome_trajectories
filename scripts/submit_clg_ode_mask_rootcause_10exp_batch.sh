#!/bin/bash

set -euo pipefail

# 10 experiments aligned with the debugging priority:
# 1) One diagnostic baseline with full mask logging.
# 2) Nine-grid pos_weight × lambda_density to test whether BCE is forcing density.
# 3) One focal run as an alternative to pos_weight tuning (submitted separately).

sbatch scripts/submit_clg_ode_maskdiag_fold0.sh
bash scripts/submit_clg_ode_posweight_density_grid_fold0_batch.sh
sbatch scripts/submit_clg_ode_focal_density_fold0.sh

