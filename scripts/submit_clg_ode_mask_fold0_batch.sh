#!/bin/bash

set -euo pipefail

sbatch scripts/submit_clg_ode_mask_fold0_a.sh
sbatch scripts/submit_clg_ode_mask_fold0_b.sh
sbatch scripts/submit_clg_ode_mask_fold0_c.sh
sbatch scripts/submit_clg_ode_mask_fold0_d.sh

