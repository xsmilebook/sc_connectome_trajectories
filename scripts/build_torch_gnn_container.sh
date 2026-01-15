#!/bin/bash
set -euo pipefail

module load singularity

eval "$(python3 -m scripts.render_paths \
  --set CONTAINER=local.containers.torch_gnn \
  --resolve)"

mkdir -p "$(dirname "$CONTAINER")"
singularity build --remote "$CONTAINER" scripts/containers/torch_gnn.def
