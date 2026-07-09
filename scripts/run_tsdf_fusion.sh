#!/usr/bin/env bash
set -euo pipefail

python src/fusion/posedepthfuse.py --config configs/tsdf/fuse.template.yml

