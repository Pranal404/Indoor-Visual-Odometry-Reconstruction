#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_classical_vo.sh
bash scripts/run_classical_metrics.sh
bash scripts/run_tsdf_fusion.sh
bash scripts/run_tsdf_metrics.sh

