#!/usr/bin/env bash
set -euo pipefail

python src/metrics/tsdfmetrics.py configs/tsdf/tsdfmetrics.template.yml

