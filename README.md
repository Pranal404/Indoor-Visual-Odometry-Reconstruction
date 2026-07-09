# Classical RGB-D Visual Odometry and TSDF Reconstruction

This repository contains the classical part of my indoor visual odometry and
3D reconstruction work. It focuses on a transparent RGB-D Pipeline:

1. Read a recorded RGB-D sequence;
2. Estimate camera motion with classical feature tracking and PnP;
3. Export camera poses;
4. Evaluate trajectory stability without assuming ground truth;
5. Fuse RGB-D frames into a TSDF mesh with Open3D.

The repository is intentionally limited to the classical implementation. Raw
camera recordings, trained models, large mesh outputs and private experiment
Folders are not included.

## Pipeline

```text
RGB images + depth images
        |
        v
Four-frame classical RGB-D VO
        |
        v
Camera pose dump + run log
        |
        v
Local pose metrics and trajectory plots
        |
        v
TSDF fusion using colour, depth and estimated poses
        |
        v
Mesh output and basic mesh statistics
```

## Repository Layout

```text
configs/
  classical/          Classical VO and pose-metric templates
  tsdf/               TSDF fusion and TSDF metric templates

src/
  classical/          Offline classical RGB-D VO runner and metrics
  fusion/             Pose-depth TSDF fusion script
  tdreconstruct/      Local Open3D TSDF wrapper
  metrics/            Small TSDF metric helper

scripts/              Example run commands
examples/             Placeholder sequence structure
figures/              Lightweight example figures
outputs_sample/       Lightweight example metric summaries
docs/                 Method and reproducibility notes
```

## Input Data Format

Place your own sequence outside the repository or under an ignored `data/`
folder:

```text
data/sequence01/
  rgb/
    color_0000.png
    color_0001.png
  depth/
    depth_0000.png
    depth_0001.png
```

Depth PNG files are expected to be `uint16` millimetres by default. This can be
changed in the config using `depth.scale`.

## Quick Start

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit the paths and camera intrinsics in:

```text
configs/classical/classicvo.template.yml
configs/classical/classicmetrics.template.yml
configs/tsdf/fuse.template.yml
configs/tsdf/tsdfmetrics.template.yml
```

Run the complete classical route:

```bash
bash scripts/run_classical_pipeline.sh
```

Or run the stages separately:

```bash
python src/classical/classicrunner.py configs/classical/classicvo.template.yml
python src/classical/classicmetrics.py configs/classical/classicmetrics.template.yml
python src/fusion/posedepthfuse.py --config configs/tsdf/fuse.template.yml
python src/metrics/tsdfmetrics.py configs/tsdf/tsdfmetrics.template.yml
```

## Included Sample Evidence

The repository includes a few lightweight outputs so the project can be
reviewed without downloading the original camera recordings:

- `figures/classical/classicseq05stability.png`
- `figures/classical/classicseq06stability.png`
- `figures/tsdf/tsdfmetriccomparison.png`
- `outputs_sample/classical_summarytable.csv`
- `outputs_sample/classical_summary.json`

## What Is Not Included

This repository does not include:

- raw Astra or stereo recordings;
- private thesis outputs;
- learning-based pose, stereo, NeRF or Gaussian Splatting code;
- large meshes, videos or checkpoint files.

Those belong in separate repositories or should be installed from their
original upstream projects.

