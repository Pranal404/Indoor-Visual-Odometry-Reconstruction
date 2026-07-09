# Method Notes

## Classical Visual Odometry

The classical route uses an RGB-D sequence. Features are detected in the first
frame of a four-frame window and tracked through the following frames with
optical flow. The source-frame depth lifts 2D feature locations into 3D. The
3D points and tracked 2D image positions are then used by PnP-RANSAC to
estimate relative camera motion.

The runner stores one camera pose for every processed frame. When blur,
tracking, valid depth or PnP fails, the previous pose is repeated. This makes
the exported pose sequence complete while still preserving failure evidence in
the run log.

## TSDF Fusion

The TSDF stage takes colour images, metric depth images, camera intrinsics and
camera-to-world poses. Each valid RGB-D frame is integrated into an Open3D
scalable TSDF volume. The final mesh is extracted after all accepted frames
have been fused.

This stage depends strongly on pose quality. If many poses are repeated or
locally unstable, the TSDF mesh can show smearing, duplicated surfaces or
missing structure.

