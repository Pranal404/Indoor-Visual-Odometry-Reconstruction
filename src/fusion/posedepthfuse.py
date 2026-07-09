import argparse  # argparse reads command-line arguments like --config.
import os  # os provides file-path and folder utilities.
import sys  # sys lets this script import the local src package when run directly.
from pathlib import Path  # Path gives safer object-style filesystem paths.

import cv2  # cv2 reads color images and uint16 depth PNG files.
import numpy as np  # numpy stores poses, intrinsics, and depth maps.
import yaml  # yaml reads the small config file for this script.


# Repo root is two levels above this file: src/fusion -> src -> project root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Add the repository root so "src..." imports work when this file is run directly.
sys.path.insert(0, str(REPO_ROOT))

# The local TSDF wrapper lives inside this repository.
from src.tdreconstruct.tsdfopentd import TSDFOpen3D


def LoadConfig(config_path):
    # def starts a function definition.
    # LoadConfig is CamelCase because this project convention avoids underscores in functions.
    # config_path is the input YAML path.
    with open(config_path, "r") as f:
        # with automatically closes the file after reading.
        # open(..., "r") opens the file in read mode.
        return yaml.safe_load(f)
        # yaml.safe_load converts YAML text into a Python dictionary.
        # return sends that dictionary back to the caller.


def EnsureFolder(path):
    # This function creates a folder if it does not already exist.
    os.makedirs(path, exist_ok=True)
    # os.makedirs creates parent folders too.
    # exist_ok=True means do not crash if the folder already exists.


def ReadPoseDump(path):
    # This function loads the compressed camera-pose dump file.
    dump = np.load(path, allow_pickle=True)
    # np.load reads .npz files.
    # allow_pickle=True is needed because image_names were saved as Python objects.

    names = [str(x) for x in dump["image_names"]]
    # dump["image_names"] is the ordered filename list from the pose runner.
    # str(x) makes sure every entry is a normal Python string.

    twcs = dump["Twc"].astype(np.float64)
    # Twc means camera-to-world pose.
    # astype(np.float64) gives stable numeric precision for TSDF integration.

    k = dump["K"].astype(np.float64)
    # K is the 3x3 camera intrinsic matrix at the same resolution as the pose dump.

    width = int(dump["W"])
    # int converts NumPy scalar width into a normal Python integer.

    height = int(dump["H"])
    # height is the image height expected by the pose dump and all matching depth maps.

    return names, twcs, k, width, height
    # Return all pose dump fields needed by the fusion loop.


def FindDepthPath(depth_dir, image_name):
    # This function finds the depth file matching one saved image name.
    direct_path = os.path.join(depth_dir, image_name)
    # First try exact same filename, useful for uint16 PNG depth.

    if os.path.isfile(direct_path):
        # os.path.isfile returns True when the exact file exists.
        return direct_path
        # Return immediately if exact filename exists.

    stem = os.path.splitext(image_name)[0]
    # os.path.splitext splits "00012.png" into ("00012", ".png").
    # [0] keeps only the stem without extension.

    npy_path = os.path.join(depth_dir, stem + ".npy")
    # Try float depth stored as NumPy .npy.

    if os.path.isfile(npy_path):
        # If the .npy depth exists, use it.
        return npy_path

    png_path = os.path.join(depth_dir, stem + ".png")
    # Try uint16 PNG depth with the same stem.

    if os.path.isfile(png_path):
        # If the .png depth exists, use it.
        return png_path

    if stem.startswith("color_"):
        # Astra RGB files are often color_XXXX.png while depth is depth_XXXX.png.
        depth_stem = "depth_" + stem[len("color_"):]
        astra_png_path = os.path.join(depth_dir, depth_stem + ".png")
        astra_npy_path = os.path.join(depth_dir, depth_stem + ".npy")

        if os.path.isfile(astra_png_path):
            return astra_png_path

        if os.path.isfile(astra_npy_path):
            return astra_npy_path

    return None
    # None means there is no matching depth for this frame.


def LoadDepthMeters(path):
    # This function loads either .npy meter depth or uint16 PNG millimetre depth.
    ext = os.path.splitext(path)[1].lower()
    # ext stores the lowercase file extension, for example ".npy" or ".png".

    if ext == ".npy":
        # .npy depth files are expected to already be in meters.
        return np.load(path).astype(np.float32)
        # np.load reads the array, astype makes the output float32.

    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # cv2.IMREAD_UNCHANGED preserves uint16 depth values.

    if depth is None:
        # cv2.imread returns None when reading fails.
        raise RuntimeError(f"Could not read depth file: {path}")

    if depth.dtype == np.uint16:
        # uint16 PNG depth convention is millimetres.
        depth = depth.astype(np.float32) / 1000.0
        # Convert millimetres to meters.
    else:
        # Any other numeric image is treated as already numeric depth.
        depth = depth.astype(np.float32)

    return depth
    # Return depth in meters.


def ReadColor(path):
    # This function reads one RGB/left image for color TSDF fusion.
    color = cv2.imread(path, cv2.IMREAD_COLOR)
    # cv2 reads BGR format by default; TSDF wrapper already expects colorbgr.

    if color is None:
        # If reading failed, return None so the caller can skip or raise.
        return None

    return color
    # Return the BGR image array.


def ResizeToPoseSize(color, depth, width, height):
    # Resize loaded files to the exact image size used by the poses and K.
    target_shape = (int(height), int(width))
    resized_color = False
    resized_depth = False

    if color.shape[:2] != target_shape:
        color = cv2.resize(color, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        resized_color = True

    if depth.shape[:2] != target_shape:
        depth = cv2.resize(depth, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        resized_depth = True

    return color, depth.astype(np.float32), resized_color, resized_depth


def ScaleDepth(depth, depth_scale):
    # Scale valid metric depth without changing invalid zero/NaN/inf pixels.
    depth = np.asarray(depth, dtype=np.float32)

    if float(depth_scale) == 1.0:
        return depth

    scaled_depth = depth.copy()
    valid = np.isfinite(scaled_depth) & (scaled_depth > 0.0)
    scaled_depth[valid] *= float(depth_scale)
    return scaled_depth


def BuildValidDepth(depth, min_depth, max_depth, filter_depth_range):
    # This function builds a valid-depth mask and zeroes values Open3D cannot use.
    valid = np.isfinite(depth) & (depth > 0.0)
    # Open3D ignores zero depth. NaN, inf, and negative depths must be removed.

    if filter_depth_range:
        # Optionally keep only depths inside the configured metric range.
        valid = valid & (depth >= float(min_depth))
        valid = valid & (depth <= float(max_depth))

    clean_depth = np.where(valid, depth, 0.0).astype(np.float32)
    # np.where writes original depth where valid is True, otherwise 0.
    # Open3D ignores depth value 0.

    valid_ratio = float(np.mean(valid))
    # np.mean treats True as 1 and False as 0, giving the valid pixel ratio.

    return clean_depth, valid_ratio
    # Return cleaned depth and its valid ratio.


def FuseSequence(cfg):
    # This function performs the full TSDF fusion loop for one estimator output.
    pose_dump = cfg["input"]["pose_dump"]
    # Path to the exported pose dump.

    color_dir = cfg["input"]["color_dir"]
    # Folder containing RGB images or rectified left images.

    depth_dir = cfg["input"]["depth_dir"]
    # Folder containing estimator depth maps.

    resize_to_pose_size = bool(cfg["input"].get("resize_to_pose_size", True))
    # If True, resize color/depth files to posesdump W,H before fusion.

    out_mesh = cfg["output"]["mesh_path"]
    # Path where the final mesh should be saved.

    min_depth = float(cfg["fusion"].get("min_depth", 0.3))
    # Minimum accepted depth in meters.

    max_depth = float(cfg["fusion"].get("max_depth", 4.0))
    # Maximum accepted depth in meters.

    min_valid_ratio = float(cfg["fusion"].get("min_valid_ratio", 0.01))
    # Frames with fewer valid pixels than this are skipped.

    fuse_all_frames = bool(cfg["fusion"].get("fuse_all_frames", False))
    # If True, do not apply depth-range filtering or valid-ratio frame skipping.

    filter_depth_range = bool(cfg["fusion"].get("filter_depth_range", True))
    # If False, keep all finite positive depth values instead of applying min/max depth.

    skip_low_valid_ratio = bool(cfg["fusion"].get("skip_low_valid_ratio", True))
    # If False, fuse frames even when their valid-depth ratio is very small.

    if fuse_all_frames:
        filter_depth_range = False
        skip_low_valid_ratio = False

    pose_scale = float(cfg["fusion"].get("pose_scale", 1.0))
    # Optional multiplier for estimated translation scale.

    depth_scale = float(cfg["fusion"].get("depth_scale", 1.0))
    # Optional multiplier for loaded metric depth before filtering and TSDF fusion.

    unfiltered_depth_trunc = float(cfg["fusion"].get("unfiltered_depth_trunc", 100.0))
    # Open3D still needs a finite truncation value when depth-range filtering is off.

    tsdf_min_depth = min_depth if filter_depth_range else 0.0
    # TSDFOpen3D applies the same minimum-depth rule internally.

    tsdf_depth_trunc = max_depth if filter_depth_range else unfiltered_depth_trunc
    # TSDFOpen3D applies the same maximum-depth rule internally.

    voxel_length = float(cfg["tsdf"].get("voxel_length", 0.015))
    # TSDF voxel size in meters.

    sdf_trunc = float(cfg["tsdf"].get("sdf_trunc", 0.05))
    # Truncation distance around each surface.

    min_triangles = int(cfg["tsdf"].get("min_triangles", 100))
    # Small connected mesh components below this count are removed by your wrapper.

    names, twcs, k, width, height = ReadPoseDump(pose_dump)
    # Load pose names, poses, intrinsics, and expected resolution.

    EnsureFolder(os.path.dirname(out_mesh))
    # Ensure the output mesh folder exists.

    tsdf = TSDFOpen3D(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        depth_trunc=tsdf_depth_trunc,
        min_depth=tsdf_min_depth,
        color_type="rgb8",
        conf_thresh=0.0,
    )
    # Create the Open3D scalable TSDF volume.

    fused = 0
    # Count frames actually fused.

    skipped = 0
    # Count frames skipped for missing files or bad valid ratio.

    skipped_missing = 0
    # Count frames skipped because color or depth files are missing/unreadable.

    skipped_low_valid = 0
    # Count frames skipped only by valid-ratio filtering.

    resized_color_frames = 0
    # Count color frames resized to pose dump resolution.

    resized_depth_frames = 0
    # Count depth frames resized to pose dump resolution.

    print("Pose frames          :", len(names))
    print("Pose image size      :", (width, height))
    print("Resize to pose size  :", resize_to_pose_size)
    print("Depth range filtering:", filter_depth_range)
    print("Valid-ratio skipping :", skip_low_valid_ratio)
    print("Depth scale          :", depth_scale)
    print("Pose scale           :", pose_scale)
    print("TSDF depth trunc     :", tsdf_depth_trunc)

    for index, name in enumerate(names):
        # for loops over every processed frame in exact pose order.
        # enumerate gives both index and filename.
        color_path = os.path.join(color_dir, name)
        # Build image path using the filename saved in the pose dump.

        depth_path = FindDepthPath(depth_dir, name)
        # Find matching estimator depth by same filename stem.

        if depth_path is None or not os.path.isfile(color_path):
            # Skip if either depth or color image is missing.
            skipped += 1
            skipped_missing += 1
            continue

        color = ReadColor(color_path)
        # Load BGR image.

        if color is None:
            # Skip unreadable images.
            skipped += 1
            skipped_missing += 1
            continue

        depth = LoadDepthMeters(depth_path)
        # Load depth in meters.

        if resize_to_pose_size:
            color, depth, resized_color, resized_depth = ResizeToPoseSize(color, depth, width, height)
            resized_color_frames += int(resized_color)
            resized_depth_frames += int(resized_depth)

        if color.shape[:2] != (height, width):
            # color.shape[:2] is (H, W), matching the pose dump dimensions.
            raise RuntimeError(f"Image size mismatch for {name}: {color.shape[:2]} expected {(height, width)}")

        if depth.shape[:2] != (height, width):
            # Depth pixels must align with the color pixels and K matrix.
            raise RuntimeError(f"Depth size mismatch for {name}: {depth.shape[:2]} expected {(height, width)}")

        depth = ScaleDepth(depth, depth_scale)
        # Apply experiment-time depth scaling before min/max depth filtering.

        depth, valid_ratio = BuildValidDepth(depth, min_depth, max_depth, filter_depth_range)
        # Remove invalid depths and measure how much useful depth remains.

        if skip_low_valid_ratio and valid_ratio < min_valid_ratio:
            # Skip frames with too little usable depth.
            skipped += 1
            skipped_low_valid += 1
            continue

        twc = twcs[index].copy()
        # Copy pose to avoid modifying the loaded pose dump.

        twc[:3, 3] *= pose_scale
        # Scale only translation, not rotation.

        tsdf.Integrate(depthm=depth, kip=k, Twc=twc, colorbgr=color, conf=None)
        # Fuse this RGB-D frame into the TSDF volume.

        fused += 1
        # Increase fused counter.

        if fused % 20 == 0:
            # Print progress every 20 fused frames.
            print(f"[TSDF] fused={fused} frame={name} valid={100.0 * valid_ratio:.1f}%")

    if fused == 0:
        # If nothing fused, paths or filtering are wrong.
        raise RuntimeError("No frames fused. Check pose dump, color path, depth path, and depth range.")

    mesh = tsdf.Extractmesh(mintriangles=min_triangles)
    # Extract triangle mesh from the TSDF volume.

    tsdf.savemesh(out_mesh, mesh)
    # Save final mesh as .ply.

    print("Fused frames :", fused)
    # Print total fused frames.

    print("Skipped      :", skipped)
    # Print skipped frames.

    print("Missing files :", skipped_missing)
    # Print frames skipped because files were unavailable.

    print("Low valid     :", skipped_low_valid)
    # Print frames skipped by valid-ratio filtering.

    print("Resized color :", resized_color_frames)
    # Print color frames resized to pose dump resolution.

    print("Resized depth :", resized_depth_frames)
    # Print depth frames resized to pose dump resolution.

    print("Saved mesh   :", out_mesh)
    # Print output mesh path.


def Main():
    # Main is the command-line entry point.
    parser = argparse.ArgumentParser()
    # ArgumentParser defines accepted command-line arguments.

    parser.add_argument("--config", required=True, type=str)
    # --config is required and must be a string path.

    args = parser.parse_args()
    # parse_args reads the actual command-line values.

    cfg = LoadConfig(args.config)
    # Load YAML config into a dictionary.

    FuseSequence(cfg)
    # Run the fusion pipeline.


if __name__ == "__main__":
    # This condition is True only when running this file directly.
    Main()
    # Call the entry point.
