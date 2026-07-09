"""
Offline runner for the traditional 4-frame RGB-D visual odometry baseline.

Overall explanation:

This script runs the existing Classic `VO-04-4F-V2.py` logic on saved RGB and
depth image sequences instead of using the live Astra OpenNI stream. It is used
to produce a traditional visual odometry baseline from saved RGB-D data.

What this script does:

1. It reads a YAML config file.
2. It loads the original Classic VO script as a module.
3. It reads RGB and depth images from each configured sequence.
4. It feeds saved frames into the same four-frame Classic VO pipeline.
5. It estimates frame-to-frame pose using features, optical flow, depth, and PnP.
6. It saves `posesdump.npz` so the result can be measured and fused later.
7. It saves a trajectory PLY and a CSV run log for debugging.

Coding rules used in this file:

- Every custom function starts with a capital letter.
- Every custom function has an explanation docstring.
- Every executable line has a nearby explanation comment.
- Variable names avoid underscores unless Python or a library requires them.
"""

# Explanation: This line imports argparse so the script can read the config path from the command line.
import argparse
# Explanation: This line imports csv so the script can write per-frame run logs.
import csv
# Explanation: This line imports importlib tools so the original Classic script can be loaded from its file path.
import importlib.util
# Explanation: This line imports sys so import shims can be registered before loading the Classic script.
import sys
# Explanation: This line imports types so lightweight dependency shims can be created when offline-only imports are missing.
import types
# Explanation: This line imports Path for readable filesystem path handling.
from pathlib import Path

# Explanation: This line imports OpenCV for image reading and visual odometry operations.
import cv2
# Explanation: This line imports NumPy for pose matrices and numeric arrays.
import numpy as np
# Explanation: This line imports YAML so the script can read the config file.
import yaml


# Explanation: This line starts the LoadConfig function.
def LoadConfig(configpath):
    """
    Load a YAML config file.

    Step wise:
    1. Convert the config path into a Path object.
    2. Open the YAML file.
    3. Parse YAML into a Python dictionary.
    4. Return the dictionary.
    """
    # Explanation: This line converts the config path into a Path object.
    pathobject = Path(configpath).expanduser()
    # Explanation: This line opens the config file as UTF-8 text.
    with pathobject.open("r", encoding="utf-8") as fileobject:
        # Explanation: This line parses the YAML file into Python data.
        configdata = yaml.safe_load(fileobject)
    # Explanation: This line returns the parsed config.
    return configdata


# Explanation: This line starts the MakeFolder function.
def MakeFolder(folderpath):
    """
    Create a folder if it does not already exist.

    Step wise:
    1. Convert the folder path into a Path object.
    2. Create the folder and missing parents.
    3. Return the folder path.
    """
    # Explanation: This line converts the folder path into a Path object.
    pathobject = Path(folderpath).expanduser()
    # Explanation: This line creates the folder when needed.
    pathobject.mkdir(parents=True, exist_ok=True)
    # Explanation: This line returns the folder path.
    return pathobject


# Explanation: This line starts the ListImages function.
def ListImages(folderpath):
    """
    List image files in stable sorted order.

    Step wise:
    1. Convert the folder path into a Path object.
    2. Keep common image extensions.
    3. Sort the paths.
    4. Return the sorted list.
    """
    # Explanation: This line converts the folder path into a Path object.
    pathobject = Path(folderpath).expanduser()
    # Explanation: This line defines accepted image suffixes.
    suffixset = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    # Explanation: This line collects supported image files from the folder.
    filelist = [itempath for itempath in pathobject.iterdir() if itempath.suffix.lower() in suffixset]
    # Explanation: This line sorts files by filename.
    filelist.sort()
    # Explanation: This line returns the sorted image list.
    return filelist


# Explanation: This line starts the InstallShims function.
def InstallShims():
    """
    Install minimal import shims for live-camera libraries when running offline.

    Step wise:
    1. Create a dummy `openni` module when OpenNI is unavailable.
    2. Create a dummy `open3d` module when Open3D is unavailable.
    3. Register those shims in sys.modules before importing the Classic script.
    """
    # Explanation: This line checks whether the openni module is missing.
    if "openni" not in sys.modules:
        # Explanation: This line creates a lightweight openni module object.
        openmodule = types.ModuleType("openni")
        # Explanation: This line creates a lightweight openni2 object for imports.
        openmodule.openni2 = types.SimpleNamespace()
        # Explanation: This line registers the openni module shim.
        sys.modules["openni"] = openmodule
    # Explanation: This line checks whether the open3d module is missing.
    if "open3d" not in sys.modules:
        # Explanation: This line registers a lightweight open3d module shim.
        sys.modules["open3d"] = types.ModuleType("open3d")


# Explanation: This line starts the LoadClassicModule function.
def LoadClassicModule(scriptpath):
    """
    Load the original Classic VO script as a Python module.

    Step wise:
    1. Install offline import shims for camera-only dependencies.
    2. Build an import specification from the script path.
    3. Execute the script module without calling its main function.
    4. Return the loaded module.
    """
    # Explanation: This line installs shims before the Classic script imports OpenNI or Open3D.
    InstallShims()
    # Explanation: This line converts the Classic script path into a Path object.
    pathobject = Path(scriptpath).expanduser()
    # Explanation: This line creates a Python import specification from the file path.
    specobject = importlib.util.spec_from_file_location("classicvofourframe", pathobject)
    # Explanation: This line creates a module object from the import specification.
    moduleobject = importlib.util.module_from_spec(specobject)
    # Explanation: This line registers the module object under a stable name.
    sys.modules["classicvofourframe"] = moduleobject
    # Explanation: This line executes the original Classic script as an imported module.
    specobject.loader.exec_module(moduleobject)
    # Explanation: This line returns the loaded Classic module.
    return moduleobject


# Explanation: This line starts the ReadFrame function.
def ReadFrame(rgbpath, depthpath, depthscale):
    """
    Read one RGB-D frame from disk.

    Step wise:
    1. Read the RGB image in OpenCV BGR format.
    2. Read the depth image without changing its bit depth.
    3. Convert RGB to grayscale.
    4. Convert depth to metres.
    5. Resize depth if it does not match RGB size.
    6. Return grayscale, colour, and metric depth.
    """
    # Explanation: This line reads the colour image from disk.
    colorimage = cv2.imread(str(rgbpath), cv2.IMREAD_COLOR)
    # Explanation: This line checks whether RGB loading failed.
    if colorimage is None:
        # Explanation: This line raises a clear error for unreadable RGB files.
        raise FileNotFoundError(f"Could not read RGB image: {rgbpath}")
    # Explanation: This line reads the depth image from disk.
    rawdepth = cv2.imread(str(depthpath), cv2.IMREAD_UNCHANGED)
    # Explanation: This line checks whether depth loading failed.
    if rawdepth is None:
        # Explanation: This line raises a clear error for unreadable depth files.
        raise FileNotFoundError(f"Could not read depth image: {depthpath}")
    # Explanation: This line converts colour to grayscale for feature tracking.
    grayimage = cv2.cvtColor(colorimage, cv2.COLOR_BGR2GRAY)
    # Explanation: This line converts raw depth into float metres.
    depthimage = rawdepth.astype(np.float32) / float(depthscale)
    # Explanation: This line checks whether depth size differs from colour size.
    if depthimage.shape[:2] != grayimage.shape[:2]:
        # Explanation: This line resizes depth with nearest-neighbour interpolation.
        depthimage = cv2.resize(depthimage, (grayimage.shape[1], grayimage.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Explanation: This line returns the loaded frame tuple expected by the Classic VO logic.
    return grayimage, colorimage, depthimage


# Explanation: This line starts the SaveCsv function.
def SaveCsv(rowlist, outputpath):
    """
    Save a list of dictionaries into a CSV file.

    Step wise:
    1. Return early when there are no rows.
    2. Create the output folder.
    3. Use the first row keys as CSV columns.
    4. Write all rows.
    """
    # Explanation: This line checks whether there is anything to save.
    if len(rowlist) == 0:
        # Explanation: This line returns early for an empty table.
        return
    # Explanation: This line creates the output folder.
    MakeFolder(Path(outputpath).expanduser().parent)
    # Explanation: This line reads CSV column names from the first row.
    fieldnames = list(rowlist[0].keys())
    # Explanation: This line opens the CSV output file.
    with Path(outputpath).expanduser().open("w", newline="", encoding="utf-8") as fileobject:
        # Explanation: This line creates the CSV writer.
        writerobject = csv.DictWriter(fileobject, fieldnames=fieldnames)
        # Explanation: This line writes the CSV header row.
        writerobject.writeheader()
        # Explanation: This line writes every data row.
        writerobject.writerows(rowlist)


# Explanation: This line starts the SavePly function.
def SavePly(centerarray, outputpath):
    """
    Save camera centers as a simple trajectory PLY.

    Step wise:
    1. Convert centers into a NumPy array.
    2. Write PLY vertices for camera centers.
    3. Write PLY edges between consecutive centers.
    """
    # Explanation: This line converts camera centers into a numeric array.
    centerarray = np.asarray(centerarray, dtype=np.float64)
    # Explanation: This line creates the output folder.
    MakeFolder(Path(outputpath).expanduser().parent)
    # Explanation: This line opens the PLY output file.
    with Path(outputpath).expanduser().open("w", encoding="utf-8") as fileobject:
        # Explanation: This line writes the PLY header.
        fileobject.write("ply\nformat ascii 1.0\n")
        # Explanation: This line writes the vertex count.
        fileobject.write(f"element vertex {centerarray.shape[0]}\n")
        # Explanation: This line writes the x property.
        fileobject.write("property float x\n")
        # Explanation: This line writes the y property.
        fileobject.write("property float y\n")
        # Explanation: This line writes the z property.
        fileobject.write("property float z\n")
        # Explanation: This line writes the edge count.
        fileobject.write(f"element edge {max(0, centerarray.shape[0] - 1)}\n")
        # Explanation: This line writes the first edge property.
        fileobject.write("property int vertex1\n")
        # Explanation: This line writes the second edge property.
        fileobject.write("property int vertex2\n")
        # Explanation: This line ends the PLY header.
        fileobject.write("end_header\n")
        # Explanation: This line loops through camera centers.
        for centerobject in centerarray:
            # Explanation: This line writes one camera center vertex.
            fileobject.write(f"{centerobject[0]} {centerobject[1]} {centerobject[2]}\n")
        # Explanation: This line loops through consecutive center indexes.
        for centerindex in range(centerarray.shape[0] - 1):
            # Explanation: This line writes one trajectory edge.
            fileobject.write(f"{centerindex} {centerindex + 1}\n")


# Explanation: This line starts the AppendPose function.
def AppendPose(poselist, currentpose):
    """
    Append one camera-to-world pose to the pose list.

    Step wise:
    1. Copy the current pose matrix.
    2. Append the copy to the pose list.
    """
    # Explanation: This line appends a copied pose matrix.
    poselist.append(currentpose.copy())


# Explanation: This line starts the RunSequence function.
def RunSequence(classicmodule, sequenceobject, configdata):
    """
    Run Classic 4-frame VO on one offline RGB-D sequence.

    Step wise:
    1. Load RGB and depth file lists.
    2. Read camera and threshold settings.
    3. Slide a four-frame window through the sequence.
    4. Estimate t to t+1 pose using Classic feature/depth/PnP logic.
    5. Append a pose for every frame, reusing the previous pose on failure.
    6. Save poses, trajectory PLY, and run log.
    """
    # Explanation: This line reads the sequence id.
    sequenceid = str(sequenceobject["id"])
    # Explanation: This line reads the RGB image folder.
    rgbfolder = sequenceobject["rgb-dir"]
    # Explanation: This line reads the depth image folder.
    depthfolder = sequenceobject["depth-dir"]
    # Explanation: This line creates the sequence output folder.
    outputdir = MakeFolder(sequenceobject["output-dir"])
    # Explanation: This line lists RGB image files.
    rgblist = ListImages(rgbfolder)
    # Explanation: This line lists depth image files.
    depthlist = ListImages(depthfolder)
    # Explanation: This line computes the shared available frame count.
    framecount = min(len(rgblist), len(depthlist))
    # Explanation: This line checks whether enough frames exist for four-frame VO.
    if framecount < 4:
        # Explanation: This line raises a clear error for too few frames.
        raise ValueError(f"{sequenceid} needs at least four RGB-D frames")
    # Explanation: This line reads camera settings from config.
    cameraobject = configdata.get("camera", {})
    # Explanation: This line reads horizontal focal length.
    fxvalue = float(cameraobject.get("fx", 580.0))
    # Explanation: This line reads vertical focal length.
    fyvalue = float(cameraobject.get("fy", 580.0))
    # Explanation: This line reads horizontal principal point.
    cxvalue = float(cameraobject.get("cx", 320.0))
    # Explanation: This line reads vertical principal point.
    cyvalue = float(cameraobject.get("cy", 240.0))
    # Explanation: This line creates the camera tuple expected by Classic functions.
    ktuple = (fxvalue, fyvalue, cxvalue, cyvalue)
    # Explanation: This line creates the 3 by 3 camera intrinsic matrix.
    matrixk = np.array([[fxvalue, 0.0, cxvalue], [0.0, fyvalue, cyvalue], [0.0, 0.0, 1.0]], dtype=np.float32)
    # Explanation: This line reads depth scale from config.
    depthscale = float(configdata.get("depth", {}).get("scale", 1000.0))
    # Explanation: This line reads Classic VO settings from config.
    settings = configdata.get("classic", {})
    # Explanation: This line reads blur rejection threshold.
    lpthresh = float(settings.get("laplacian-threshold", 80.0))
    # Explanation: This line reads maximum Shi-Tomasi corners.
    gfcornermax = int(settings.get("goodfeature-max-corners", 3500))
    # Explanation: This line reads Shi-Tomasi quality level.
    gflevel = float(settings.get("goodfeature-quality", 0.015))
    # Explanation: This line reads optical-flow magnitude filtering threshold.
    magthresh = float(settings.get("lk-magnitude-threshold", 150.0))
    # Explanation: This line reads optical-flow error threshold.
    errthresh = float(settings.get("lk-error-threshold", 20.0))
    # Explanation: This line reads stochastic optimiser iteration count.
    optimiseriters = int(settings.get("optimisation-iters", 30))
    # Explanation: This line reads the maximum number of frames to process.
    maxframes = int(settings.get("max-frames", 0))
    # Explanation: This line limits frame count when max-frames is configured.
    framecount = min(framecount, maxframes) if maxframes > 0 else framecount
    # Explanation: This line reads the first frame so image size can be stored.
    firstgray, firstcolor, firstdepth = ReadFrame(rgblist[0], depthlist[0], depthscale)
    # Explanation: This line reads image height and width.
    imageheight, imagewidth = firstgray.shape[:2]
    # Explanation: This line creates the initial camera-to-world pose.
    currentpose = np.eye(4, dtype=np.float64)
    # Explanation: This line creates the output pose list.
    poselist = []
    # Explanation: This line stores the first pose as identity.
    AppendPose(poselist, currentpose)
    # Explanation: This line creates a run log row list.
    logrows = []
    # Explanation: This line starts a loop over four-frame anchor indexes.
    for frameindex in range(framecount - 3):
        # Explanation: This line reads frame t.
        framet = ReadFrame(rgblist[frameindex], depthlist[frameindex], depthscale)
        # Explanation: This line reads frame t plus one.
        frametone = ReadFrame(rgblist[frameindex + 1], depthlist[frameindex + 1], depthscale)
        # Explanation: This line reads frame t plus two.
        framettwo = ReadFrame(rgblist[frameindex + 2], depthlist[frameindex + 2], depthscale)
        # Explanation: This line reads frame t plus three.
        frametthree = ReadFrame(rgblist[frameindex + 3], depthlist[frameindex + 3], depthscale)
        # Explanation: This line unpacks frame t.
        grayt, colort, deptht = framet
        # Explanation: This line unpacks frame t plus one.
        grayone, colorone, depthone = frametone
        # Explanation: This line unpacks frame t plus two.
        graytwo, colortwo, depthtwo = framettwo
        # Explanation: This line unpacks frame t plus three.
        graythree, colorthree, depththree = frametthree
        # Explanation: This line applies Classic CLAHE preprocessing to frame t.
        grayt = classicmodule.clahe(grayt)
        # Explanation: This line applies Classic CLAHE preprocessing to frame t plus one.
        grayone = classicmodule.clahe(grayone)
        # Explanation: This line applies Classic CLAHE preprocessing to frame t plus two.
        graytwo = classicmodule.clahe(graytwo)
        # Explanation: This line applies Classic CLAHE preprocessing to frame t plus three.
        graythree = classicmodule.clahe(graythree)
        # Explanation: This line computes blur status for frame t.
        blurt, scoret = classicmodule.laplacian(grayt, lpthresh)
        # Explanation: This line computes blur status for frame t plus one.
        blurone, scoreone = classicmodule.laplacian(grayone, lpthresh)
        # Explanation: This line computes blur status for frame t plus two.
        blurtwo, scoretwo = classicmodule.laplacian(graytwo, lpthresh)
        # Explanation: This line computes blur status for frame t plus three.
        blurthree, scorethree = classicmodule.laplacian(graythree, lpthresh)
        # Explanation: This line creates the default log row for the current anchor frame.
        logobject = {"frame": frameindex, "status": "started", "tracks": 0, "median1": "", "median2": "", "median3": ""}
        # Explanation: This line checks whether any frame in the four-frame window is blurry.
        if blurt or blurone or blurtwo or blurthree:
            # Explanation: This line records the skipped blur status.
            logobject["status"] = "blur"
            # Explanation: This line appends the log row.
            logrows.append(logobject)
            # Explanation: This line appends the unchanged pose for frame t plus one.
            AppendPose(poselist, currentpose)
            # Explanation: This line moves to the next window.
            continue
        # Explanation: This line detects Shi-Tomasi features on the anchor frame.
        featurepoints, featurescores = classicmodule.goodfeature(grayt, gfcornermax, gflevel)
        # Explanation: This line checks whether feature detection failed.
        if featurepoints is None or len(featurepoints) == 0:
            # Explanation: This line records the skipped feature status.
            logobject["status"] = "no features"
            # Explanation: This line appends the log row.
            logrows.append(logobject)
            # Explanation: This line appends the unchanged pose for frame t plus one.
            AppendPose(poselist, currentpose)
            # Explanation: This line moves to the next window.
            continue
        # Explanation: This line tracks features through four frames with Classic optical-flow filtering.
        pointt, pointone, pointtwo, pointthree = classicmodule.lkoptical4fil(grayt, grayone, graytwo, graythree, featurepoints, magthresh, errthresh)
        # Explanation: This line records the number of surviving tracks.
        logobject["tracks"] = int(len(pointthree))
        # Explanation: This line checks whether four-frame tracking failed.
        if len(pointthree) == 0:
            # Explanation: This line records the skipped tracking status.
            logobject["status"] = "no tracks"
            # Explanation: This line appends the log row.
            logrows.append(logobject)
            # Explanation: This line appends the unchanged pose for frame t plus one.
            AppendPose(poselist, currentpose)
            # Explanation: This line moves to the next window.
            continue
        # Explanation: This line backprojects anchor-frame feature points into 3D using depth.
        pointsthree = classicmodule.backproject(pointt, deptht, *ktuple, k=1)
        # Explanation: This line builds a valid 3D point mask.
        validmask = np.isfinite(pointsthree).all(axis=1)
        # Explanation: This line checks whether no valid 3D points remain.
        if not np.any(validmask):
            # Explanation: This line records the skipped depth status.
            logobject["status"] = "no depth"
            # Explanation: This line appends the log row.
            logrows.append(logobject)
            # Explanation: This line appends the unchanged pose for frame t plus one.
            AppendPose(poselist, currentpose)
            # Explanation: This line moves to the next window.
            continue
        # Explanation: This line keeps valid 3D points.
        pointsthree = pointsthree[validmask]
        # Explanation: This line keeps valid tracked points in frame t plus one.
        pointone = pointone[validmask]
        # Explanation: This line keeps valid tracked points in frame t plus two.
        pointtwo = pointtwo[validmask]
        # Explanation: This line keeps valid tracked points in frame t plus three.
        pointthree = pointthree[validmask]
        # Explanation: This line estimates pose from frame t to frame t plus one.
        rotationone, translationone, inlierone = classicmodule.pnpransac(pointsthree, pointone, *ktuple)
        # Explanation: This line estimates pose from frame t to frame t plus two.
        rotationtwo, translationtwo, inliertwo = classicmodule.pnpransac(pointsthree, pointtwo, *ktuple)
        # Explanation: This line estimates pose from frame t to frame t plus three.
        rotationthree, translationthree, inlierthree = classicmodule.pnpransac(pointsthree, pointthree, *ktuple)
        # Explanation: This line checks whether PnP failed.
        if rotationone is None or rotationtwo is None or rotationthree is None:
            # Explanation: This line records the skipped PnP status.
            logobject["status"] = "pnp failed"
            # Explanation: This line appends the log row.
            logrows.append(logobject)
            # Explanation: This line appends the unchanged pose for frame t plus one.
            AppendPose(poselist, currentpose)
            # Explanation: This line moves to the next window.
            continue
        # Explanation: This line refines the t to t plus one pose using the existing Classic optimiser.
        rotationopt, translationopt = classicmodule.poseoptim(rotationone, translationone, pointsthree, pointone, ktuple, imageheight, imagewidth, optimiseriters)
        # Explanation: This line builds the optimised rigid transform from t to t plus one.
        relativert = classicmodule.RTmatrix(rotationopt, translationopt)
        # Explanation: This line builds auxiliary transforms for reprojection diagnostics.
        rttwo = classicmodule.RTmatrix(rotationtwo, translationtwo)
        # Explanation: This line builds auxiliary transforms for reprojection diagnostics.
        rtthree = classicmodule.RTmatrix(rotationthree, translationthree)
        # Explanation: This line computes reprojection errors for the four-frame window.
        errors, errormasks = classicmodule.Reprojecterror(pointsthree, [pointone, pointtwo, pointthree], [(rotationopt, translationopt), (rotationtwo, translationtwo), (rotationthree, translationthree)], ktuple, imageheight, imagewidth)
        # Explanation: This line computes median reprojection error scores.
        scorevalue, medians = classicmodule.errorscore(errors)
        # Explanation: This line records successful status.
        logobject["status"] = "ok"
        # Explanation: This line records the first median reprojection error.
        logobject["median1"] = float(medians[0])
        # Explanation: This line records the second median reprojection error.
        logobject["median2"] = float(medians[1])
        # Explanation: This line records the third median reprojection error.
        logobject["median3"] = float(medians[2])
        # Explanation: This line appends the log row.
        logrows.append(logobject)
        # Explanation: This line updates camera-to-world pose using the inverse camera-motion convention from the original script.
        currentpose = currentpose @ np.linalg.inv(relativert.astype(np.float64))
        # Explanation: This line appends the new pose for frame t plus one.
        AppendPose(poselist, currentpose)
    # Explanation: This line pads missing final frames with the latest pose.
    while len(poselist) < framecount:
        # Explanation: This line appends the latest pose for a frame without a four-frame estimate.
        AppendPose(poselist, currentpose)
    # Explanation: This line converts the pose list into an array.
    posearray = np.asarray(poselist[:framecount], dtype=np.float64)
    # Explanation: This line converts RGB paths into image-name strings.
    namelist = np.asarray([pathobject.name for pathobject in rgblist[:framecount]])
    # Explanation: This line saves the Classic pose dump with poses, intrinsics, size, and image names.
    np.savez_compressed(outputdir / "posesdump.npz", Twc=posearray, K=matrixk, W=np.asarray(imagewidth), H=np.asarray(imageheight), image_names=namelist)
    # Explanation: This line saves camera centers as a trajectory PLY.
    SavePly(posearray[:, :3, 3], outputdir / "trajectory.ply")
    # Explanation: This line saves the per-frame run log.
    SaveCsv(logrows, outputdir / "runlog.csv")
    # Explanation: This line prints a sequence completion message.
    print(f"Classic VO complete for {sequenceid}: {outputdir}")


# Explanation: This line starts the ParseArgs function.
def ParseArgs():
    """
    Parse command line arguments.

    Step wise:
    1. Create an argument parser.
    2. Add the config path argument.
    3. Return parsed arguments.
    """
    # Explanation: This line creates the command-line parser.
    parser = argparse.ArgumentParser(description="Run Classic 4-frame VO on offline RGB-D sequences.")
    # Explanation: This line adds the YAML config argument.
    parser.add_argument("config", help="Path to classicvo.yml")
    # Explanation: This line parses and returns command-line arguments.
    return parser.parse_args()


# Explanation: This line starts the Main function.
def Main():
    """
    Run the offline Classic VO batch.

    Step wise:
    1. Parse command-line arguments.
    2. Load the config file.
    3. Load the original Classic VO script.
    4. Run every configured sequence.
    """
    # Explanation: This line parses command-line arguments.
    args = ParseArgs()
    # Explanation: This line loads the YAML config.
    configdata = LoadConfig(args.config)
    # Explanation: This line loads the original Classic VO module.
    classicmodule = LoadClassicModule(configdata["classic-script"])
    # Explanation: This line loops through configured sequences.
    for sequenceobject in configdata.get("sequences", []):
        # Explanation: This line runs Classic VO for one sequence.
        RunSequence(classicmodule, sequenceobject, configdata)


# Explanation: This line runs the script entry point.
if __name__ == "__main__":
    # Explanation: This line calls the main workflow.
    Main()
