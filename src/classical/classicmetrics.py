"""
Metrics script for the traditional Classic visual odometry baseline.

Overall explanation:

This script evaluates Classic VO outputs produced by `classicrunner.py`. It is
intended to give the traditional feature/depth/PnP method a clear result
structure for local motion, trajectory shape, and run stability.

What this script does:

1. It reads a YAML config file.
2. It loads one Classic `posesdump.npz` file per configured sequence.
3. It computes local translation and local rotation from frame-to-frame poses.
4. It plots the raw trajectory.
5. It keeps reference-based metrics disabled unless the config explicitly enables them.
6. It saves no-ground-truth trajectory stability visuals.
7. It optionally computes ATE and RPE only as baseline-relative metrics.
8. It saves CSV files, plots, JSON summaries, and one combined summary table.

Coding rules used in this file:

- Every custom function starts with a capital letter.
- Every custom function has an explanation docstring.
- Every executable line has a nearby explanation comment.
- Variable names avoid underscores unless Python or a library requires them.
"""

# Explanation: This line imports argparse so the script can read the config path from the command line.
import argparse
# Explanation: This line imports csv so tables can be saved.
import csv
# Explanation: This line imports json so summaries can be saved.
import json
# Explanation: This line imports math so numeric checks can use standard helpers.
import math
# Explanation: This line imports Path for readable filesystem path handling.
from pathlib import Path

# Explanation: This line imports Matplotlib for saving figures.
import matplotlib
# Explanation: This line selects a non-interactive backend for image saving.
matplotlib.use("Agg")
# Explanation: This line imports Matplotlib plotting helpers.
import matplotlib.pyplot as plt
# Explanation: This line imports NumPy for pose and metric calculations.
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
    3. Parse YAML into Python data.
    4. Return the parsed config.
    """
    # Explanation: This line converts the config path into a Path object.
    pathobject = Path(configpath).expanduser()
    # Explanation: This line opens the config file as text.
    with pathobject.open("r", encoding="utf-8") as fileobject:
        # Explanation: This line parses the YAML text.
        configdata = yaml.safe_load(fileobject)
    # Explanation: This line returns the parsed config.
    return configdata


# Explanation: This line starts the MakeFolder function.
def MakeFolder(folderpath):
    """
    Create a folder if needed.

    Step wise:
    1. Convert the folder path into a Path object.
    2. Create missing parent folders.
    3. Return the folder path.
    """
    # Explanation: This line converts the folder path into a Path object.
    pathobject = Path(folderpath).expanduser()
    # Explanation: This line creates the folder and parent folders.
    pathobject.mkdir(parents=True, exist_ok=True)
    # Explanation: This line returns the created folder path.
    return pathobject


# Explanation: This line starts the RemoveFile function.
def RemoveFile(filepath):
    """
    Remove one stale output file when it exists.

    Step wise:
    1. Convert the input path into a Path object.
    2. Check whether the file exists.
    3. Delete only that file.
    """
    # Explanation: This line converts the file path into a Path object.
    pathobject = Path(filepath).expanduser()
    # Explanation: This line checks whether the file exists.
    if pathobject.exists():
        # Explanation: This line deletes the stale output file.
        pathobject.unlink()


# Explanation: This line starts the SaveCsv function.
def SaveCsv(rowlist, outputpath):
    """
    Save dictionary rows into a CSV file.

    Step wise:
    1. Return early when there are no rows.
    2. Create the output folder.
    3. Use row keys as CSV columns.
    4. Write the table.
    """
    # Explanation: This line checks whether there are rows to save.
    if len(rowlist) == 0:
        # Explanation: This line returns early for an empty table.
        return
    # Explanation: This line creates the output folder.
    MakeFolder(Path(outputpath).expanduser().parent)
    # Explanation: This line reads column names from the first row.
    fieldnames = list(rowlist[0].keys())
    # Explanation: This line opens the CSV file for writing.
    with Path(outputpath).expanduser().open("w", newline="", encoding="utf-8") as fileobject:
        # Explanation: This line creates a CSV writer.
        writerobject = csv.DictWriter(fileobject, fieldnames=fieldnames)
        # Explanation: This line writes the CSV header.
        writerobject.writeheader()
        # Explanation: This line writes all CSV rows.
        writerobject.writerows(rowlist)


# Explanation: This line starts the SaveJson function.
def SaveJson(dataobject, outputpath):
    """
    Save structured data as JSON.

    Step wise:
    1. Create the output folder.
    2. Open the JSON file.
    3. Write the data with indentation.
    """
    # Explanation: This line creates the output folder.
    MakeFolder(Path(outputpath).expanduser().parent)
    # Explanation: This line opens the JSON file.
    with Path(outputpath).expanduser().open("w", encoding="utf-8") as fileobject:
        # Explanation: This line writes formatted JSON data.
        json.dump(dataobject, fileobject, indent=2)


# Explanation: This line starts the LoadPoses function.
def LoadPoses(posepath):
    """
    Load camera-to-world poses from a pose dump.

    Step wise:
    1. Open the npz pose dump.
    2. Prefer the `Twc` key.
    3. Fall back to any N by 4 by 4 pose array.
    4. Return normalized 4 by 4 poses.
    """
    # Explanation: This line converts the pose path into a Path object.
    pathobject = Path(posepath).expanduser()
    # Explanation: This line loads the NumPy pose dump.
    posedata = np.load(pathobject, allow_pickle=True)
    # Explanation: This line checks whether the expected Twc key exists.
    if "Twc" in posedata.files:
        # Explanation: This line reads poses from the Twc key.
        posearray = np.asarray(posedata["Twc"], dtype=np.float64)
    # Explanation: This line handles pose dumps without the Twc key.
    else:
        # Explanation: This line creates an empty pose placeholder.
        posearray = None
        # Explanation: This line loops through all npz keys.
        for keyname in posedata.files:
            # Explanation: This line reads the candidate array.
            candidate = np.asarray(posedata[keyname])
            # Explanation: This line checks whether the candidate looks like pose matrices.
            if candidate.ndim >= 3 and candidate.shape[-2:] == (4, 4):
                # Explanation: This line accepts the candidate pose array.
                posearray = candidate.astype(np.float64)
                # Explanation: This line stops the fallback search.
                break
        # Explanation: This line checks whether no pose array was found.
        if posearray is None:
            # Explanation: This line raises a clear error for invalid pose dumps.
            raise KeyError(f"No pose array found in {pathobject}")
    # Explanation: This line reshapes flattened 4 by 4 poses when needed.
    if posearray.ndim == 2 and posearray.shape[1] == 16:
        # Explanation: This line converts flat poses into matrices.
        posearray = posearray.reshape((-1, 4, 4))
    # Explanation: This line checks whether the pose shape is valid.
    if posearray.ndim != 3 or posearray.shape[-2:] != (4, 4):
        # Explanation: This line raises a clear error for unsupported pose shapes.
        raise ValueError(f"Unsupported pose shape: {posearray.shape}")
    # Explanation: This line returns the pose array.
    return posearray


# Explanation: This line starts the LoadReference function.
def LoadReference(referencepath):
    """
    Load an optional transforms.json reference trajectory.

    Step wise:
    1. Return None when the reference path is blank.
    2. Read the JSON file.
    3. Extract each frame transform matrix.
    4. Return camera-to-world poses.
    """
    # Explanation: This line checks whether YAML loaded the reference path as None.
    if referencepath is None:
        # Explanation: This line returns no reference when the config value is empty.
        return None
    # Explanation: This line checks whether the reference path is blank.
    if str(referencepath).strip() == "":
        # Explanation: This line returns no reference for blank paths.
        return None
    # Explanation: This line converts the reference path into a Path object.
    pathobject = Path(referencepath).expanduser()
    # Explanation: This line checks whether the reference file exists.
    if not pathobject.exists():
        # Explanation: This line returns no reference for missing files.
        return None
    # Explanation: This line opens the transforms JSON file.
    with pathobject.open("r", encoding="utf-8") as fileobject:
        # Explanation: This line parses the JSON file.
        dataobject = json.load(fileobject)
    # Explanation: This line creates an empty pose list.
    poselist = []
    # Explanation: This line loops through reference frames.
    for frameobject in dataobject.get("frames", []):
        # Explanation: This line checks whether the frame has a transform matrix.
        if "transform_matrix" in frameobject:
            # Explanation: This line appends the transform matrix as a pose.
            poselist.append(np.asarray(frameobject["transform_matrix"], dtype=np.float64))
    # Explanation: This line checks whether no poses were loaded.
    if len(poselist) == 0:
        # Explanation: This line returns no reference when the file has no poses.
        return None
    # Explanation: This line returns the reference pose array.
    return np.asarray(poselist, dtype=np.float64)


# Explanation: This line starts the GetCenters function.
def GetCenters(posearray):
    """
    Extract camera centers from camera-to-world poses.

    Step wise:
    1. Convert poses into a NumPy array.
    2. Read the translation column.
    3. Return N by 3 camera centers.
    """
    # Explanation: This line converts poses into a NumPy array.
    posearray = np.asarray(posearray, dtype=np.float64)
    # Explanation: This line returns the translation column as camera centers.
    return posearray[:, :3, 3]


# Explanation: This line starts the RotationAngle function.
def RotationAngle(rotationmatrix):
    """
    Convert a rotation matrix into an angle in degrees.

    Step wise:
    1. Compute the matrix trace.
    2. Convert trace into cosine angle.
    3. Clamp the cosine for numerical stability.
    4. Return the angle in degrees.
    """
    # Explanation: This line computes the trace of the rotation matrix.
    tracevalue = float(np.trace(rotationmatrix))
    # Explanation: This line computes the cosine of the rotation angle.
    cosvalue = (tracevalue - 1.0) / 2.0
    # Explanation: This line clamps the cosine value into a valid range.
    cosvalue = max(-1.0, min(1.0, cosvalue))
    # Explanation: This line returns the angle in degrees.
    return float(np.degrees(np.arccos(cosvalue)))


# Explanation: This line starts the PathLength function.
def PathLength(centerarray):
    """
    Compute total trajectory path length.

    Step wise:
    1. Convert centers into a NumPy array.
    2. Compute distances between consecutive centers.
    3. Sum those distances.
    """
    # Explanation: This line converts centers into a NumPy array.
    centerarray = np.asarray(centerarray, dtype=np.float64)
    # Explanation: This line checks whether fewer than two centers exist.
    if centerarray.shape[0] < 2:
        # Explanation: This line returns zero path length for too few centers.
        return 0.0
    # Explanation: This line computes frame-to-frame displacement vectors.
    differences = np.diff(centerarray, axis=0)
    # Explanation: This line returns the summed displacement magnitudes.
    return float(np.sum(np.linalg.norm(differences, axis=1)))


# Explanation: This line starts the Summarize function.
def Summarize(values):
    """
    Summarize a numeric sequence.

    Step wise:
    1. Convert values into a finite NumPy array.
    2. Return count, mean, median, RMSE, min, and max.
    """
    # Explanation: This line converts values into a numeric array.
    valuearray = np.asarray(values, dtype=np.float64)
    # Explanation: This line keeps only finite values.
    valuearray = valuearray[np.isfinite(valuearray)]
    # Explanation: This line checks whether the array is empty.
    if valuearray.size == 0:
        # Explanation: This line returns empty statistics for no data.
        return {"count": 0}
    # Explanation: This line returns summary statistics.
    return {"count": int(valuearray.size), "mean": float(np.mean(valuearray)), "median": float(np.median(valuearray)), "rmse": float(np.sqrt(np.mean(valuearray ** 2))), "min": float(np.min(valuearray)), "max": float(np.max(valuearray))}


# Explanation: This line starts the SaveLine function.
def SaveLine(xvalues, yvalues, titletext, xlabel, ylabel, outputpath):
    """
    Save one line plot.

    Step wise:
    1. Create the output folder.
    2. Plot x and y values.
    3. Label the plot.
    4. Save the figure.
    """
    # Explanation: This line creates the output folder.
    MakeFolder(Path(outputpath).expanduser().parent)
    # Explanation: This line creates a new figure.
    plt.figure(figsize=(10, 4))
    # Explanation: This line draws the line plot.
    plt.plot(xvalues, yvalues, marker="o", markersize=2, linewidth=1)
    # Explanation: This line sets the plot title.
    plt.title(titletext)
    # Explanation: This line labels the x axis.
    plt.xlabel(xlabel)
    # Explanation: This line labels the y axis.
    plt.ylabel(ylabel)
    # Explanation: This line adds a grid.
    plt.grid(True, alpha=0.3)
    # Explanation: This line adjusts plot spacing.
    plt.tight_layout()
    # Explanation: This line saves the figure.
    plt.savefig(str(outputpath), dpi=180)
    # Explanation: This line closes the figure.
    plt.close()


# Explanation: This line starts the SaveTrajectory function.
def SaveTrajectory(centerarray, titletext, outputpath):
    """
    Save a top-down trajectory plot.

    Step wise:
    1. Plot X against Z.
    2. Keep equal axis scaling.
    3. Save the figure.
    """
    # Explanation: This line converts centers into a numeric array.
    centerarray = np.asarray(centerarray, dtype=np.float64)
    # Explanation: This line creates the output folder.
    MakeFolder(Path(outputpath).expanduser().parent)
    # Explanation: This line creates a new figure.
    plt.figure(figsize=(7, 7))
    # Explanation: This line plots the trajectory.
    plt.plot(centerarray[:, 0], centerarray[:, 2], marker="o", markersize=2, linewidth=1.2, label=titletext)
    # Explanation: This line labels the x axis.
    plt.xlabel("X position")
    # Explanation: This line labels the z axis.
    plt.ylabel("Z position")
    # Explanation: This line sets the title.
    plt.title(titletext)
    # Explanation: This line keeps equal map scale.
    plt.axis("equal")
    # Explanation: This line adds a grid.
    plt.grid(True, alpha=0.3)
    # Explanation: This line adds a legend.
    plt.legend()
    # Explanation: This line adjusts plot spacing.
    plt.tight_layout()
    # Explanation: This line saves the plot.
    plt.savefig(str(outputpath), dpi=180)
    # Explanation: This line closes the figure.
    plt.close()


# Explanation: This line starts the AlignCenters function.
def AlignCenters(sourcecenters, referencecenters):
    """
    Align estimated centers to reference centers with Sim3.

    Step wise:
    1. Match shared frame count.
    2. Remove both means.
    3. Estimate rotation with SVD.
    4. Estimate scale.
    5. Estimate translation.
    6. Return aligned centers and ATE errors.
    """
    # Explanation: This line computes matched pair count.
    paircount = min(sourcecenters.shape[0], referencecenters.shape[0])
    # Explanation: This line keeps matched source centers.
    sourcearray = np.asarray(sourcecenters[:paircount], dtype=np.float64)
    # Explanation: This line keeps matched reference centers.
    referencearray = np.asarray(referencecenters[:paircount], dtype=np.float64)
    # Explanation: This line computes source mean.
    sourcemean = np.mean(sourcearray, axis=0)
    # Explanation: This line computes reference mean.
    referencemean = np.mean(referencearray, axis=0)
    # Explanation: This line centers source positions.
    sourcecentered = sourcearray - sourcemean
    # Explanation: This line centers reference positions.
    referencecentered = referencearray - referencemean
    # Explanation: This line computes covariance matrix.
    covmatrix = sourcecentered.T @ referencecentered / float(paircount)
    # Explanation: This line runs singular value decomposition.
    umat, singularvalues, vmat = np.linalg.svd(covmatrix)
    # Explanation: This line creates the reflection correction matrix.
    correction = np.eye(3)
    # Explanation: This line checks whether rotation would reflect the trajectory.
    if np.linalg.det(vmat.T @ umat.T) < 0:
        # Explanation: This line corrects the last axis sign.
        correction[-1, -1] = -1.0
    # Explanation: This line computes alignment rotation.
    rotationmatrix = vmat.T @ correction @ umat.T
    # Explanation: This line computes source variance.
    sourcevariance = np.mean(np.sum(sourcecentered ** 2, axis=1))
    # Explanation: This line computes Sim3 scale.
    scalevalue = float(np.sum(singularvalues * np.diag(correction)) / max(sourcevariance, 1e-12))
    # Explanation: This line computes alignment translation.
    translationvector = referencemean - scalevalue * (rotationmatrix @ sourcemean)
    # Explanation: This line applies Sim3 alignment to centers.
    alignedcenters = (scalevalue * (rotationmatrix @ sourcearray.T)).T + translationvector
    # Explanation: This line computes ATE errors.
    errorarray = np.linalg.norm(alignedcenters - referencearray, axis=1)
    # Explanation: This line returns alignment outputs.
    return alignedcenters, errorarray, scalevalue, rotationmatrix, translationvector


# Explanation: This line starts the AlignPoses function.
def AlignPoses(posearray, scalevalue, rotationmatrix, translationvector):
    """
    Apply Sim3 alignment to camera-to-world poses.

    Step wise:
    1. Copy the pose array.
    2. Rotate camera orientations.
    3. Scale, rotate, and translate camera centers.
    4. Return aligned poses.
    """
    # Explanation: This line copies the input poses.
    alignedposes = posearray.copy()
    # Explanation: This line loops through every pose.
    for poseindex in range(alignedposes.shape[0]):
        # Explanation: This line aligns camera orientation.
        alignedposes[poseindex, :3, :3] = rotationmatrix @ posearray[poseindex, :3, :3]
        # Explanation: This line aligns camera center.
        alignedposes[poseindex, :3, 3] = scalevalue * (rotationmatrix @ posearray[poseindex, :3, 3]) + translationvector
    # Explanation: This line returns aligned poses.
    return alignedposes


# Explanation: This line starts the ComputeLocalMotion function.
def ComputeLocalMotion(posearray):
    """
    Compute local frame-to-frame translation and rotation.

    Step wise:
    1. Compare every pose to the next pose.
    2. Compute relative translation magnitude.
    3. Compute relative rotation angle.
    4. Return per-frame rows.
    """
    # Explanation: This line creates an empty row list.
    rowlist = []
    # Explanation: This line loops through consecutive pose pairs.
    for frameindex in range(posearray.shape[0] - 1):
        # Explanation: This line computes relative camera motion.
        relativepose = np.linalg.inv(posearray[frameindex]) @ posearray[frameindex + 1]
        # Explanation: This line appends local motion metrics.
        rowlist.append({"frame": frameindex, "translationm": float(np.linalg.norm(relativepose[:3, 3])), "rotationdeg": RotationAngle(relativepose[:3, :3])})
    # Explanation: This line returns local motion rows.
    return rowlist


# Explanation: This line starts the ComputeRpe function.
def ComputeRpe(posearray, referencearray, intervallist):
    """
    Compute Relative Pose Error over configured frame intervals.

    Step wise:
    1. Match estimated and reference pose counts.
    2. Compare relative motion over each interval.
    3. Store translation and rotation error summaries.
    """
    # Explanation: This line computes matched pose count.
    paircount = min(posearray.shape[0], referencearray.shape[0])
    # Explanation: This line creates an empty RPE row list.
    rowlist = []
    # Explanation: This line loops through each configured interval.
    for intervalvalue in intervallist:
        # Explanation: This line creates an empty translation error list.
        translationerrors = []
        # Explanation: This line creates an empty rotation error list.
        rotationerrors = []
        # Explanation: This line loops through pose pairs separated by the interval.
        for frameindex in range(paircount - int(intervalvalue)):
            # Explanation: This line computes estimated relative motion.
            estmove = np.linalg.inv(posearray[frameindex]) @ posearray[frameindex + int(intervalvalue)]
            # Explanation: This line computes reference relative motion.
            refmove = np.linalg.inv(referencearray[frameindex]) @ referencearray[frameindex + int(intervalvalue)]
            # Explanation: This line computes relative pose error.
            errormove = np.linalg.inv(refmove) @ estmove
            # Explanation: This line appends translation error magnitude.
            translationerrors.append(float(np.linalg.norm(errormove[:3, 3])))
            # Explanation: This line appends rotation error angle.
            rotationerrors.append(RotationAngle(errormove[:3, :3]))
        # Explanation: This line summarizes translation errors.
        transsummary = Summarize(translationerrors)
        # Explanation: This line summarizes rotation errors.
        rotsummary = Summarize(rotationerrors)
        # Explanation: This line appends one RPE summary row.
        rowlist.append({"interval": int(intervalvalue), "paircount": transsummary.get("count", 0), "translationrmsem": transsummary.get("rmse", ""), "rotationrmsedeg": rotsummary.get("rmse", ""), "translationmeanm": transsummary.get("mean", ""), "rotationmeandeg": rotsummary.get("mean", "")})
    # Explanation: This line returns RPE rows.
    return rowlist


# Explanation: This line starts the SaveMotionStabilityVisual function.
def SaveMotionStabilityVisual(centerarray, localrows, titletext, outputpath):
    """
    Save a no-ground-truth Classic pose stability visual.

    Step wise:
    1. Draw the estimated Classic trajectory without a reference line.
    2. Colour trajectory points by cumulative estimated path length.
    3. Plot frame-to-frame translation below the trajectory.
    4. Save one figure for qualitative stability discussion.

    Important:
    This is not ground-truth drift. It shows estimated path shape and local
    motion stability because external ground-truth camera poses are unavailable.
    """
    # Explanation: This line converts centers into a numeric array.
    centerarray = np.asarray(centerarray, dtype=np.float64)
    # Explanation: This line checks whether there are enough centers to plot.
    if centerarray.shape[0] < 2:
        # Explanation: This line returns early when a trajectory cannot be formed.
        return
    # Explanation: This line creates the output folder.
    MakeFolder(Path(outputpath).expanduser().parent)
    # Explanation: This line computes frame-to-frame translation from camera centers.
    stepvalues = np.linalg.norm(centerarray[1:] - centerarray[:-1], axis=1)
    # Explanation: This line computes cumulative estimated path distance for each pose.
    distancevalues = np.concatenate(([0.0], np.cumsum(stepvalues)))
    # Explanation: This line reads translation values from the local motion table.
    localvalues = [row["translationm"] for row in localrows]
    # Explanation: This line creates a two-row figure.
    figobject, axislist = plt.subplots(2, 1, figsize=(8, 9))
    # Explanation: This line plots the estimated Classic trajectory as a light line.
    axislist[0].plot(centerarray[:, 0], -centerarray[:, 2], color="blue", linewidth=1.0, alpha=0.45, label="Classic VO")
    # Explanation: This line draws trajectory points coloured by cumulative motion.
    scatterobject = axislist[0].scatter(centerarray[:, 0], -centerarray[:, 2], c=distancevalues, cmap="viridis", s=18)
    # Explanation: This line marks the trajectory start.
    axislist[0].scatter(centerarray[0, 0], -centerarray[0, 2], color="white", edgecolor="black", s=45, label="Start")
    # Explanation: This line marks the trajectory end.
    axislist[0].scatter(centerarray[-1, 0], -centerarray[-1, 2], color="black", edgecolor="white", s=45, label="End")
    # Explanation: This line labels the top plot x axis.
    axislist[0].set_xlabel("Estimated X")
    # Explanation: This line labels the top plot y axis.
    axislist[0].set_ylabel("Estimated negative Z")
    # Explanation: This line sets equal trajectory scaling.
    axislist[0].axis("equal")
    # Explanation: This line adds a grid to the trajectory plot.
    axislist[0].grid(True, alpha=0.3)
    # Explanation: This line adds a legend to the trajectory plot.
    axislist[0].legend()
    # Explanation: This line adds a colourbar for cumulative estimated path length.
    figobject.colorbar(scatterobject, ax=axislist[0], label="Cumulative estimated path length m")
    # Explanation: This line creates frame indexes for local motion.
    framevalues = np.arange(len(localvalues))
    # Explanation: This line plots local translation over time.
    axislist[1].plot(framevalues, localvalues, color="blue", linewidth=1.4)
    # Explanation: This line fills the local translation curve area.
    axislist[1].fill_between(framevalues, localvalues, color="blue", alpha=0.18)
    # Explanation: This line labels the bottom plot x axis.
    axislist[1].set_xlabel("Frame")
    # Explanation: This line labels the bottom plot y axis.
    axislist[1].set_ylabel("Frame-to-frame translation m")
    # Explanation: This line sets the local motion title.
    axislist[1].set_title("Local motion stability")
    # Explanation: This line adds a grid to the local motion plot.
    axislist[1].grid(True, alpha=0.3)
    # Explanation: This line sets the figure title.
    figobject.suptitle(titletext)
    # Explanation: This line adjusts figure spacing.
    figobject.tight_layout()
    # Explanation: This line saves the figure.
    figobject.savefig(str(outputpath), dpi=180)
    # Explanation: This line closes the figure.
    plt.close(figobject)


# Explanation: This line starts the EvaluateSequence function.
def EvaluateSequence(sequenceobject, configdata):
    """
    Evaluate one Classic VO sequence.

    Step wise:
    1. Load Classic poses.
    2. Compute local motion.
    3. Save local plots and raw trajectory.
    4. Save no-ground-truth local stability visual.
    5. Optionally compute reference-based ATE and RPE when explicitly enabled.
    6. Return one summary row.
    """
    # Explanation: This line reads the sequence id.
    sequenceid = str(sequenceobject["id"])
    # Explanation: This line reads the pose dump path.
    posepath = sequenceobject["pose-dump"]
    # Explanation: This line creates the metrics output folder.
    outputdir = MakeFolder(sequenceobject["metrics-dir"])
    # Explanation: This line loads Classic camera poses.
    posearray = LoadPoses(posepath)
    # Explanation: This line extracts camera centers.
    centerarray = GetCenters(posearray)
    # Explanation: This line computes local motion rows.
    localrows = ComputeLocalMotion(posearray)
    # Explanation: This line saves local motion rows.
    SaveCsv(localrows, outputdir / "localmotion.csv")
    # Explanation: This line saves local translation plot.
    SaveLine([row["frame"] for row in localrows], [row["translationm"] for row in localrows], f"{sequenceid} Classic local translation", "Frame", "Translation m", outputdir / "localtranslation.png")
    # Explanation: This line saves local rotation plot.
    SaveLine([row["frame"] for row in localrows], [row["rotationdeg"] for row in localrows], f"{sequenceid} Classic local rotation", "Frame", "Rotation deg", outputdir / "localrotation.png")
    # Explanation: This line saves raw trajectory plot.
    SaveTrajectory(centerarray, f"{sequenceid} Classic VO trajectory", outputdir / "trajectory.png")
    # Explanation: This line saves the no-ground-truth pose stability visual.
    SaveMotionStabilityVisual(centerarray, localrows, f"{sequenceid} Classic estimated pose stability", outputdir / "trajectorystabilitymain.png")
    # Explanation: This line removes a stale reference-based drift visual from older runs.
    RemoveFile(outputdir / "trajectorydriftmain.png")
    # Explanation: This line reads Classic metric settings from the config.
    metricpart = configdata.get("classicmetrics", {})
    # Explanation: This line checks whether reference-based metrics are explicitly enabled.
    referencemetrics = bool(metricpart.get("enable-reference-metrics", False))
    # Explanation: This line starts the sequence summary object.
    summaryobject = {"id": sequenceid, "posecount": int(posearray.shape[0]), "rawpathlengthm": PathLength(centerarray), "localtranslation": Summarize([row["translationm"] for row in localrows]), "localrotation": Summarize([row["rotationdeg"] for row in localrows]), "referencemetricsenabled": referencemetrics}
    # Explanation: This line loads optional reference poses only when reference metrics are enabled.
    referenceposes = LoadReference(sequenceobject.get("reference-transforms-json", "")) if referencemetrics else None
    # Explanation: This line checks whether reference-based metrics can be computed.
    if referenceposes is not None and referencemetrics:
        # Explanation: This line extracts reference centers.
        referencecenters = GetCenters(referenceposes)
        # Explanation: This line aligns Classic centers to reference centers.
        alignedcenters, ateerrors, scalevalue, rotationmatrix, translationvector = AlignCenters(centerarray, referencecenters)
        # Explanation: This line aligns full Classic poses for RPE.
        alignedposes = AlignPoses(posearray, scalevalue, rotationmatrix, translationvector)
        # Explanation: This line creates ATE rows.
        aterows = [{"frame": int(indexvalue), "ateerrorm": float(errorvalue)} for indexvalue, errorvalue in enumerate(ateerrors)]
        # Explanation: This line saves ATE rows.
        SaveCsv(aterows, outputdir / "ate.csv")
        # Explanation: This line saves ATE line plot.
        SaveLine([row["frame"] for row in aterows], [row["ateerrorm"] for row in aterows], f"{sequenceid} Classic ATE", "Frame", "ATE m", outputdir / "ate.png")
        # Explanation: This line reads RPE intervals from config.
        intervals = metricpart.get("rpe-intervals", [1, 5, 10, 20])
        # Explanation: This line computes matched pose count.
        paircount = min(alignedposes.shape[0], referenceposes.shape[0])
        # Explanation: This line computes RPE rows.
        rperows = ComputeRpe(alignedposes[:paircount], referenceposes[:paircount], intervals)
        # Explanation: This line saves RPE rows.
        SaveCsv(rperows, outputdir / "rpe.csv")
        # Explanation: This line records ATE summary.
        summaryobject["ate"] = Summarize(ateerrors)
        # Explanation: This line records final drift.
        summaryobject["finaldrifterm"] = float(np.linalg.norm(alignedcenters[-1] - referencecenters[:alignedcenters.shape[0]][-1]))
        # Explanation: This line records aligned path length.
        summaryobject["alignedpathlengthm"] = PathLength(alignedcenters)
        # Explanation: This line records reference path length.
        summaryobject["referencepathlengthm"] = PathLength(referencecenters[:alignedcenters.shape[0]])
        # Explanation: This line records path length error.
        summaryobject["pathlengtherrorm"] = abs(summaryobject["alignedpathlengthm"] - summaryobject["referencepathlengthm"])
        # Explanation: This line records RPE rows in the summary.
        summaryobject["rpe"] = rperows
    # Explanation: This line checks whether reference-based metrics are disabled.
    if not referencemetrics:
        # Explanation: This line removes a stale ATE table from older reference-based runs.
        RemoveFile(outputdir / "ate.csv")
        # Explanation: This line removes a stale ATE plot from older reference-based runs.
        RemoveFile(outputdir / "ate.png")
        # Explanation: This line removes a stale RPE table from older reference-based runs.
        RemoveFile(outputdir / "rpe.csv")
    # Explanation: This line saves the sequence summary JSON.
    SaveJson(summaryobject, outputdir / "summary.json")
    # Explanation: This line returns a compact summary table row.
    return {"id": sequenceid, "posecount": summaryobject["posecount"], "rawpathlengthm": summaryobject.get("rawpathlengthm", ""), "localtranslationrmsem": summaryobject.get("localtranslation", {}).get("rmse", ""), "localrotationrmsedeg": summaryobject.get("localrotation", {}).get("rmse", ""), "referencemetricsenabled": summaryobject.get("referencemetricsenabled", False)}


# Explanation: This line starts the ParseArgs function.
def ParseArgs():
    """
    Parse command line arguments.

    Step wise:
    1. Create a command-line parser.
    2. Add the config path argument.
    3. Return parsed arguments.
    """
    # Explanation: This line creates the command-line parser.
    parser = argparse.ArgumentParser(description="Generate metrics for Classic VO pose dumps.")
    # Explanation: This line adds the YAML config argument.
    parser.add_argument("config", help="Path to classicmetrics.yml")
    # Explanation: This line parses and returns arguments.
    return parser.parse_args()


# Explanation: This line starts the Main function.
def Main():
    """
    Run the Classic metrics workflow.

    Step wise:
    1. Parse command-line arguments.
    2. Load the config.
    3. Evaluate each configured sequence.
    4. Save one combined summary table.
    """
    # Explanation: This line parses command-line arguments.
    args = ParseArgs()
    # Explanation: This line loads the metrics config.
    configdata = LoadConfig(args.config)
    # Explanation: This line creates an empty summary row list.
    summaryrows = []
    # Explanation: This line loops through configured sequences.
    for sequenceobject in configdata.get("sequences", []):
        # Explanation: This line evaluates one sequence and appends its summary row.
        summaryrows.append(EvaluateSequence(sequenceobject, configdata))
    # Explanation: This line reads the combined metrics output root.
    outputroot = MakeFolder(configdata["output-root"])
    # Explanation: This line saves the combined summary table.
    SaveCsv(summaryrows, outputroot / "summarytable.csv")
    # Explanation: This line saves the combined summary JSON.
    SaveJson({"sequences": summaryrows}, outputroot / "summary.json")
    # Explanation: This line prints the output location.
    print(f"Classic metrics complete: {outputroot}")


# Explanation: This line runs the script entry point.
if __name__ == "__main__":
    # Explanation: This line calls the main workflow.
    Main()
