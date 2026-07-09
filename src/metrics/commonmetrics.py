"""
Common metric helper functions.

This file contains shared utilities used by pose, depth, TSDF, and Gaussian
metric scripts. The goal is to keep loading, saving, plotting, and file matching
in one place so each metric file stays simple.

Python syntax notes:
- A function definition starts with the keyword `def`.
- A dictionary stores key-value pairs.
- A list stores ordered values.
- `Path` objects are safer than raw strings for filesystem paths.
- `return` sends a value back to the caller.
"""

# This line imports JSON support from Python's standard library.
import json
# This line imports math functions from Python's standard library.
import math
# This line imports operating-system helpers from Python's standard library.
import os
# This line imports Path, which represents filesystem paths clearly.
from pathlib import Path

# This line imports OpenCV for image and depth loading.
import cv2
# This line imports Matplotlib for saving plots.
import matplotlib.pyplot as plt
# This line imports NumPy for numeric arrays and matrix operations.
import numpy as np
# This line imports YAML support for reading configuration files.
import yaml


# This function reads a YAML configuration file and returns a Python dictionary.
def LoadConfig(configPath):
    # This line converts the input string into a Path object.
    pathObject = Path(configPath).expanduser()
    # This line opens the config file in read mode.
    with pathObject.open("r", encoding="utf-8") as fileObject:
        # This line safely parses YAML text into Python dictionaries and lists.
        configData = yaml.safe_load(fileObject)
    # This line returns the parsed configuration to the caller.
    return configData


# This function creates a directory if it does not already exist.
def MakeDir(folderPath):
    # This line converts the input into a Path object and expands "~" if present.
    pathObject = Path(folderPath).expanduser()
    # This line creates the directory and parent directories when needed.
    pathObject.mkdir(parents=True, exist_ok=True)
    # This line returns the created or existing directory path.
    return pathObject


# This function saves a Python dictionary as a readable JSON file.
def SaveJson(dataObject, outputPath):
    # This line converts the output path into a Path object.
    pathObject = Path(outputPath).expanduser()
    # This line ensures the parent output folder exists.
    MakeDir(pathObject.parent)
    # This line opens the output file in write mode.
    with pathObject.open("w", encoding="utf-8") as fileObject:
        # This line writes formatted JSON using two spaces of indentation.
        json.dump(dataObject, fileObject, indent=2)


# This function writes rows of dictionaries into a CSV text file.
def SaveCsv(rowList, outputPath):
    # This line converts the output path into a Path object.
    pathObject = Path(outputPath).expanduser()
    # This line ensures the parent output folder exists.
    MakeDir(pathObject.parent)
    # This line handles the case where there is no data to write.
    if len(rowList) == 0:
        # This line writes an empty file so the user sees that the metric ran.
        pathObject.write_text("", encoding="utf-8")
        # This line exits the function early because there are no rows.
        return
    # This line gets the CSV column names from the first dictionary.
    keyList = list(rowList[0].keys())
    # This line opens the output CSV file in write mode.
    with pathObject.open("w", encoding="utf-8") as fileObject:
        # This line writes the header row.
        fileObject.write(",".join(keyList) + "\n")
        # This loop writes one CSV row for each result dictionary.
        for rowObject in rowList:
            # This line converts each value into a string for CSV writing.
            valueList = [str(rowObject.get(keyName, "")) for keyName in keyList]
            # This line writes the comma-separated row.
            fileObject.write(",".join(valueList) + "\n")


# This function lists image files from a folder in sorted order.
def ListImages(folderPath):
    # This line converts the folder path into a Path object.
    pathObject = Path(folderPath).expanduser()
    # This line returns an empty list when the folder path is missing.
    if not pathObject.exists():
        # This line gives the caller a safe empty result.
        return []
    # This line defines common image filename extensions.
    suffixSet = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    # This line collects files whose suffix matches the image suffix set.
    imageList = [filePath for filePath in pathObject.iterdir() if filePath.suffix.lower() in suffixSet]
    # This line sorts paths lexicographically, which works for zero-padded frame names.
    imageList.sort()
    # This line returns the sorted image list.
    return imageList


# This function reads an RGB image from disk.
def ReadImage(imagePath):
    # This line reads the image with OpenCV in BGR channel order.
    bgrImage = cv2.imread(str(imagePath), cv2.IMREAD_COLOR)
    # This line handles failed reads explicitly.
    if bgrImage is None:
        # This line raises an error with the problematic file path.
        raise FileNotFoundError(f"Could not read image: {imagePath}")
    # This line converts OpenCV BGR order into normal RGB order.
    rgbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
    # This line returns the RGB image array.
    return rgbImage


# This function reads a depth image and converts it into meters.
def ReadDepth(depthPath, depthScale):
    # This line reads the depth file without changing its bit depth.
    rawDepth = cv2.imread(str(depthPath), cv2.IMREAD_UNCHANGED)
    # This line handles failed reads explicitly.
    if rawDepth is None:
        # This line raises an error with the problematic file path.
        raise FileNotFoundError(f"Could not read depth: {depthPath}")
    # This line converts integer depth into floating-point numbers.
    depthImage = rawDepth.astype(np.float32)
    # This line divides by the configured scale, usually 1000 for millimetres.
    depthImage = depthImage / float(depthScale)
    # This line returns metric depth in metres.
    return depthImage


# This function matches files from two folders by filename stem.
def MatchByStem(firstList, secondList):
    # This line builds a dictionary from filename stem to path for the second list.
    secondMap = {filePath.stem: filePath for filePath in secondList}
    # This line creates an empty list for matched file pairs.
    matchList = []
    # This loop checks every file in the first list.
    for firstPath in firstList:
        # This line looks for the same filename stem in the second folder.
        secondPath = secondMap.get(firstPath.stem)
        # This line keeps only pairs that exist in both folders.
        if secondPath is not None:
            # This line stores the matched pair.
            matchList.append((firstPath, secondPath))
    # This line returns matched file pairs.
    return matchList


# This function saves a simple line plot.
def SaveLinePlot(xValues, yValues, titleText, xLabel, yLabel, outputPath):
    # This line ensures the parent folder exists before plotting.
    MakeDir(Path(outputPath).expanduser().parent)
    # This line creates a new figure with a fixed size.
    plt.figure(figsize=(10, 4))
    # This line draws a line graph with small circular markers.
    plt.plot(xValues, yValues, marker="o", markersize=2, linewidth=1)
    # This line sets the plot title.
    plt.title(titleText)
    # This line labels the horizontal axis.
    plt.xlabel(xLabel)
    # This line labels the vertical axis.
    plt.ylabel(yLabel)
    # This line enables a light grid for readability.
    plt.grid(True, alpha=0.3)
    # This line tightens layout spacing before saving.
    plt.tight_layout()
    # This line saves the plot image to disk.
    plt.savefig(str(outputPath), dpi=160)
    # This line closes the figure to free memory.
    plt.close()


# This function saves a top-down X-Z camera trajectory plot.
def SaveTrajectoryPlot(centerList, labelText, outputPath):
    # This line converts camera centers into a NumPy array.
    centerArray = np.asarray(centerList, dtype=np.float64)
    # This line ensures the output folder exists.
    MakeDir(Path(outputPath).expanduser().parent)
    # This line creates a new figure.
    plt.figure(figsize=(6, 6))
    # This line draws the X-Z camera path when there are valid centers.
    if centerArray.size > 0:
        # This line plots X on the horizontal axis and Z on the vertical axis.
        plt.plot(centerArray[:, 0], centerArray[:, 2], marker="o", markersize=2, linewidth=1, label=labelText)
    # This line labels the X axis.
    plt.xlabel("X position")
    # This line labels the Z axis.
    plt.ylabel("Z position")
    # This line sets equal scale so trajectory shape is not distorted.
    plt.axis("equal")
    # This line enables a grid for readability.
    plt.grid(True, alpha=0.3)
    # This line shows the label in a legend.
    plt.legend()
    # This line tightens layout spacing.
    plt.tight_layout()
    # This line saves the trajectory plot.
    plt.savefig(str(outputPath), dpi=160)
    # This line closes the plot.
    plt.close()


# This function computes PSNR from mean squared error.
def ComputePsnr(firstImage, secondImage):
    # This line converts the first image to floating-point values in [0, 1].
    firstFloat = firstImage.astype(np.float32) / 255.0
    # This line converts the second image to floating-point values in [0, 1].
    secondFloat = secondImage.astype(np.float32) / 255.0
    # This line computes mean squared error between images.
    mseValue = float(np.mean((firstFloat - secondFloat) ** 2))
    # This line returns infinity when images are exactly identical.
    if mseValue <= 1e-12:
        # This line represents perfect reconstruction.
        return float("inf")
    # This line converts MSE into PSNR in decibels.
    psnrValue = 20.0 * math.log10(1.0 / math.sqrt(mseValue))
    # This line returns the PSNR value.
    return psnrValue


# This function creates a compact summary from numeric row values.
def SummarizeRows(rowList, keyName):
    # This line collects valid numeric values from the chosen column.
    valueList = [float(rowObject[keyName]) for rowObject in rowList if keyName in rowObject and rowObject[keyName] != ""]
    # This line handles empty metric columns.
    if len(valueList) == 0:
        # This line returns an empty summary when no values exist.
        return {"count": 0}
    # This line converts the Python list into a NumPy array.
    valueArray = np.asarray(valueList, dtype=np.float64)
    # This line returns count, mean, median, minimum, and maximum.
    return {
        "count": int(valueArray.size),
        "mean": float(np.mean(valueArray)),
        "median": float(np.median(valueArray)),
        "min": float(np.min(valueArray)),
        "max": float(np.max(valueArray)),
    }

