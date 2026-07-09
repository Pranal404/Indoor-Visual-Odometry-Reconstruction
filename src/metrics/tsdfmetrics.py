"""
TSDF output metrics.

This file reads a TSDF point cloud or mesh file and reports geometry statistics.
It is intentionally simple because most TSDF outputs in this project are judged
with visual point cloud or mesh screenshots plus basic geometry counts.

Python syntax notes:
- `try` and `except` let the script continue when an optional library is missing.
- A dictionary is a convenient way to store named metric values.
"""

# This line imports argparse so this file can run from the command line.
import argparse
# This line imports Path for filesystem paths.
from pathlib import Path

# This line imports NumPy for bounding-box calculations.
import numpy as np

# This line imports shared helpers.
from commonmetrics import MakeDir, SaveJson


# This function tries to load Open3D only when TSDF metrics are requested.
def LoadOpen3d():
    # This line starts an exception-protected import block.
    try:
        # This line imports Open3D under the common alias o3d.
        import open3d as o3d
        # This line returns the imported Open3D module.
        return o3d
    # This line catches import failures.
    except Exception:
        # This line returns None when Open3D is unavailable.
        return None


# This function computes geometry statistics from a point cloud or mesh file.
def ComputeTsdfStats(tsdfPath):
    # This line expands the TSDF file path.
    pathObject = Path(tsdfPath).expanduser()
    # This line creates a result dictionary with the path.
    resultObject = {"path": str(pathObject)}
    # This line handles missing or empty TSDF path.
    if str(tsdfPath).strip() == "" or not pathObject.exists():
        # This line records that the TSDF file was not found.
        resultObject["status"] = "missing"
        # This line returns the missing-file result.
        return resultObject
    # This line records file size in bytes.
    resultObject["filesizebytes"] = int(pathObject.stat().st_size)
    # This line attempts to import Open3D.
    o3d = LoadOpen3d()
    # This line handles missing Open3D.
    if o3d is None:
        # This line records that only file-size metrics are available.
        resultObject["status"] = "open3d_not_available"
        # This line returns the partial result.
        return resultObject
    # This line tries to read the file as a triangle mesh.
    meshObject = o3d.io.read_triangle_mesh(str(pathObject))
    # This line counts mesh vertices.
    vertexCount = len(meshObject.vertices)
    # This line counts mesh triangles.
    triangleCount = len(meshObject.triangles)
    # This line checks whether mesh loading succeeded.
    if vertexCount > 0:
        # This line converts mesh vertices into a NumPy array.
        pointArray = np.asarray(meshObject.vertices)
        # This line records mesh status.
        resultObject["status"] = "mesh"
        # This line records vertex count.
        resultObject["vertices"] = int(vertexCount)
        # This line records triangle count.
        resultObject["triangles"] = int(triangleCount)
    else:
        # This line reads the same file as a point cloud.
        cloudObject = o3d.io.read_point_cloud(str(pathObject))
        # This line converts point cloud points into a NumPy array.
        pointArray = np.asarray(cloudObject.points)
        # This line records point cloud status.
        resultObject["status"] = "pointcloud"
        # This line records point count.
        resultObject["points"] = int(pointArray.shape[0])
    # This line computes bounding box only when there are points.
    if pointArray.size > 0:
        # This line computes minimum XYZ coordinate.
        minPoint = np.min(pointArray, axis=0)
        # This line computes maximum XYZ coordinate.
        maxPoint = np.max(pointArray, axis=0)
        # This line computes bounding-box size.
        boxSize = maxPoint - minPoint
        # This line records minimum point.
        resultObject["bboxmin"] = minPoint.tolist()
        # This line records maximum point.
        resultObject["bboxmax"] = maxPoint.tolist()
        # This line records bounding-box size.
        resultObject["bboxsize"] = boxSize.tolist()
    # This line returns all TSDF statistics.
    return resultObject


# This function runs TSDF metrics and writes outputs.
def RunTsdfMetrics(configData):
    # This line creates the TSDF output folder.
    outputDir = MakeDir(Path(configData["output-dir"]) / "tsdf")
    # This line reads the configured TSDF file path.
    tsdfPath = configData["paths"].get("tsdf-file", "")
    # This line computes TSDF geometry statistics.
    summaryObject = ComputeTsdfStats(tsdfPath)
    # This line saves the statistics as JSON.
    SaveJson(summaryObject, outputDir / "tsdf_summary.json")
    # This line returns the summary to the master runner.
    return summaryObject


# This function reads the YAML config file.
def LoadConfig(configPath):
    # This line imports YAML only when the command-line workflow is used.
    import yaml
    # This line opens the config file.
    with Path(configPath).expanduser().open("r", encoding="utf-8") as fileObject:
        # This line returns parsed YAML.
        return yaml.safe_load(fileObject)


# This function parses command-line arguments.
def ParseArgs():
    # This line creates a command-line parser.
    parser = argparse.ArgumentParser(description="Compute simple TSDF mesh or point-cloud statistics.")
    # This line adds the required config argument.
    parser.add_argument("config", help="Path to tsdf metrics YAML config.")
    # This line returns parsed arguments.
    return parser.parse_args()


# This function runs the command-line TSDF metric workflow.
def Main():
    # This line reads command-line arguments.
    args = ParseArgs()
    # This line loads the YAML config.
    configData = LoadConfig(args.config)
    # This line runs the TSDF metrics.
    summaryObject = RunTsdfMetrics(configData)
    # This line prints the summary for terminal use.
    print(summaryObject)


# This line runs Main only when this file is executed directly.
if __name__ == "__main__":
    # This line starts the command-line workflow.
    Main()
