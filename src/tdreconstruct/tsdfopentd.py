import cv2
import numpy as np
import open3d as o3d
from typing import Optional

class TSDFOpen3D:

    def __init__(
            self,
            # Each cube in the 3D grid is 3 cm × 3 cm × 3 cm.
            # Size of one voxel cube im meters.
            voxel_length: float = 0.03,
            sdf_trunc: float = 0.09, # voxels within a measured surface are updated.
            depth_trunc: float = 10.0, # Ignore any depth value greater than this distance.
            min_depth: float = 0.3,
            color_type: str = "rgb8", # RGB or None
            conf_thresh: float = 0.0, # If conf map is provided
    ):      
            self.voxel_length = float(voxel_length)
            self.sdf_trunc = float(sdf_trunc)
            self.depth_trunc = float(depth_trunc)
            self.color_type = str(color_type).lower() # lower: converts into lowercase
            self.conf_thresh = float(conf_thresh)
            self.min_depth = float(min_depth)

            if self.color_type not in ["rgb8", "none"]:
                raise ValueError("color_type must be 'rgb8' or 'none'")

            if self.color_type == "none":
                ct = o3d.pipelines.integration.TSDFVolumeColorType.NoColor
            else:
                ct = o3d.pipelines.integration.TSDFVolumeColorType.RGB8

            # Scalable volume is easiest (no need for fixed bounds)
            self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.voxel_length,
                sdf_trunc=self.sdf_trunc,
                color_type=ct,
            )

    def UseIntrinsic(self, K: np.ndarray, width: int, height: int):
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    def Integrate(
        self,
        depthm: np.ndarray, # numpy.ndarray: data type (class) used to represent a NumPy array.
        kip: np.ndarray,
        Twc: np.ndarray,
        colorbgr: Optional[np.ndarray] = None,
        conf: Optional[np.ndarray] = None,
    ):
        """ 
        depthm: (H,W) float32 depth in METERS
        K: (3,3) intrinsics
        Twc: (4,4) camera->world transform (extrinsic expected by Open3D)
        colorbgr: (H,W,3) uint8 BGR (OpenCV) or None
        conf: (H,W) float confidence in [0,1] or None """

        if depthm is None:
             return
        
        depth = np.asarray(depthm, dtype=np.float32)

        # Depth in min to max range 
        valid = (
            np.isfinite(depth) & 
            (depth >= self.min_depth) & 
            (depth <= self.depth_trunc))

        if depth.ndim != 2:
             raise ValueError(f"depth must be (H,W). Got {depth.shape}")
        
        H, W = depth.shape

        # Confidence Masking for Open3D to ingnore depth==0
        if conf is not None and self.conf_thresh > 0.0:
             c = np.asarray(conf, dtype=np.float32)
             if c.shape != depth.shape:
                  raise ValueError(f"conf shape {c.shape} != depth shape {depth.shape}")
             depth = depth.copy() # Original Input should remain unchanged
             depth[c < self.conf_thresh] = 0.0

        # Clamp the depth to a max range.
        """depth = np.where((depth > 0) & np.isfinite(depth) & 
        (depth <= self.depth_trunc), depth, 0.0).astype(np.float32)"""
        depth = np.where(valid, depth, 0.0).astype(np.float32)

        # Kill isolated speckles before integration
        """depthmm = (depth * 1000.0).astype(np.uint16)
        depthmm = cv2.medianBlur(depthmm, 5)
        depth = depthmm.astype(np.float32) / 1000.0"""

        # Keep the masked metric depth as-is for one clean test.
        # Median blur can oversmooth sparse stereo depth and spread bad neighborhoods.
        depth = depth.astype(np.float32)

        Intrinstic = self.UseIntrinsic(np.asarray(kip, dtype = np.float32), W, H)
        #Extrinsic = np.asarray(Twc, dtype= np.float64)
        Extrinsic = np.linalg.inv(np.asarray(Twc, dtype= np.float64))

        depth_o3d = o3d.geometry.Image(depth) # convert numpy depth array into open3d image object.

        if self.color_type == "none" or colorbgr is None:
             # If no color pass a dummy open3D RGBD object as Open3d needs it.
             dummyrgbd = np.zeros((H, W, 3), dtype = np.uint8)
             color_o3d = o3d.geometry.Image(dummyrgbd)
        else:
            color = np.asarray(colorbgr)
            if color.shape[:2] != (H, W): # Take first two H and W
                raise ValueError(f"color shape {color.shape} != depth shape {(H, W)}")
            if color.dtype != np.uint8:       
                color = color.astype(np.uint8)
            
            # Open3D expects RGB Order
            # -1 Reverses the third dimension as its written in third one so BGR -> RGB.
            colorrgb = color[:, :, ::-1].copy()
            color_o3d = o3d.geometry.Image(colorrgb)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
             color_o3d,
             depth_o3d,
             depth_scale = 1.0, # Depth already in Meters
             depth_trunc = self.depth_trunc,
             convert_rgb_to_intensity=False,
        )

        """Integrate the current RGBD frame into the TSDF volume.
         rgbd      -> depth + color image (Open3D object)
         intrinsic -> camera intrinsics (fx, fy, cx, cy)
         extrinsic -> camera pose (4x4 matrix, camera -> world)
         This updates the 3D voxel grid using the current frame."""

        self.volume.integrate(rgbd, Intrinstic, Extrinsic)

    def Extractmesh(self, mintriangles: int = 100):
            # Rxtract a triangle mesh from TSDF volume.
            mesh = self.volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()

            # Optional: Remove Tiny Isolated Components: 
            if mintriangles > 0:
                clusters, counts, _ = mesh.cluster_connected_triangles()
                counts = np.asarray(counts)
                clusters = np.asarray(clusters)
                keep = counts[clusters] >= int(mintriangles)
                mesh.remove_triangles_by_mask(~keep)
                mesh.remove_unreferenced_vertices()
                mesh.compute_vertex_normals()
            
            return mesh
    
    def savemesh(self, filepath: str, mesh):
            o3d.io.write_triangle_mesh(filepath, mesh)

    
                  



                  

         



