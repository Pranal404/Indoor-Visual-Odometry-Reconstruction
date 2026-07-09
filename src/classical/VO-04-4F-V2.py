from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from collections import deque 
from openni import openni2 
import numpy as np
import open3d as o3d
import glob
import time
import cv2
import os


def astraopenni(timeout_ms=100):
    openni2.initialize() #Load openin drivers/plugins. 
    device = openni2.Device.open_any() # Open first available device

    # Create two stream objects for depth and color
    depthst = device.create_depth_stream()
    colorst = device.create_color_stream()
    
    try:
        device.set_depth_color_sync_enabled(True) # Firmware to time-sync depth and color streams
         # Align depth pixels into the color camera’s coordinate system (if available)
        device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR) 
    except Exception:
        pass
    
    # Start capturing deoth and color frames
    depthst.start()
    colorst.start()
       
    try:
        # Loop forever for frames
        while True: # While true runs forever unless interuppted with condition
            # If a strean is invalid yield 
            try:
                openni2.wait_for_any_stream([depthst, colorst], timeout_ms)
            except Exception:
                # timeout or transient error → no frame this tick
                yield None
                continue

            try:
                dframe = depthst.read_frame() # Read one depth frame
                cframe = colorst.read_frame() # Read one color frame
            except Exception:
                yield None
                continue

            # Convert the raw depth buffer (uint16, millimeters) into a 2D NumPy array
            # uint8 and unit16 unssigned integer 8bit color image 0-255 brightness
            fdepth = np.frombuffer(dframe.get_buffer_as_uint16(), dtype=np.uint16) \
                    .reshape(dframe.height, dframe.width)
            # Convert the raw color buffer (uint8, interleaved) into an H×W×3 array
            # HxWx3 3-> Channel 0 = Blue (B) | Channel 1 = Green (G) | Channel 2 = Red (R)
            #img[:,:,0] → entire channel 0. | img[:,:,1] → entire channel 1. | img[:,:,2] → entire channel 2.
            fcolor = np.frombuffer(cframe.get_buffer_as_uint8(), dtype=np.uint8) \
                    .reshape(cframe.height, cframe.width, 3)
            
            fdepthm = fdepth.astype(np.float32)/1000.0 # Convert depth tp meters 
            fmean = cv2.cvtColor(fcolor, cv2.COLOR_BGR2GRAY) #Convert color to graysacle

            # If color and depth resolution differ, we resize it to match
            if fmean.shape != fdepthm.shape: 
                height, width = fdepthm.shape #Taking depth size
                # resize fcolor as it will get recomputed to fmean.
                fcolor = cv2.resize(fcolor, (height,width), interpolation=cv2.INTER_AREA) #interpolate nearest neighbour.
            # return a tuple Tuple items are ordered, unchangeable, and allow duplicate values.
            yield fmean, fcolor, fdepthm     

   # stop streams and unload OpenNI to free the device even if an error occurs
    finally:
        # Stop the depth stream if it started successfully
        try: depthst.stop()
        except Exception: pass
        # Stop the color stream if it started successfully
        try: colorst.stop()
        except Exception: pass
        # Unload the OpenNI2 runtime (release drivers/resources)
        try: openni2.unload()
        except Exception: pass

def clahe(fmeanprev):
    if fmeanprev.dtype != np.uint8: #unit8 is unsigned integer, ranging from 0 to 255
        fmeanprev = cv2.normalize(fmeanprev, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)  
    clathe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))   
    fmeanprev = clathe.apply(fmeanprev)
    fmeanprev = cv2.GaussianBlur(fmeanprev, (3,3), 0)
    return fmeanprev

def laplacian(frame, lpthresh):
    # Opencv Laplacian
    #thresh=80.0 
    lap = cv2.Laplacian(frame, cv2.CV_64F) #-> 64bit float format.
    score = lap.var() # Laplacian variance -> Sharpness measure.
    
    # Threshold Check
    if score < lpthresh:
        return True, score # Blurry
    else:
        return False, score # Sharp
        
def goodfeature(fmeanprev, gfcornermax, gflevel):
    """
    Detect Shi–Tomasi corners on a grayscale image.
    Returns Nx1x2 float32 array as expected by OpenCV's LK.
    points = [[[x0, y0]],
              [[x1, y1]],
              .......... <- Nx1x2
    """
    pts = cv2.goodFeaturesToTrack(
        fmeanprev, 
        maxCorners=gfcornermax, 
        qualityLevel=gflevel, 
        minDistance=7,
        blockSize=7,
        useHarrisDetector=False)
    if pts is None:
        return np.empty((0, 1, 2), np.float32), np.empty((0,), dtype=np.float32) # No feature detect value 
    pts = pts.reshape(-1, 1, 2) # Reshape to (N,1,2)

    """Compute Shi-Tomsai response (min eigenvalue) over time corner strength 
    OpenCV maps are indexed like eigen[row][col]->x,y and NumPy arrays are indexed as:array[row,col]->y,x
    So we must read scores using eigen[y, x]"""
    eigen = cv2.cornerMinEigenVal(fmeanprev, blockSize=7, ksize=3)
   
    fh, fw = fmeanprev.shape 
    gridh , gridw = 8, 8 # single gridsize is Hxw -> 8x8
    # Divide frame h into grid having 8x8 cells Ex Image is 640x640, so each cell in grid is 640/8=80
    cellw = max(1, fw//gridw) # width of a cell in pixels 
    cellh = max(1, fh//gridh) # Height of a cell in pixels

    """Entire frame is split into 8x8 Grids, so create a list holding each grids
    Example of Grid 3x3 each [] holdes feature points for that grid of frame
    Grid: gridlist = [
    [ [], [], [], ],   # row 0 with three cells
    [ [], [], [], ],   # row 1 with three cells 
    [ [], [], [], ]    # row 2] with three cells
    """
    # Create a grid for frame, with cells Total cells: NxN 
    Grid = []
    for r in range(gridw):
        cellrow = []
        for c in range(gridh):
            cellrow.append([])
        Grid.append(cellrow)  

    #loop all the points pick them and place in their relative cells 
    for p in pts:
        # singe P which is (N,1,2) look like p=[[x0,y0],[x1,y1],...]-> p[0][0] = x-coordinate p[0][1] = y-coordinate
        x, y = int(p[0,0]), int(p[0,1])
        x = max(0, min(fw - 1, x)) # min(w - 1, x) ensures x ≤ w-1 and max(0, ...) ensures x ≥ 0
        y = max(0, min(fh - 1, y)) # min(w - 1, y) ensures y ≤ w-1 and max(0, ...) ensures y ≥ 0
        score = float(eigen[y ,x]) # Pick the eigen value of point[y, x]
        """cellx,celly is cell location where the relative feature point belongs
          Ex: Image width = 64 gridwidth = 8 → cellwidth = 80 point x = 230 cellx = 230 // 80 = 2
          Image height = 480 gridheight = 6 → cellheight = 80 point y = 310 celly = 310 // 80 = 3"""
        cellx = min(x // cellw, gridw -1) # min(., gridw-1) prevents going outside grid limits
        celly = min(y // cellh, gridh -1) 
        Grid[cellx][celly].append((x,y,score)) # Add [point x,y,score] at location cellx,celly

    cellpts = []
    cellscores = []
    sortpts = 12
    # Sort the points in cell and pick the top highest ones
    for r in range(gridw):
        for c in range(gridh):
            cell = Grid[r][c] #Pick the cell from Grid
            if not cell:
                continue
            # t[2] takes score[third] element from [x,y,score] and reverse sorts from highest to lowest.
            cell.sort(key=lambda t: t[2], reverse = True) 
            for x,y,score in cell[:sortpts]: # Take top 10 scores[points]
                cellpts.append([[x,y]]) # LK needs[[x, y]]  → shape (1,2) / (N,1,2)
                cellscores.append(score)
    if not cellpts:
        return np.empty((0,1,2), np.float32), np.empty((0,), np.float32)
    return np.array(cellpts, np.float32), np.array(cellscores, np.float32)

def Fastdetect(fmeant):
    #fast = cv2.FastFeatureDetector_create()
    fast = cv2.FastFeatureDetector_create(
    threshold=20,
    nonmaxSuppression=True,
    type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    """
    INFO:
    list of OpenCV KeyPoint objects outputs. Each KeyPoint contains many attributes [x, y, size, angle, etc.], and 
    only need coordinates [x,y]. list of tuples → each is one feature coordinate.
    convert that list into a NumPy array of type float32 required by OpenCV most geometry functions.
    Required shape: (N, 1, 2)  →  N points, each containing one (x, y) coordinate. |
    -1 Let NumPy automatically figure out how many points                      
     1 here is **one set of coordinates per point
     2  Each coordinate has 2 values (x and y)                                              
    """
    pts = fast.detect(fmeant, None) 
    Keypts = np.array([k.pt for k in pts], np.float32).reshape(-1,1,2)
    return Keypts

#.....LK OPTICAL NOTMAL........#
def lkoptical4(fmeant, fmeant1, fmeant2, fmeant3, fpts): 

    def empty():
        return (np.empty((0,2), np.float32),)*4 # Times four for all four frames. 

     # No points to track                                   
    if fpts is None or len(fpts) == 0:
        return empty()
    ptst0 = np.asarray(fpts, np.float32).reshape(-1, 1, 2) # reshape to (n,1,2)
   
    # Track pts from t to t+1 #pts1 approximated points
    ptst1, st1, e1 = cv2.calcOpticalFlowPyrLK(fmeant, fmeant1, ptst0, None)
    if ptst1 is None or st1 is None:
        return empty()
    #rravel() just flattens a NumPy array to 1-D
    # Then check Lk status returned to St with values 1 for “tracked OK” and 0 for “failed”.
    m = st1.ravel() == 1 
    ptst0, ptst1 = ptst0[m],ptst1[m] #Boolean check
    if len(ptst1) == 0:
        return empty()
    
    # Track pts from t to t+2 #pts2 approximated points
    ptst2, st2, e2 = cv2.calcOpticalFlowPyrLK(fmeant1, fmeant2, ptst1, None)
    if ptst2 is None or st2 is None:
        return empty()
    m = st2.ravel() == 1 
    ptst0, ptst1, ptst2 = ptst0[m],ptst1[m],ptst2[m] #second boolen check
    if len(ptst2) == 0:
        return empty()

    # Track pts from t to t+3 #pts3 approximated points
    ptst3, st3, e3 = cv2.calcOpticalFlowPyrLK(fmeant2, fmeant3, ptst2, None)
    if ptst3 is None or st3 is None:
        return empty()
    m = st3.ravel() == 1 
    # Applying boolean mask at every frame for all pts from previous frames assures
    # That only true points stay in all the frames
    ptst0,ptst1,ptst2,ptst3 = ptst0[m],ptst1[m],ptst2[m],ptst3[m] # Third boolean check
    if len(ptst3) == 0:
        return empty()
    
    # Reshape to (N,2)
    return ptst0.reshape(-1,2),ptst1.reshape(-1,2),ptst2.reshape(-1,2),ptst3.reshape(-1,2) 

def lkoptical4fil(fmeant, fmeant1, fmeant2, fmeant3, fpts, magthresh, errthresh): 

    def empty():
        return (np.empty((0,2), np.float32),)*4 # Times four for all four frames. 
    # No points to track
    if fpts is None or len(fpts) == 0:
        return empty()
    
    def filter(fprev, fcurr, st, err):
        """ fprev: list of [ptst0, ptst1, ..., ptstK] each (N,1,2)
            fcurr: pts in next frame (N,1,2)
            st, err: LK status and error for that step"""
        if fcurr is None or st is None or err is None:
            return None
        st = st.reshape(-1) # n dimension 
        err = err.reshape(-1)
        # For boolean masking match we shape it to (-1, 2) from (-1, 1, 2)
        fprevlast = fprev[-1].reshape(-1, 2) # Tracks(fprev) is list -1 tkaes latest array of pts
        fcurr = fcurr.reshape(-1, 2)

        # Check for succesfully traced points
        stt = (st == 1)
        if not np.any(stt):
            return None
        p0 = fprevlast[stt]
        p1 = fcurr[stt]
        e = err[stt] 

        if p1.shape[0] == 0: # O-> Takes Row-wise to check 
            return None

        # Magnitude compute difference of points tracked motion estimation distnac too high reject
        mag = np.linalg.norm(p1 - p0, axis=1)
        if mag.size == 0:
            return None
        magmedian = np.median(mag)
        sustain = (e < errthresh) & (mag < magthresh * (magmedian + 1e-6))
        if not np.any(sustain):
            return None
        
        fprevn = []
        # Loop to take all points from previous frame so only points which survive are consistent in all frames.
        # boolean mask is applied consistently to all previous frames 
        for f in fprev:
            # Apply both stt and mag boolean and reshape for cv2
            fn = f.reshape(-1, 2)[stt][sustain].reshape(-1, 1, 2) 
            fprevn.append(fn) 
        fcurrn = p1[sustain].reshape(-1, 1, 2) # This becomes previous frame for next tracking

        return fprevn, fcurrn
    
    # Initial points from frame t
    ptst0 = np.asarray(fpts, np.float32).reshape(-1, 1, 2)
    tracks = [ptst0] # create List:tracks[]
   
    # Track pts from t to t+1 #pts1 approximated points 
    # Python list: tracks[-1]-> takes latest in list, everytime we append a new index is created.
    ptst1, st1, e1 = cv2.calcOpticalFlowPyrLK(fmeant, fmeant1, tracks[-1], None)
    filpts = filter(tracks, ptst1, st1, e1)
    if filpts is None:
        return empty()
    # Replace the old track list with the newly filtered version: fprevn, and pts1 is fcurrn(filtered version)
    # tracks becomes the new filtered list of previous frames, and ptst1 is the current frame, appended onto tracks.
    tracks, ptst1 = filpts #fprevn(list of previous frames), fcurrn
    tracks.append(ptst1) # append the fcurrn

    # Track pts from t to t+2 #pts2 approximated points
    ptst2, st2, e2 = cv2.calcOpticalFlowPyrLK(fmeant1, fmeant2, tracks[-1], None)
    filpts = filter(tracks, ptst2, st2, e2)
    if filpts is None:
        return empty()
    tracks, ptst2 = filpts # Replace the old track list with the newly filtered version
    tracks.append(ptst2)
 
    # Track pts from t to t+3 #pts3 approximated points
    ptst3, st3, e3 = cv2.calcOpticalFlowPyrLK(fmeant2, fmeant3, tracks[-1], None)
    filpts = filter(tracks, ptst3, st3, e3)
    if filpts is None:
        return empty()
    tracks, ptst3 = filpts # Replace the old track list with the newly filtered version
    tracks.append(ptst3)

    """tracks = [ptst0]  # after step 1
    tracks = [ptst0, ptst1] # after step 1 finished
    tracks = [ptst0, ptst1, ptst2] # after step 2 finished ans so on"""

    # Unpack
    ptst0, ptst1, ptst2, ptst3 = tracks

    return(
    ptst0.reshape(-1, 2),
    ptst1.reshape(-1, 2),
    ptst2.reshape(-1, 2),
    ptst3.reshape(-1, 2),)

# Pinhole Projection
def backproject(pixeluv, depthmeter, fx, fy, cx, cy, k=1):
 
    uv = np.asarray(pixeluv, dtype=np.float32).reshape(-1,2) #Convert uv in array and reshape(n,2)->n uv values for column u,v
    height, width = depthmeter.shape 
    xyz = np.empty((len(uv), 3), np.float32) # len uv is N as uv is (N,3) 

    # For all u,v values do cooresponding xyz values 
    for i, (u,v) in enumerate(uv): # indexing the uv pair
        ui = int (np.clip(u, 0, width-1))  # np.clip(value,low,high) clip upto width values, same for height
        vi = int (np.clip(v, 0, height-1)) 
        z = float(depthmeter[vi,ui]) # for every u,v pair have corresponding z values

        # checking for neighbourhood depth values
        #k will set a patch size and height is to limit patch upto boundary height of image
        vl = max(0, vi-k)
        vr = min(height, vi+k+1) 
        ul = max(0, ui-k)
        ur = min(width, ui+k+1)
        depthpatch = depthmeter[vl:vr, ul:ur]
        patchval = depthpatch[depthpatch > 0.0]
        z = float(np.median(patchval))if patchval.size else float ("nan")

        # Sensor depth values is not usuable, write nan and move to next pixel 
        if not np.isfinite(z) or z <= 0.0: # z is negative
            xyz[i] = [np.nan, np.nan, np.nan]
            continue

        # pinhole: X=(u-cx)/fx * Z, Y=(v-cy)/fy * Z, 
        x = (ui - cx) / fx * z
        y = (vi - cy) / fy * z
        xyz[i] = [x, y, z]

    return xyz

def forwardproject(point3d, Kp): # u = fx*X/Z + cx | # v = fy*Y/Z + cy

    fx, fy, cx, cy = Kp
    points = np.asarray(point3d, np.float32).reshape(-1, 3) # convert points into array shape (n,3)
    x,y,z = points[:,0], points[:,1], points[:,2] #slice individual values
    uv = np.empty((len(points), 2), np.float32) #len(points) is N, create (n,2) empty array
    vp = z > 0.0 # Check a valid point
    uv[:] = np.nan # If not default to nan
    uv[vp, 0] = fx * (x[vp] / z[vp]) + cx #write all x values
    uv[vp, 1] = fy * (y[vp] / z[vp]) + cy #write all y values
    return uv


def pnpransac(pts3d,p1lk,fx,fy,cx,cy):
    # Form intrinstic matrix
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    
    # Reshape for opencv -> (N, 1, 3), 1×3 triplet [X, Y, Z].
    pts3d = pts3d.reshape(-1,1,3).astype(np.float32)
     # -> (N, 1, 2), 1×2 pair [u, v].
    p1lk = p1lk.reshape(-1,1,2).astype(np.float32)
    
    #rvec 3×1 rotation vector, tvec 3×1 translation
    #pose True if RANSAC found a valid pose; False if it failed
    pose, rvec, tvec, inliers = cv2.solvePnPRansac( pts3d, p1lk, K, None,
        iterationsCount=100, reprojectionError=3.0, confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE)
    
    # P3P: 3 points (but gives up to 4 ambiguous poses → needs extra check/inliers)
    # EPnP/ITERATIVE: typically ≥4 points.
    if not pose or inliers is None or len(inliers) < 6:
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    return R.astype(np.float32), t.astype(np.float32), inliers.reshape(-1)

    #solvePnPRansac returns the rotation as an axis–angle vector rvec (shape 3×1 or (3,)):
    #Direction = rotation axis
    #Magnitude ||rvec|| = rotation angle (radians)
    #cv2.Rodrigues(rvec) converts that axis–angle into a 3×3 rotation matrix R.

def fbundle(frames):
    #deque allows to add and remove from both ends.
    # Hold max two items, prev and current frame 
    framequ = deque(maxlen=4)  
    for f in frames:
        if f is None:
            continue
        framequ.append(f)
        if len(framequ) == 4:
            yield framequ[0], framequ[1], framequ[2], framequ[3]  # return tuples of frame

    """
    INFO:
    Numpy array extracting Rotation Matrix  
    T = np.arange(16).reshape(4,4)
    T = [[ 0,  1,  2,  3],  Top-left 3×3: blk = T[:3, :3]
        [ 4,  5,  6,  7],   blk = [[ 0,  1,  2],
        [ 8,  9, 10, 11],          [ 4,  5,  6],
        [12, 13, 14, 15]]          [ 8,  9, 10]] Last column, rows 0..2:
    col = T[:3, 3]
    col = [ 3,  7, 11]   # shape (3,) — a 1D view of that column segment
    """

def RTmatrix(R, t, dtype=np.float32):
    RT = np.eye(4, dtype=dtype) # Cretes a 4x4 identity matrix
    RT[:3,:3] = R # assign top left 3x3 as R which is 3x3
    t = np.asarray(t).reshape(3) #np.assaray converts into n-dimension array we want shape 3
    # T[:3, 3] is a 1-D view of that column segment with shape (3,)
    RT[:3, 3] = t # assign last column as t which is 3x1
    return RT

def Reprojecterror(pts3dt:np.ndarray,obspts,RTs,
                   kp:tuple[float,float,float,float],
                   fheight,fwidth):

    P = np.array(pts3dt, dtype=np.float32).reshape(-1,3) ##d point -> N,3
    residual, masks = [],[]
    
    for (R,t), obs in zip(RTs,obspts):
        R = np.asarray(R, dtype=np.float32).reshape(3,3) # Rotation matrix 3x3
        t = np.asarray(t, dtype=np.float32).reshape(1,3) # Translation vector (3, )
        obs = np.asarray(obs, np.float32).reshape(-1,2)
        # Transform 3D points through RT -> equation:Xc = RP + t
        # P is tranposed to make it single column 3xN so as to matrix mul with 3x3 Rotation matrix
        # R@p.T is tranposed to get to Nx3 and then add t which is 3x1 to get Nx3
        Tpxyz = (R @ P.T).T + t 

        # Forward project the points to image plane to get u,v.
        Tpuv = forwardproject(Tpxyz, kp) # Nx2
        
        # Masking valid points
        # axis=1 means “operate across columns for each row.”
        # .all(axis=1) → reduce each row to a single boolean: True only if both x and y in that row are finite.
        # m is a boolean array of shape (N,) (one True/False per point).
        # m &= condition does an element-wise AND in place:

        m = np.isfinite(Tpuv).all(axis=1) & np.isfinite(obs).all(axis=1)
        if fheight is not None and fwidth is not None:
            # “keep row i only if Tpuv[i] and U_obs[i] are both valid (u,v) pixels.”
            m &= (Tpuv[:,0] >= 0) & (Tpuv[:,0] < fwidth) & \
                 (Tpuv[:,1] >= 0) & (Tpuv[:,1] < fheight)
            
        if not np.any(m):
            # No valiid points in this frame so store the empty residual
            residual.append(np.array([], dtype=np.float32)) 
            masks.append(m)
            continue

        # Per point reprojection error
        diff = obs[m] -Tpuv[m] #(M,2)
        error = np.linalg.norm(diff, axis=1) #(M, )
        # Everytime append a new index is created so id0-> t to t+1 errors, id1 id1-> t+1 to t+2 errors, id2-> t+2 to t+3 errors
        residual.append(error.astype(np.float32)) # 1D Vector of N residuals
        masks.append(m) # Full length Mask over N

    return residual, masks 
   
#......POSE REFINE......#.......TOO HEAVY...........
def poserefine(pts3dt, pt1lk, pt2lk, pt3lk, 
               R1, t1, R2, t2, R3, t3, kp, 
               fheight, fwidth, sigma, ptsmin):

    pts3dt = np.asarray(pts3dt, np.float32).reshape(-1, 3) #(N, xyz)->(N,3)
    pt1lk = np.asarray(pt1lk,np.float32).reshape(-1,2) #(N, XY)->(N,2)
    pt2lk = np.asarray(pt2lk,np.float32).reshape(-1,2) #(N, XY)->(N,2)
    pt3lk = np.asarray(pt3lk,np.float32).reshape(-1,2) #(N, XY)->(N,2)
    N = pts3dt.shape[0]  # 0-> in shape[0] takes size of number of rows, so N rows of columns XYZ
    if N == 0:    
        print("NO POINTS")
        return None

    # Get Errors and mask per reprojection
    obspts = [pt1lk,pt2lk,pt3lk]
    RTs = [(R1,t1),(R2,t2),(R3,t3)]
    errors, masks = Reprojecterror(pts3dt,obspts,RTs,kp,fheight,fwidth)
    if errors is None or len(errors) != 3:
        print("INVALID ERROR")
        return None
    
    # Unpack 
    elist = [] # error list
    mlist = [] # Mask list
    for e, m in zip(errors, masks):
    # Flatten: lays the array out in a single straight 1D line,in the same order the data already existed in memory
        earr = np.asarray(e, np.float32).flatten() # 1D array flatten turns into 1D array.
        marr = np.asarray(m, bool).reshape(-1) # 1D array
        if marr.shape[0] != N: # shape[0] row-wise, we check the matching
            print("MASK LENGTH MISMATCH")
            return None
        elist.append(earr)
        mlist.append(mlist)
    # unpack the values
    e1, e2, e3 = elist
    m1, m2, m3 = mlist

    """Each baseline error vector only contains errors for True entries in its mask
       EX: m1 = [ T, F, T, T, F ] | e1 = [ 0.1, 0.3, 0.2 ]
       e1[0] → corresponds to mask index 0 (P0)
       e1[1] → corresponds to mask index 2 (P2)
       e1[2] → corresponds to mask index 3 (P3) 
       There is no error for positions where m1[i] == False, 
       so e1 is NOT aligned with the original index space, as mask has both True and False, but
       e1 ,has values where mask is TRUE
       The reprojection error of a 3D point in frame t→t+1 must line up with the reprojection error 
       of the SAME 3D point in t→t+2 and t→t+3, before we can compare or threshold them.
       False gets filled with inf """
    
    e1full = np.full(N, np.inf, dtype=np.float32)
    e2full = np.full(N, np.inf, dtype=np.float32)
    e3full = np.full(N, np.inf, dtype=np.float32)

    """Ex: m1 = [T, F, T, T, F, T]     # True at indices [0, 2, 3, 5]
           e1 = [0.7, 0.3, 0.5, 0.4]   # errors for [0,2,3,5]
           e1_full = [inf, inf, inf, inf, inf, inf]
           e1_full[m1] = e1
           Result: index:       0     1     2     3     4     5
                    m1:         T     F     T     T     F     T
                    e1_full:   0.7   inf   0.3   0.5   inf   0.4"""
    e1full[m1] = e1
    e2full[m2] = e2
    e3full[m3] = e3

    # A point must be valid in all the frames
    mall = m1 & m2 & m3 # Global all frames
    if not np.any(mall):
        print("NO VALID POINTS")
        return None
    
    # Restrict errors to points valid in all baselines
    e1sel = e1full[mall]
    e2sel = e2full[mall]
    e3sel = e3full[mall]

    # Threshold based on median * sigma for each baseline 
    med1 = np.median(e1sel) if e1sel.size > 0 else np.inf
    med2 = np.median(e2sel) if e2sel.size > 0 else np.inf
    med3 = np.median(e3sel) if e3sel.size > 0 else np.inf
    if not np.isfinite(med1) or not np.isfinite(med2) or not np.isfinite(med3):
        print("NO MEDIANS")
        return None
    thr1 = sigma * (med1 + 1e-6)
    thr2 = sigma * (med2 + 1e-6)
    thr3 = sigma * (med3 + 1e-6)

    # Among mall points, keep those below threshold for all three errors sustainlc: local
    sustainlc = (e1sel < thr1) & (e2sel < thr2) & (e3sel < thr3)
    if not np.any(sustainlc):
        print("NO POINTS")
        return None

    # Map sustain (over mall) back to global N-length mask sustaingb: gloabal
    sustaingb = np.zeros(N, dtype=bool)
    sustaingb[mall] = sustainlc

    """
    When you index ALL arrays using the SAME global boolean mask,They all select the SAME 3D points,
    In the SAME order, Preserving perfect alignment across:
    3D points (pts3dt) | 2D points at t+1 (pt1lk) |2D points at t+2 (pt2lk)2D points at t+3 (pt3lk)
    """
    pts3dtref = pts3dt[sustaingb]
    pt1ref    = pt1lk[sustaingb]
    pt2ref    = pt2lk[sustaingb]
    pt3ref    = pt3lk[sustaingb]

    if pts3dtref.shape[0] < ptsmin:
        print(f"POSE:{pts3dtref.shape[0]} FILTERED POINTS")
        return None

    # Re-run PnP on refined inliers -----
    R1ref, t1ref, _ = pnpransac(pts3dtref, pt1ref, *kp)
    R2ref, t2ref, _ = pnpransac(pts3dtref, pt2ref, *kp)
    R3ref, t3ref, _ = pnpransac(pts3dtref, pt3ref, *kp)

    if R1ref is None or R2ref is None or R3ref is None:
        print("PNP FAILED")
        return None

    # Recompute erros with refined RTs (for logging/gating) -----
    obsptsref = [pt1ref, pt2ref, pt3ref]
    RTsref    = [(R1ref, t1ref), (R2ref, t2ref), (R3ref, t3ref)]

    errorsref, masksref = Reprojecterror(pts3dtref,obsptsref,RTsref,kp,fheight,fwidth)

    return (R1ref, t1ref,
            R2ref, t2ref,
            R3ref, t3ref,
            pts3dtref, pt1ref, pt2ref, pt3ref,
            errorsref)
#......POSE REFINE......#  

#...SOCHASTIC OPTIMISATION
def poseoptim( Rin, tin, pts3dt, pt1lk, kp,fheight, fwidth, poniters,
                                              rotsigma=1e-3,
                                            transsigma=1e-3):

    pts3dt = np.asarray(pts3dt, np.float32).reshape(-1, 3) #(N, 3)
    pt1lk  = np.asarray(pt1lk,  np.float32).reshape(-1, 2) #(N, 2)
    N = pts3dt.shape[0]  # 0-> in shape[0] takes size of number of rows, so N rows of columns XYZ
    if N == 0:    
        print("NO POINTS")
        return Rin, tin #  Initial pose t -> t+1 from pnpransac
    
    """
    Rotation matrices (3×3) cannot be directly perturbed — adding noise to R breaks
    orthogonality and produces an invalid rotation, so we convert it into rotation matrix.
    A rotation vector r = [rx, ry, rz] encodes:
    Axis  = Direction of r, direction of the vector
    Angle = |r|  (in radians), magnitude of the vector |r|
    cv2.Rodrigues converts: rvec (3×1)  <-->  R (3×3)
    optimize rotation as a simple 3-vector instead of a 3×3 matrix so it enables 
    small “wiggles” (perturbations) during pose refinement without breaking rotation constraints.

    Convert Rinit → rvecinit -> Add tiny noise to rvec (safe) -> Convert back rvec → Rtrial using Rodrigues
    Evaluate reprojection error Keep Rtrial if error improves
    """
    rvecin, _ = cv2.Rodrigues(Rin) # Input (3x3) Rotation Matrix
    rvecin = rvecin.reshape(3) # (3, 1) Rotation Matrix a single Vector
    tvecin = np.asarray(tin, np.float32).reshape(3) # Convert to (3, 1)
    
    # concatenate()`function to combine arrays of the same shape along a particular axis
    vecparam = np.concatenate([rvecin, tvecin]) #

    def costfn(paramvec):
        rvec = paramvec[:3].reshape(3, 1) # Extract Rotation 
        tvec = paramvec[3:].reshape(3, 1) # Extract Translation
        
        # Convert back to rotation matrix for reprojection error
        #_ because cv2 returns R and t so rmatrix = R, _ is t, but we need R
        rmatrix, _ = cv2.Rodrigues(rvec) 
        tmatrix = tvec # tvec is (3,1) should be fine

        RTmatrix = [(rmatrix, tmatrix)]
        obsptstl = [pt1lk]

        errors, mask = Reprojecterror(pts3dt, obsptstl, RTmatrix, kp, fheight, fwidth)

        if errors is None or len(errors) == 0 or errors[0] is None: #[0] -> Rows
            return float("inf")

        e = np.asarray(errors[0], np.float32).flatten()
        if e.size == 0:
            return float("inf")

        return float(np.mean(e**2))

    # Start from initial Pose #costref amd paramref is actual reprojection error
    # which is used as truth to compare 
    paramrvec = vecparam.copy() # Actual Pose converted to Rvec
    costrvec = costfn(paramrvec) # Mean of Actual Pose Rvec

    for _ in range(poniters):
        para = paramrvec.copy() # Copy Actual Pose Rvec
        # np.random.randn(3) generates a 3-element vector  from a standard normal distribution
        para[:3] += np.random.randn(3) * rotsigma # Tweaked or Nudged Rotation part of Pose on Rvec
        para[3:] += np.random.randn(3) * transsigma # # Tweaked or Nudged Translation part of Pose on Rvec

        c = costfn(para) # The tweaked or nudged Pose
        if c < costrvec:
            costrvec = c
            paramrvec = para # If tweaked is less use tweaked para
    
    # Convert paramsrvec back to R, t, from concatenation 
    rvecless = paramrvec[:3].reshape(3, 1) # Rotation
    tvecless = paramrvec[3:].reshape(3, 1) # Translation 

    Rless, _ = cv2.Rodrigues(rvecless) # Rotation component
    tless = tvecless # 
  
    return Rless, tless

#...SOCHASTIC OPTIMISATION

def errorscore(errors):
    medians = [] # This will stack error triplets
    for err in errors: # errors are the triplets
        if err is None:
            medians.append(np.inf) # infinite: To treat it as worst possibe error, as no number we ever be larger.
            continue
        # Create a array of all errors and flatten it to single array for each error of triplet
        arr = np.asarray(err).flatten() 
        if arr.size == 0:
            medians.append(np.inf)
        else:
            medians.append(float(np.median(arr)))

    med1, med2, med3 = medians #unpack to variables
    score = max(med1, med2, med3)
    return score, (med1,med2,med3)  
    
def ploterror(residual, stepidx=None,   
             labels=("T→T+1", "T→T+2", "T→T+3"),
             title="4F:REPROJECTION ERROR:TIME"):
    
    M1, M2, M3 = [], [], []
    for triplet in residual:
        if triplet is None  or len(triplet) !=3:
            M1.append(np.nan)
            M2.append(np.nan)
            M3.append(np.nan)
            continue
        vals = []
        for r in triplet:
            if r is None:
                vals.append(np.nan)
                continue      
            arr = np.asarray(r).flatten()
            if arr.size == 0:
                vals.append(np.nan)
            else:
                vals.append(float(np.median(arr)))   
        val1, val2, val3 = vals
        M1.append(val1)
        M2.append(val2)
        M3.append(val3)
    M1 = np.array(M1, dtype=float)
    M2 = np.array(M2, dtype=float)
    M3 = np.array(M3, dtype=float)  

    if stepidx is None:
            stepidx = np.arange(len(M1))  
    plt.figure()
    plt.plot(stepidx, M1, linewidth=2, label=labels[0])
    plt.plot(stepidx, M2, linewidth=2, label=labels[1])
    plt.plot(stepidx, M3, linewidth=2, label=labels[2])
    plt.xlabel("4F INDEX [ANCHOR FRAME T)")
    plt.ylabel("REPROJECTION ERROR [PX, MEDIAN]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("RP Error 4F Refined.png", dpi=200)

#Pose Trajectory
class TrajectoryLogger:
    def __init__(self):
        self.T = np.eye(4, dtype=float) #Identity matrix
        self.centers = [self.T[:3, 3].copy()]

    def step(self, RT, invert=False):
        RT = np.asarray(RT, dtype=float)
        if invert:
            RT = np.linalg.inv(RT) 
        self.T = self.T @ RT
        self.centers.append(self.T[:3, 3].copy())

    def writeply(self, outpath):
        C = np.asarray(self.centers, dtype=float)
        N = len(C)
        with open(outpath, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {N}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element edge {max(0, N-1)}\n")
            f.write("property int vertex1\nproperty int vertex2\n")
            f.write("end_header\n")
            for x, y, z in C:
                f.write(f"{x} {y} {z}\n")
            for i in range(N - 1):
                f.write(f"{i} {i+1}\n")

# Plot Data Detailed
class  Visualisers:
    # Methods: plot CDF | tracking survivial tracks | Trajectory 2d & 3D
    
    def __init__(self, Title = "VO"):
        self.Title = Title
        
    # Compute the CFD Methdod
    def cdf(self,errors):
        """
        errors: 1D array-like of reprojection errors
        returns: sorted errors, cdf values
        """
        errors = np.asarray(errors).astype(np.float64)
        errors = errors[np.isfinite(errors)] # Make the erros Finite
        errors = np.abs(errors) # Take absolute values of errors
        if errors.size == 0:
            return np.array([]), np.array([])
        errorssort = np.sort(errors)
        # Linspace: creates a sequence of numbers where the difference between any two consecutive numbers is constant.
        cdf = np.linspace(0.0, 1.0, errorssort.size, endpoint=True)
        return errorssort, cdf
    
    # CDf :2F
    def plotcdf(self, errorstack, label = "4F", show = True, path = None):
        """ Errorstack: list of arrays/lists of reprojection errors per frame
        e.g. residual_history[t] = [e0, e1, ..., eN] """
        
        # Flatten all errors into a one big vector
        allerr = [], [], []
        for triplet in errorstack:
            if triplet is None:
                continue
            for k in range(3):
                r = triplet[k]
                if r is None:
                    continue
                arr = np.asarray(r).ravel()
                allerr[k].append(arr)

        plt.figure()
        colors = ["tab:blue", "tab:orange", "tab:green"]
        for k in range(3):
            if not allerr[k]:
                continue
            errs = np.concatenate(allerr[k])
            x, cdf = self.cdf(errs)
            plt.plot(x, cdf, label=label[k], color=colors[k])

        plt.xlabel("REPROJECTION ERROR[PX]")
        plt.ylabel("CDF")
        plt.title(f"{self.Title} – CDF REPROJECTION ERROR[4F]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        if path is not None:
            plt.savefig(path, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    # Tracking Survival Curve 
    def plottracksurvive(self, trackcts,
                               xlabel="Frame Index",
                               ylabel="Number of Tracked Points",
                               show=True, path=None):
        """trackcts (dict):
            dict {label: 1D array/list of track counts per frame}
            Example for one method:
                trackcts = { "2F refined": trackcts,
                          "4F refined": trackctslastframe}"""
        plt.figure()
        for label, counts in trackcts.items():
            counts = np.asarray(counts)
            frames = np.arange(len(counts))
            plt.plot(frames, counts, label=label)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{self.Title} – TRACK SURVIVAL:FEATURE RETENTION RF")
        plt.grid(True, alpha=0.3)
        plt.legend()
        if path is not None:
            plt.savefig(path, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    # 2D Top-Down Trajectory 
    def plottrack2d(self, centers, label="TRAJECTORY",
                           xy=("x", "z"),
                           show=True, path=None):
        
        """ centers: array-like of shape (N,3) with camera centers in world coordinates
                 e.g. centers[n] = [Xn, Yn, Zn]
        xy: which coordinates to plot, default ("x","z") means (X,Z)"""

        centers = np.asarray(centers)
        if centers.ndim != 2 or centers.shape[1] != 3:
            print("[Plot Trajectory 2d]: Centers Must Be [N,3]")
            return

        # map "x","y","z" -> indices
        axismap = {"x": 0, "y": 1, "z": 2}
        ix = axismap[xy[0].lower()]
        iy = axismap[xy[1].lower()]

        plt.figure()
        plt.plot(centers[:, ix], centers[:, iy], "-", label=label)
        plt.xlabel(f"{xy[0].upper()} (m)")
        plt.ylabel(f"{xy[1].upper()} (m)")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.title(f"{self.Title} – PLAN:TRAJECTORY ({xy[0].upper()} vs {xy[1].upper()})")
        plt.legend()
        if path is not None:
            plt.savefig(path, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()
        
    # 3D Top-Down Tracjectory
    def plottrack3d(self, centers, label="TRAJECTORY",
                                show=True, path=None):
        
        """centers: array-like of shape (N,3) with camera centers
        Colors the points by time (0 -> 1)."""

        centers = np.asarray(centers)
        if centers.ndim != 2 or centers.shape[1] != 3:
            print("[plot Trajectory 3d]: Centers Must Be [N,3]")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        N = centers.shape[0]
        t = np.linspace(0.0, 1.0, N)  # normalized time

        sc = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                        c=t, cmap="viridis", s=5, label=label)
        # PLot connecting lines
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
                color="k", linewidth=0.5)

        ax.set_xlabel("X[m]", labelpad=5)
        ax.set_ylabel("Y[m]", labelpad=10)
        #ax.set_zlabel("Z[m]")
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 2 decimals
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax.set_title(f"{self.Title} – 3D TRAJECTORY")
        #fig.colorbar(sc, ax=ax, label="TIME NORMALIZED", pad = 0.13)

        if path is not None:
            plt.savefig(path, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()  

def main():
    # ORBECC ASTRA PARAMETERS
    fx = 580.0
    fy = 580.0
    cx = 320.0
    cy = 240.0
    kp = (fx,fy,cx,cy) 
    logger = TrajectoryLogger()
    errorstack = []
    trackcts = []
    """
    Extrinsics (R,t,T): where the camera is in 3D world → visualize this trail.
    Intrinsics (K,f): how 3D points map to the 2D image plane → 
    needed for PnP, backprojection, projection.
    Input the frames form astra via openni
    loop around four consetuctive frames  
    """
    # THRESHOLD TUNNING PARAMETERS:....................
    # Laplacian Threshold to reject blur images
    lpthresh = 80.0 
    # Max Corners for good feature..... 
    gfcornermax = 3500
    gfLevel = 0.015
    # Lkoptical4fil.................... 
    magthresh = 150.0
    errthresh = 20.0
    # Poseoptim 
    poniters = 30
    # THRESHOLD TUNNING PARAMETERS

    # Opencv display window
    cvname = "4F VO"
    cv2.namedWindow(cvname, cv2.WINDOW_NORMAL)         
    cv2.resizeWindow(cvname, 640, 480)

    # OPENNI: FRAMES INPUT
    frames = astraopenni(timeout_ms=100) 

    # FRAMES FOUR: SLIDING WINDOW -> 4F 
    for (framet, framet1, framet2, framet3) in fbundle(frames):
        fmeant, fcolort, fdeptht = framet
        fmeant1, fcolort1, fdeptht1 = framet1        
        fmeant2, fcolort2, fdeptht2 = framet2  
        fmeant3, fcolort3, fdeptht3 = framet3  
        """ For a color image, fcolort.shape is typically (height, width, channels)
        shape[:2] means “take the first two elements"""
        fheight, fwidth = fcolort.shape[:2]

        # CLAHE: CLIPPING TOO BRIGHT AND LOW VAlUES OF INTESITY HISTOGRAM ON FRAMES
        fmeant = clahe(fmeant)
        fmeant1 = clahe(fmeant1)
        fmeant2 = clahe(fmeant2)
        fmeant3 = clahe(fmeant3)

        # BLUR: LAPLACIAN HIGH SHARP CORNERS LOW SMOOTH CORNERS
        br, scoref = laplacian(fmeant, lpthresh)
        br1, scoref1 = laplacian(fmeant1, lpthresh)
        br2, scoref2 = laplacian(fmeant2, lpthresh)
        br3, scoref3 = laplacian(fmeant3, lpthresh)
        if br or br1 or br2 or br3:
            # f-> formated string literal, varibale insde->{}, replace by its value. 
            print(f"BLUR FRAME: " f"scores = t:{scoref:.1f}, t+1:{scoref1:.1f},"
                                  f"t+2:{scoref2:.1f}, t+3:{scoref3:.1f}")   
            vis = fcolort3.copy()
            cv2.putText(vis, "BLUR FRAME", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(cvname, vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue

        # FEATURE SHI-TOMASI ON ANCHOR FRAME T 
        fpts, fscore = goodfeature(fmeant, gfcornermax, gfLevel) # Detect features 3 good feature 
        #fpts = Fastdetect(fmeant) # detect features FAST 
        if fpts is None or len(fpts) == 0:
            print("GRID FAIL ON T")
            vis = fcolort.copy()
            cv2.putText(vis, "NO FEATURES (t)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(cvname, vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue
            
        # LUCAS KANNDA: PREDICTING U,V DISPLACEMENT IN FRAME t+1,t+2,t+3    
        pt, pt1, pt2, pt3 = lkoptical4fil(fmeant, fmeant1, fmeant2, fmeant3, 
                                          fpts, magthresh, errthresh)
        if len(pt3) == 0: #If frame had nothing.
            print("NO 4F TRACKING")
            vis = fcolort3.copy()
            cv2.putText(vis, "NO TRACKS ", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(cvname, vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue
        print(f"FRAMES VALID -> t+3 :{len(pt3)}")

        # BACKPROJECT POINTS U,V TO XYZ IN FRAME t
        # * -> Python “spreads” the tuple into individual args (positional).
        bckptst = backproject(pt,fdeptht,*kp, k=1)
        #boolean mask of length N that’s True only where the 3D point is finite
        valid3d = np.isfinite(bckptst).all(axis=1) #Check if points are finite in boolean
        """NumPy boolean indexing, a.k.a. masking,
        valid3d is a (N,) boolean mask built from your back-projected 3D points (bckpts). 
        True means that track’s XYZ is finite (and often Z in-range).
        Applying it to p0 (and p1) keeps only the rows whose corresponding 3D was valid."""
        if not np.any(valid3d):
            print("NO VALID BCK 3D POINTS")
            continue
        pts3dt = bckptst[valid3d] # only valid ones checked from boolean 3D in t for pnp 
        ptvis = pt[valid3d] # Need for visualise.
        pt1lk = pt1[valid3d] 
        pt2lk = pt2[valid3d]
        pt3lk = pt3[valid3d]

        # PNP RIGID BODY TRANSFORMATION t-> 3D,2D TO t1,t2,t3
        R1, t1, inl1 = pnpransac(pts3dt,pt1lk,*kp) #inl are inliners returned use it maybe for pose refinement.
        R2, t2, inl2 = pnpransac(pts3dt,pt2lk,*kp)
        R3, t3, inl3 = pnpransac(pts3dt,pt3lk,*kp)
        if R1 is None or R2 is None or R3 is None:
            print("Pnp Failed Reuse Last Valid Pose")
            continue
        # ---- print pose ----
        #print("\nPose t -> t+1")
        #print("R =\n", R)
        #print("t =", t)
        
        #.....NOMINAL APPROACH.........................................................................#
        """# RT MATRIX WHOLE
        RT1 = RTmatrix(R1, t1) # For t->t1
        RT2 = RTmatrix(R2, t2) # For t->t2
        RT3 = RTmatrix(R3, t3) # For t->t3
        #print("RT1   |   RT2   |   RT3")
        #print(np.hstack((RT1, RT2, RT3)))

        # REPROJECTIOPN ERROR
        obspts = [pt1lk,pt2lk,pt3lk]
        RTs = [(R1,t1),(R2,t2),(R3,t3)]
        errors, mask = Reprojecterror(pts3dt,obspts,RTs,kp,fheight,fwidth)

        # MEDIAN SCORE
        score, (med1,med2,med3) = errorscore(errors)
        print(f"ERROR MEDIANS:t->t+1={med1:.4f}, t->t+2={med2:.4f}, t->t+3={med3:.4f}")
        errorstack.append(errors) # List of all triplets(errors:t-t+1,errors:t+1-t+2,errors:t+2-t+3)

        # PLOT CENTERS WITH RT INVERSE CAMERA TO WORLD AND FILTER BAD ONES
        Thresh = 0.10
        if np.isfinite(score) and score < Thresh:
            logger.step(RT1, invert=True)  # or False, depending on your RT convention
        else:
            print(f"VO BAD(SCORE={score:.4f}), SKIP POSE")"""
        #.....NOMINAL APPROACH.........................................................................#

        # POSE REFINE APPROACH.........................................................................#
        """#.....POSE REFINE: Refining Pose off the threshold and masking
        # and recaculating Pose on refined Pose. 
        # 4F MULTI-FRAME POSE REFINEMENT 
        posref = poserefine(
            pts3dt, pt1lk, pt2lk, pt3lk,
            R1, t1, R2, t2, R3, t3, kp, fheight,fwidth,sigma=3.0, ptsmin=20)
        if posref is None:
            print("refine_pose_4F: skipping this window")
            continue

        (R1ref, t1ref, R2ref, t2ref, R3ref, t3ref, 
        pts3dtref, pt1ref,pt2ref, pt3ref, errosref) = posref
        
        # REPROJECTIOPN ERROR WITH REFINED 
        obspts = [pt1ref,pt2ref,pt3ref]
        RTs = [(R1ref,t1ref),(R2ref,t2ref),(R3ref,t3ref)]
        errorsref, maskref = Reprojecterror(pts3dt,obspts,RTs,kp,fheight,fwidth)

        # RTref MATRIX
        RT1ref = RTmatrix(R1ref, t1ref) # For t->t1
        RT2ref = RTmatrix(R2ref, t2ref) # For t->t2
        RT3ref = RTmatrix(R3ref, t3ref) # For t->t3

        # MEDIAN SCORE
        score, (med1,med2,med3) = errorscore(errorsref)
        print(f"ERROR MEDIANS:t->t+1={med1:.4f}, t->t+2={med2:.4f}, t->t+3={med3:.4f}")
        errorstack.append(errorsref) # List of all triplets(errors:t-t+1,errors:t+1-t+2,errors:t+2-t+3)
        
        # PLOT CENTERS WITH RT INVERSE CAMERA TO WORLD AND FILTER BAD ONES
        Thresh = 0.10
        if np.isfinite(score) and score < Thresh:
            logger.step(RT1ref, invert=True)  # or False, depending on your RT convention
        else:
            print(f"VO BAD(SCORE={score:.4f}), SKIP POSE")"""
        #.......POSE REFINME.........................................................................#


        #.....STOCHASTIC OPTIMISATION APPROACH.......................................................#
        R1opt, t1opt = poseoptim( R1, t1, pts3dt, pt1lk, kp, fheight, fwidth,poniters,
                                                                    rotsigma=1e-3,
                                                                    transsigma=1e-3)

        # RT MATRIX WHOLE (using refined R1_opt,t1_opt, original R2,R3)
        RT1 = RTmatrix(R1opt, t1opt)  # refined t->t+1
        RT2 = RTmatrix(R2, t2)          # original t->t+2
        RT3 = RTmatrix(R3, t3)          # original t->t+3

        # REPROJECTION ERROR (4F), NOW USING REFINED T->T+1
        obspts = [pt1lk, pt2lk, pt3lk]
        RTs    = [(R1opt, t1opt), (R2, t2), (R3, t3)]
        errors, mask = Reprojecterror(pts3dt, obspts, RTs, kp, fheight, fwidth)

        # Compute overall median score
        score, (med1, med2, med3) = errorscore(errors)
        print(f"ERROR MEDIANS (REFINED t->t+1): "
              f"t->t+1={med1:.4f}, t->t+2={med2:.4f}, t->t+3={med3:.4f}")
        errorstack.append(errors)  # list of all triplets
        trackcts.append(len(pt3))

        # PLOT CENTERS WITH RT INVERSE CAMERA TO WORLD AND FILTER BAD ONES
        """Thresh = 0.10 * 2
        if np.isfinite(score) and score < Thresh:
            logger.step(RT1, invert=True)  # or False, depending on your RT convention
        else:
            print(f"VO BAD (SCORE={score:.4f}), SKIP POSE")"""
        #.....STOCHASTIC OPTIMISATION APPROACH.......................................................#
        logger.step(RT1, invert=True) # No Threshold

        # VISUALIZE ALL THE POINTS
        vis = fcolort1.copy()
        for (x0, y0), (x1, y1) in zip(ptvis, pt1lk):
            cv2.circle(vis, (int(x1), int(y1)), 2, (0, 255, 0), -1)
            cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 255), 1)
        cvname = "LK tracks (t -> t+1)"
        cv2.namedWindow(cvname, cv2.WINDOW_NORMAL)         
        cv2.resizeWindow(cvname, 600, 600)
        cv2.imshow(cvname, vis) 
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC to quit
            break
    cv2.destroyAllWindows()

    # WRITE CENTERS PLY 
    logger.writeply("Pose 4F Refined.ply") 

    # PLOT MEAN OF REPROJECTION ERRORS 
    ploterror(errorstack)

       # Visualise Data
    centers = np.array(logger.centers) # Centers 
    Vis = Visualisers(Title = "VO 4F RF")
    Vis.plotcdf(errorstack, label = "4F RF", path = "VO 4F CDF RF.png")
    Vis.plottracksurvive({"4F RF":trackcts}, path = "VO 4F TRACKS RF.png")
    Vis.plottrack2d(centers, label = "4F RF", path = "VO 4F TRAJ2D RF.png")
    Vis.plottrack3d(centers, label = "4F RF", path = "VO 4F TRAJ3D RF.png")

if __name__ == '__main__':
    main()
