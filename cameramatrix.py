
import numpy as np
import cv2
import loadimg

def create_CM(images):
# Define calibration pattern (e.g., a checkerboard)
    pattern_size = (8, 6)  # Number of inner corners in rows and columns

    # Arrays to store object points and image points from all images
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane

    # Initialize object points (coordinates of corners in the calibration pattern)
    objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find corners in the calibration pattern
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
            
    # Calibrate camera to find intrinsic and extrinsic parameters
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    mean_error = 0
    for i in range(len(obj_points)):
        img_points_reprojected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
        error = cv2.norm(img_points[i], img_points_reprojected, cv2.NORM_L2) / len(img_points_reprojected)
        mean_error += error

    mean_error /= len(obj_points)  
    return cameraMatrix , distCoeffs , mean_error




