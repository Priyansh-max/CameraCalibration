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

    # print(cameraMatrix)

    # print(distCoeffs)

    # print(rvecs)
    
    # img_points_reprojected, _ = cv2.projectPoints(obj_points, rvecs, tvecs, cameraMatrix, distCoeffs)
    
    # reprojection_errors = np.sqrt(np.sum((img_points_reprojected.squeeze() - img_points)**2, axis=1))
    # mean_reprojection_error = np.mean(reprojection_errors)
    # print("point by point Reprojection Error:", mean_reprojection_error)
    
    # error = np.sqrt(np.mean(np.square(img_points_reprojected - img_points)))
    # print("overall pose extimation Reprojection error:", error)

    mean_error = 0
    for i in range(len(obj_points)):
        img_points_reprojected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
        error = cv2.norm(img_points[i], img_points_reprojected, cv2.NORM_L2) / len(img_points_reprojected)
        mean_error += error

    mean_error /= len(obj_points)
    # print(f"Mean reprojection error: {mean_error}")
    
    
    return cameraMatrix , distCoeffs

# Load calibration images
# images = loadimg.loadimage()

# print(images)


# cameraMatrix , distCoeffs = create_CM(images)

# print("final CMans :",cameraMatrix)
# print("FINAL dc ans :" , distCoeffs)