import cv2
import numpy as np

# Define 3D points in the world coordinate system (corresponding to the table tennis table)
# These points must correspond to the physical points on the table, measured accurately.
object_points =np.array([
                    [0,0,0],
                    [274,0,0],
                    [274,152.5,0],
                    [0,152.5,0],
                    [0,76.25,0],
                    [274,76.25,0],
                    [137,152.5,0], 
                    [137,0,0],
                    [137,-15.25,0],
                    [137,-15.25,15.25],
                    [137,167.75,15.25],
                    [137,167.75,0],
                    [137,76.25,0],
                    [137,76.25,15.25],
                    [0,114.375,0],
                    [0,38.125,0],
                    # [274,38.125,0],
                    # [274,114.375,0],  #18
                    # [38.6,23.1,-76],  #19
                    # [38.6,129.40,-76],  #20
                ], np.float32)

# Define corresponding 2D points in the image coordinate system (these should be the pixel coordinates in the image)
image_points = np.array([
                    [1250, 719],      # Corresponding to Point 1
                    [1420, 295],      # Corresponding to Point 2
                    [897, 284],       # Corresponding to Point 3
                    [431, 660],       # Corresponding to Point 4
                    [812, 688],       # Corresponding to Point 5
                    [1153, 288],      # Corresponding to Point 6
                    [714, 426],       # Corresponding to Point 7
                    [1364, 450],      # Corresponding to Point 8
                    [1437, 453],      # Corresponding to Point 9
                    [1426, 387],      # Corresponding to Point 10
                    [659, 361],       # Corresponding to Point 11
                    [651, 419],       # Corresponding to Point 12
                    [1021, 445],       # Corresponding to Point 13
                    [1024, 381],       # Corresponding to Point 14
                    [612, 673],       # Corresponding to Point 15
                    [1020, 704],       # Corresponding to Point 16
                    # [1283, 292],       # Corresponding to Point 17
                    # [1023, 286],       # Corresponding to Point 18
                    # [1177, 973],       # Corresponding to Point 18
                    # [674, 910],       # Corresponding to Point 18
                ], dtype=np.float32)
    
# Define the camera intrinsic parameters (this should be pre-determined from a calibration process)
camera_matrix = np.array([
    [1920, 0, 960],
    [0, 1080, 540],
    [0, 0, 1]
], dtype=np.float32)

# Assuming no lens distortion for simplicity
dist_coeffs = np.zeros(4)

# SolvePnP to get initial pose estimation
success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

if success:
    print("Initial Rotation Vector:\n", rvec)
    print("Initial Translation Vector:\n", tvec)
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    print(rotation_matrix)
    
    # Refine the pose estimation using Levenberg-Marquardt optimization
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if success:
        print("Refined Rotation Vector:\n", rvec)
        print("Refined Translation Vector:\n", tvec)
        
        # Project the 3D points back to 2D to calculate reprojection error
        projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        projected_points = projected_points.reshape(-1, 2)
        
        # Calculate reprojection error
        reprojection_error = np.sqrt(np.mean(np.square(image_points - projected_points)))
        print("Reprojection Error: ", reprojection_error)
    else:
        print("Refinement using Levenberg-Marquardt optimization failed.")
else:
    print("Initial solvePnP failed.")
