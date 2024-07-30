import numpy as np
import cv2

# Define your manually measured 3D and corresponding 2D points
# Example: Replace these with your actual data
# 3D points (world coordinates)
world_points=np.array([
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
    [274,38.125,0],
    [274,114.375,0],  #18
    # [38.6,23.1,-76],  #19
    # [38.6,129.40,-76],  #20
], np.float32)
# 2D points (image coordinates) ---------- problem in thissss......................
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
        [1283, 292],       # Corresponding to Point 17
        [1023, 286],       # Corresponding to Point 18
        # [1177, 973],       # Corresponding to Point 18
        # [674, 910],       # Corresponding to Point 18
    ], dtype=np.float32)
    

# image_points = np.array([[1044,605],[1296,483],[1044,450],[743,554],
#                         [886,578],[1164,446],[911,495],[1188,535],
#                         [1217,536],[1218,506],[888,464],[886,492]], dtype=np.float32)

# image_points = np.array([[940,599],[1165,489],[936,461],[668,553],
#                             [798,576],[1045,474],[816,502],[1065,536],
#                             [1096,539],[1096,570],[799,471],[799,497]], np.float32)

# image_points=np.array([[1137,560],[1354,439],[1097,397],[743,493],
#                         [884,525],[1218,416],[938,440],[1216,491],
#                         [1248,495],[1249,462],[917,404],[913,434]], np.float32)

# world_points=np.array([[0,0,0],[274,0,0],[274,152.5,0],[0,152.5,0],
#                     [0,76.25,0],[274,76.25,0],[137,152.5,0],[137,0,0],
#                     [137,-15.25,0],[137,-15.25,15.25],[137,167.75,15.25],[137,167.75,0]], np.float32)


# Initial camera matrix (intrinsic parameters, initial guess)

# [[2.36769566e+03, 0.00000000e+00, 8.42110659e+02],
#  [0.00000000e+00, 2.39270608e+03, 2.58483659e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

camera_matrix = np.array([[1920, 0, 960], 
                          [0, 1080 , 540], 
                          [0, 0, 1]], dtype=np.float32)

# camera_matrix = np.array([[2.36769566e+03, 0.00000000e+00, 8.42110659e+02],
#  [0.00000000e+00, 2.39270608e+03, 2.58483659e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

# Distortion coefficients (initial guess)
# dist_coeffs = np.array([[-0.050, -0.7996, -0.0547, -0.0106, 2.837]], dtype=np.float32)

# Initialize dist_coeffs as a zero matrix with the same shape
dist_coeffs = np.zeros((4, 1), dtype=np.float32)


retval, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs)
    
# Project 3D points back to image plane using estimated parameters
image_points_reprojected, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, dist_coeffs)
    
# Calculate reprojection error

"""If you need to assess the accuracy of each individual point's projection separately, reprojection_errors are more suitable. 
They provide detailed insights into how well each point's 3D position corresponds to its observed 2D position in the image."""

reprojection_errors = np.sqrt(np.sum((image_points_reprojected.squeeze() - image_points)**2, axis=1))
mean_reprojection_error = np.mean(reprojection_errors)
print("point by point Reprojection Error:", mean_reprojection_error)


"""If your main goal is to get a single, aggregate measure of how well your pose estimation method performs overall, 
error (RMSE) is more appropriate. It provides a holistic view of the accuracy across all points."""

error = np.sqrt(np.mean(np.square(image_points_reprojected - image_points)))
print("overall pose extimation Reprojection error:", error)

# Initialize variables for iterative refinement
error_prev = float('inf')
camera_matrix_updated = camera_matrix.copy()

while True:
    # Solve PnP problem to estimate pose (extrinsic parameters) and refine camera matrix
    retval, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix_updated, dist_coeffs)
    
    # Project 3D points back to image plane using estimated parameters
    image_points_reprojected, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix_updated, dist_coeffs)
    
    # Calculate reprojection error
    error = np.sqrt(np.mean(np.square(image_points_reprojected - image_points)))
    
    # Check if reprojection error has improved
    if error < error_prev:
        error_prev = error
        camera_matrix = camera_matrix_updated.copy()  # Update camera matrix
        
        # Get optimal new camera matrix
        image_width = 1920
        image_height = 1080
        camera_matrix_updated, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (image_width, image_height), 1, (image_width, image_height))
        
    else:
        break  # Exit loop if no further improvement
    
# Final results
print("Final Camera Matrix:\n", camera_matrix)
print("Final Reprojection Error:", error)
print("Rotation vector (rvec):\n", rvec.ravel())
print("Translation vector (tvec):\n", tvec.ravel())
        
R, _ = cv2.Rodrigues(rvec)
print("Rotation matrix (R):\n", R)

#this method will return you the extrensic matrix
def implement_solvepnp(world_points , image_points , camera_matrix , dist_coeffs):
    retval, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs) 
    
    image_points_reprojected, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, dist_coeffs)
    
    # Calculate reprojection error

    """If you need to assess the accuracy of each individual point's projection separately, reprojection_errors are more suitable. 
    They provide detailed insights into how well each point's 3D position corresponds to its observed 2D position in the image."""

    reprojection_errors = np.sqrt(np.sum((image_points_reprojected.squeeze() - image_points)**2, axis=1))
    mean_reprojection_error = np.mean(reprojection_errors)
    print("inital point by point Reprojection Error:", mean_reprojection_error)


    """If your main goal is to get a single, aggregate measure of how well your pose estimation method performs overall, 
    error (RMSE) is more appropriate. It provides a holistic view of the accuracy across all points."""

    error = np.sqrt(np.mean(np.square(image_points_reprojected - image_points)))
    # print("inital overall pose extimation Reprojection error:", error)
    
    error_prev = float('inf')
    camera_matrix_updated = camera_matrix.copy()

    while True:
        # Solve PnP problem to estimate pose (extrinsic parameters) and refine camera matrix
        retval, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix_updated, dist_coeffs)
        
        # Project 3D points back to image plane using estimated parameters
        image_points_reprojected, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix_updated, dist_coeffs)
        
        # Calculate reprojection error
        error = np.sqrt(np.mean(np.square(image_points_reprojected - image_points)))
        
        # Check if reprojection error has improved
        if error < error_prev:
            error_prev = error
            camera_matrix = camera_matrix_updated.copy()  # Update camera matrix
            
            # Get optimal new camera matrix
            image_width = 1920
            image_height = 1080
            camera_matrix_updated, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (image_width, image_height), 1, (image_width, image_height))
            
        else:
            # print("No further improvement. Stopping refinement.")
            break  # Exit loop if no further improvement
        
    # Final results
    # print("Final Camera Matrix:\n", camera_matrix)
    # print("Final Reprojection Error:", error)
    # print("Rotation vector (rvec):\n", rvec.ravel())
    # print("Translation vector (tvec):\n", tvec.ravel())
            
    # R, _ = cv2.Rodrigues(rvec)
    # print("Rotation matrix (R):\n", R)
    
    return R , rvec , tvec

    
    
