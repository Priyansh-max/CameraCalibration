import numpy as np
import cv2

# Define your manually measured 3D and corresponding 2D points
image_points = np.array([[940,599],[1165,489],[936,461],[668,553],
                            [798,576],[1045,474],[816,502],[1065,536],
                            [1096,539],[1096,570],[799,471],[799,497]], np.float32)

world_points=np.array([[0,0,0],[274,0,0],[274,152.5,0],[0,152.5,0],
                    [0,76.25,0],[274,76.25,0],[137,152.5,0],[137,0,0],
                    [137,-15.25,0],[137,-15.25,15.25],[137,167.75,15.25],[137,167.75,0]], np.float32)

# Initial camera matrix (intrinsic parameters, initial guess)
camera_matrix = np.array([[1728, 0, 864], 
                          [0, 1080 , 540], 
                          [0, 0, 1]], dtype=np.float32)

# Distortion coefficients (initial guess)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# Initialize variables for iterative refinement
error_prev = float('inf')
camera_matrix_updated = camera_matrix.copy()


retval, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs)
print("Translation vector (tvec):\n", tvec.ravel())

R, _ = cv2.Rodrigues(rvec)
print("Rotation matrix (R):\n", R)


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
        image_width = 1728
        image_height = 1080
        camera_matrix_updated, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (image_width, image_height), 1 , (image_width, image_height))
        
    else:
        print("No further improvement. Stopping refinement.")
        break  # Exit loop if no further improvement
    
# Final results
print("Final Camera Matrix:\n", camera_matrix)
print("Final Reprojection Error:", error)
print("Rotation vector (rvec):\n", rvec.ravel())
print("Translation vector (tvec):\n", tvec.ravel())
        
R, _ = cv2.Rodrigues(rvec)
print("Rotation matrix (R):\n", R)
