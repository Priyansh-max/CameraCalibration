import numpy as np
import cv2

# # Example values (replace with your actual matrices and points)
# extrinsic_matrix = np.array([[0.51, -0.85, 0.073, 34.35],
#                              [75.265, -58.233, -1203.721, 49.44],
#                              [0.793, 0.441, -0.420, 789.45]])

# world_points=np.array([[0,0,0],[274,0,0],[274,152.5,0],[0,152.5,0],
#                     [0,76.25,0],[274,76.25,0],[137,152.5,0],[137,0,0],
#                     [137,-15.25,0],[137,-15.25,15.25],[137,167.75,15.25],[137,167.75,0]], np.float32)

# intrinsic_matrix = np.array([[1728, 0, 864], 
#                           [0, 1080 , 540], 
#                           [0, 0, 1]], dtype=np.float32)

# distortion_coeffs = np.zeros((4, 1), dtype=np.float32)

# # Project 3D points to 2D image points
# image_points, _ = cv2.projectPoints(world_points, extrinsic_matrix[:, :3], extrinsic_matrix[:, 3], intrinsic_matrix, distortion_coeffs)
# # Example actual measured image points (replace with your actual data)
# actual_image_points = np.array([[940,599],[1165,489],[936,461],[668,553],
#                             [798,576],[1045,474],[816,502],[1065,536],
#                             [1096,539],[1096,570],[799,471],[799,497]], np.float32)


# # Calculate reprojection error
# reprojection_errors = np.sqrt(np.sum((image_points.squeeze() - actual_image_points)**2, axis=1))
# mean_reprojection_error = np.mean(reprojection_errors)
# print("point by point Mean Reprojection Error:", mean_reprojection_error)

# error = np.sqrt(np.mean(np.square(image_points - actual_image_points)))
# print("overall pose extimation Reprojection error:", error)

def checkerror_result(extrinsic_matrix , world_points ,image_points, intrinsic_matrix , distortion_coeffs):
    image_points_reprojected, _ = cv2.projectPoints(world_points, extrinsic_matrix[:, :3], extrinsic_matrix[:, 3], intrinsic_matrix, distortion_coeffs)
    
    reprojection_errors = np.sqrt(np.sum((image_points_reprojected.squeeze() - image_points)**2, axis=1))
    mean_reprojection_error = np.mean(reprojection_errors)

    error = np.sqrt(np.mean(np.square(image_points_reprojected - image_points)))
    
    return mean_reprojection_error , error

    
        
    
