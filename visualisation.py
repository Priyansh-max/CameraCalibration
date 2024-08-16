import cv2
import numpy as np

# Load your pre-calibrated camera matrix and distortion coefficients
camera_matrix = np.array([[1728, 0, 864], 
                          [0, 1080 , 540], 
                          [0, 0, 1]], dtype=np.float32)  # Replace with your camera matrix
dist_coeffs = np.zeros((4, 1), dtype=np.float32) # Replace with your distortion coefficients

# Define your 3D object points (known coordinates of points on the object)
object_points=np.array([
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
    
], np.float32)

# Initialize the video capture
cap = cv2.VideoCapture('p1_serve01.mp4')  # Replace with your video file path
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through each frame in the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform feature detection and matching to get 2D image points
    # This part will depend on your specific object and how you detect and match features in your video frames.
    # For demonstration, let's assume you have pre-computed image points.
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
    ], dtype=np.float32)
    
    # Use SolvePnP to estimate the pose (rotation and translation)
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    if success:
        # Print or use rvec (rotation vector) and tvec (translation vector)
        print("Rotation Vector:\n", rvec)
        print("Translation Vector:\n", tvec)
        
        R, _ = cv2.Rodrigues(rvec)
        print("Rotation Matrix:\n", R)

        # Optionally, draw the axis representing the object in 3D space on the frame
        axis_length_x = 274 # Length of the axis in meters (adjust according to your object size)
        axis_length_y = 152.5
        axis_length_z = 20
        axis_points = np.float32([[0,0,0], [axis_length_x,0,0], [0,axis_length_y,0], [0,0,axis_length_z]]).reshape(-1,3)
        image_points1, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

        # Draw axis lines
        frame = cv2.line(frame, tuple(image_points1[0].ravel()), tuple(image_points1[1].ravel()), (0,0,255), 3)  # x-axis (red)
        frame = cv2.line(frame, tuple(image_points1[0].ravel()), tuple(image_points1[2].ravel()), (0,255,0), 3)  # y-axis (green)
        frame = cv2.line(frame, tuple(image_points1[0].ravel()), tuple(image_points1[3].ravel()), (255,0,0), 3)  # z-axis (blue)
        
        reprojected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        error = np.sqrt(np.mean(np.square(reprojected_points - image_points)))
        print("Reprojection Error:", error)

    # Display the frame with axis
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
