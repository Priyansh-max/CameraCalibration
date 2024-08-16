import cv2
import numpy as np

# Example source and destination points
# srcPoints = np.array([
#     [100, 200],  # Point in the original image
#     [300, 200],
#     [300, 400],
#     [100, 400]
# ], dtype=np.float32)

# dstPoints = np.array([
#     [0, 0],      # Corresponding point in the target plane
#     [2.74, 0],
#     [2.74, 1.525],
#     [0, 1.525]
# ], dtype=np.float32)

dstPoints = np.array([[0,0,0],[274,0,0],[274,152.5,0],[0,152.5,0],
                    [0,76.25,0],[274,76.25,0],[137,152.5,0],[137,0,0],
                    [137,-15.25,0],[137,-15.25,15.25],[137,167.75,15.25],[137,167.75,0]], np.float32)

srcPoints = np.array([[1126,648],[1503,470],[1163,422],[747,563],
                            [924,603],[1320,444],[983,483],[1339,546],
                            [1377,555],[1379,513],[949,441],[946,477]], np.float32)

# Compute homography using RANSAC
H, status = cv2.findHomography(srcPoints, dstPoints[:, :2], method=cv2.RANSAC)

print("Homography Matrix:\n", H)

srcPoints_homogeneous = np.hstack((srcPoints, np.ones((srcPoints.shape[0], 1))))

# Apply homography to source points
projected_points_homogeneous = np.dot(H, srcPoints_homogeneous.T).T
projected_points = projected_points_homogeneous[:, :2] / projected_points_homogeneous[:, 2:]

# Compute reprojection error
errors = np.sqrt(np.sum((projected_points - dstPoints) ** 2, axis=1))
mean_error = np.mean(errors)

import matplotlib.pyplot as plt

# Draw points on image
image = np.zeros((500, 500, 3), dtype=np.uint8)  # Example image

for pt in srcPoints:
    cv2.circle(image, tuple(pt.astype(int)), 5, (0, 0, 255), -1)

for pt in projected_points:
    cv2.circle(image, tuple(pt.astype(int)), 5, (255, 0, 0), -1)

# Show image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

