import numpy as np

def compute_errors(reprojected_points, actual_image_points):
    errors = reprojected_points - actual_image_points
    mse_x = np.mean(errors[:, 0] ** 2)
    mse_y = np.mean(errors[:, 1] ** 2)

    return {'MSE_X': mse_x, 'MSE_Y': mse_y}

# Example usage
camera_matrix = np.array([[1000, 0, 320],
                          [0, 1000, 240],
                          [0, 0, 1]])

rotation_matrix = np.eye(3)
translation_vector = np.array([0, 0, 0])
extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))

world_points = [
    [0, 0, 0], [274, 0, 0], [274, 152.5, 0], [0, 152.5, 0],
    [0, 76.25, 0], [274, 76.25, 0], [137, 152.5, 0], [137, 0, 0],
    [137, -15.25, 0], [137, -15.25, 15.25], [137, 167.75, 15.25], [137, 167.75, 0]
]

actual_image_points = np.array([
    [320, 240], [594, 240], [594, 392.5], [320, 392.5],
    [320, 316.25], [594, 316.25], [457, 392.5], [457, 240],
    [457, 224.75], [457, 240], [457, 407.75], [457, 392.5]
])

reprojected_points = reproject_points(camera_matrix, extrinsic_matrix, world_points)
errors = compute_errors(reprojected_points, actual_image_points)

print("Reprojected Points:\n", reprojected_points)
print("Actual Image Points:\n", actual_image_points)
print("Errors:\n", errors)