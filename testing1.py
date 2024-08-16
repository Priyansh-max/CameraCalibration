# Rotation matrix (R):
#  [[ 0.33353547, -0.94249711, -0.02129048,]
#  [ 0.32403689,  0.13582098, -0.93624396]
#  [ 0.88529892, 0.30537167,  0.35070495]]

# Translation vector (tvec):
#  [ 54.97206665 164.33393276 264.90900417]

# intrensic propertyy
# Final Camera Matrix:
#  [[ 1.7300397e+03  0.0000000e+00  9.1560248e+02]
#  [ 0.0000000e+00  2.9063455e+03 -9.5064484e+02]
#  [ 0.0000000e+00  0.0000000e+00  1.0000000e+00]]

#extrensic property
#  [[ 0.33353547, -0.94249711, -0.02129048, 54.97206665]
#  [ 0.32403689,  0.13582098, -0.93624396 , 164.33393276]
#  [ 0.88529892, 0.30537167,  0.35070495 , 264.90900417]
#  [ 0.00000000, 0.00000000,  0.00000000 , 1.000000000]]

    # [0, 0, 0],       # Point 1                                
    # [274, 0, 0],     # Point 2
    # [274, 152.5, 0], # Point 3
    # [0, 152.5, 0],   # Point 4
    # [0, 76.25, 0],   # Point 5
    # [274, 76.25, 0], # Point 6
    # [137, 152.5, 0], # Point 7
    # [137, 0, 0],     # Point 8
    # [137, -16.3, 0], # Point 9
    # [137, -16.3, 16],# Point 10
    # [137, 169.6, 16.2], # Point 11
    # [137, 169.6, 0],    # Point 12
    
    # [1252, 720],      # Corresponding to Point 1
    # [1423, 297],      # Corresponding to Point 2
    # [896, 286],       # Corresponding to Point 3
    # [430, 658],       # Corresponding to Point 4
    # [814, 688],       # Corresponding to Point 5
    # [1153, 288],      # Corresponding to Point 6
    # [714, 428],       # Corresponding to Point 7
    # [1343, 452],      # Corresponding to Point 8
    # [1432, 453],      # Corresponding to Point 9
    # [1425, 388],      # Corresponding to Point 10
    # [659, 362],       # Corresponding to Point 11
    # [657, 420],       # Corresponding to Point 12
    
# import numpy as np

# def matrix_multiply(A, B, C):
#     result = np.matmul(np.matmul(A, B), C)
#     return result

# # # Example usage:
# # # Define three example matrices A, B, and C
# intrinsic_matrix = np.array([[1920, 0, 960], 
#                           [0, 1080 , 540], 
#                           [0, 0, 1]], dtype=np.float32)


# intrinsic_matrix = np.array([[2.36769566e+03, 0.00000000e+00, 8.42110659e+02],
#  [0.00000000e+00, 2.39270608e+03, 2.58483659e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

# # [ 3.33105827e-01 -9.40854641e-01 -6.19116544e-02  6.71751989e+01]
# #  [-2.43770210e-01 -2.25045159e-02 -9.69571881e-01  7.71658153e+01]
# #  [ 9.10832912e-01  3.38062260e-01 -2.36848718e-01  3.80145618e+02]

# # [[ 3.33105827e-01 -9.40854641e-01 -6.19116544e-02  6.71751989e+01]
# #  [-2.43770210e-01 -2.25045159e-02 -9.69571881e-01  7.71658153e+01]
# #  [ 9.10832912e-01  3.38062260e-01 -2.36848718e-01  3.80145618e+02]]

# # extrinsic_matrix = np.array([[2.87824297e-01, -9.50476132e-01, -1.17270186e-01,  4.67806300e+01],
# #  [-6.22054433e-01, -9.24432381e-02, -7.77497608e-01 , 5.04041399e+01],
# #  [ 7.28152083e-01,  2.96731141e-01, -6.17855301e-01 , 3.11819743e+02]], np.float32)

# extrinsic_matrix = np.array([[ 3.33105827e-01, -9.40854641e-01, -6.19116544e-02,  6.71751989e+01],
#  [-2.43770210e-01, -2.25045159e-02, -9.69571881e-01,  7.71658153e+01],
#  [ 9.10832912e-01,  3.38062260e-01, -2.36848718e-01,  3.80145618e+02]], np.float32)

# [[ 5.55889488e-01, -8.30341021e-01, -3.89957302e-02,  4.47230738e+01],
#  [-3.41398724e-01, -1.85280904e-01, -9.21475934e-01,  1.72222366e+01],
#  [ 7.57914104e-01,  5.25551878e-01, -3.86473072e-01,  6.18727467e+02]]

# extrinsic_matrix = np.array([[ 5.55889488e-01, -8.30341021e-01, -3.89957302e-02,  4.47230738e+01],
#  [-3.41398724e-01, -1.85280904e-01, -9.21475934e-01,  1.72222366e+01],
#  [ 7.57914104e-01,  5.25551878e-01, -3.86473072e-01,  6.18727467e+02]], np.float32)

# world_point = np.array([44.7 , 17.2 , 618.7,1])

# # Call the matrix_multiply function
# result = matrix_multiply(intrinsic_matrix, extrinsic_matrix, world_point)

# # Print the result
# print("Result of matrix multiplication:")
# print(result/[result[2]])

import numpy as np

def matrix_multiply(A, B, C):
    result = np.matmul(np.matmul(A, B), C)
    return result

def re_testing(intrensic_matrix , extrensic_matrix , world_point):
    print(intrensic_matrix.shape , "hi1")
    print(extrensic_matrix.shape, "hi2")
    print(world_point.shape, "hi3")
    world_point_4x1 = [np.array([x, y, z, 1]).reshape(4, 1) for x, y, z in world_point]
    result = []
    for i in world_point_4x1:
        print(intrensic_matrix.shape)
        print(extrensic_matrix.shape)
        print(world_point_4x1.shape)
        result_imagePoint = matrix_multiply(intrensic_matrix, extrensic_matrix, i)
        result.append(result_imagePoint)
    
    return result_imagePoint

