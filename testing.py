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

# # Example usage:
# # Define three example matrices A, B, and C
# intrensic_matrix = np.array([[ 1.7300397e+03,  0.0000000e+00,  9.1560248e+02 , 0.00],[ 0.0000000e+00,  2.9063455e+03, -9.5064484e+02 , 0.00],[ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00 , 0.00]])

# extrensic_matrix = np.array([[ 0.33353547, -0.94249711, -0.02129048, 54.97206665],
#                               [ 0.32403689,  0.13582098, -0.93624396 , 164.33393276],
#                               [ 0.88529892, 0.30537167,  0.35070495 , 264.90900417],
#                               [ 0.00000000, 0.00000000,  0.00000000 , 1.000000000]])

# world_point = np.array([0, 0, 0 ,1])

# # Call the matrix_multiply function
# result = matrix_multiply(intrensic_matrix, extrensic_matrix, world_point)

# # Print the result
# print("Result of matrix multiplication:")
# print(result)

import numpy as np

def matrix_multiply(A, B, C):
    result = np.matmul(np.matmul(A, B), C)
    return result

def testing(intrensic_matrix , extrensic_matrix , world_point):
    result = matrix_multiply(intrensic_matrix, extrensic_matrix, world_point)
    
    return result
