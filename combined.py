import SolvePnp
import checkerror
import cameramatrix
import testing
import loadimg
import numpy as np

import numpy as np

def matrix_multiply(A, B, C):
    result = np.matmul(np.matmul(A, B), C)
    return result

def testing(intrensic_matrix , extrensic_matrix , world_point):
    result = matrix_multiply(intrensic_matrix, extrensic_matrix, world_point)
    
    return result

def cameraCalibration():
    
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
    
    images = loadimg.loadimage()
    
    #find the cameraMatrix and distCoeffs of the camera (Internal parameters which depends upon the camera)
    cameraMatrix , distCoeffs = cameramatrix.create_CM(images)
    
    rotation_matrix , rvec , tvecs = SolvePnp.implement_solvepnp(world_points , image_points , cameraMatrix , distCoeffs)
    
    extrensic_matrix = np.hstack((rotation_matrix, tvecs)) 
    
    mean_projected_error , error = checkerror.checkerror_result(extrensic_matrix , world_points , image_points , cameraMatrix)
    
    # verification = testing(cameraMatrix , extrensic_matrix ,world_point[0])
    
    print("***********************************************************************************")
    print("Intrensic matrix --> " , cameraMatrix)
    print("Extrensic matrix --> " , extrensic_matrix)
    print("Mean projected error --> ", mean_projected_error)
    print("error --> ", error)
    # print(world_points[0] , "corresponding image point" , image_points[0] , "verification result" , verification)
    
cameraCalibration()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


