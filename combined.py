import SolvePnp
import checkerror1
import cameramatrix
import testing1
import loadimg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def matrix_multiply(A, B, C):
    result = np.matmul(np.matmul(A, B), C)
    return result

def re_testing(intrensic_matrix , extrensic_matrix , world_point):
    world_point_4x1 = [np.array([x, y, z, 1]).reshape(4, 1) for x, y, z in world_point]
    result = []
    for i in world_point_4x1:
        result_imagePoint = matrix_multiply(intrensic_matrix, extrensic_matrix, i)
        result.append(result_imagePoint/result_imagePoint[2])
    
    return result

def plot_error_vector_field(image_points, image_points_reprojected):
    plt.figure(figsize=(10, 10))
    
    # Plot actual image points
    plt.scatter(image_points[:, 0], image_points[:, 1], color='red', label='Actual Image Points')
    
    # Plot reprojected image points
    plt.scatter(image_points_reprojected[:, 0], image_points_reprojected[:, 1], color='blue', label='Reprojected Image Points')
    
    # Draw error vectors
    for i in range(len(image_points)):
        plt.arrow(image_points[i, 0], image_points[i, 1],
                  image_points_reprojected[i, 0] - image_points[i, 0],
                  image_points_reprojected[i, 1] - image_points[i, 1],
                  head_width=5, head_length=5, fc='gray', ec='gray')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Error Vector Field: Actual vs Reprojected Image Points')
    plt.legend()
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.grid(True)
    plt.show()
    
def distance_between_points(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    squared_diff = (point1 - point2) ** 2
    sum_squared_diff = np.sum(squared_diff)
    distance = np.sqrt(sum_squared_diff)
    return distance


def cameraCalibration():
    
    # world_points=np.array([
    #                 [0,0,0],
    #                 [274,0,0],
    #                 [274,152.5,0],
    #                 [0,152.5,0],
    #                 [0,76.25,0],
    #                 [274,76.25,0],
    #                 [137,152.5,0], 
    #                 [137,0,0],
    #                 [137,-15.25,0],
    #                 [137,-15.25,15.25],
    #                 [137,167.75,15.25],
    #                 [137,167.75,0],
    #                 [137,76.25,0],
    #                 [137,76.25,15.25],
    #                 [0,114.375,0],
    #                 [0,38.125,0],
    #                 [274,38.125,0],
    #                 [274,114.375,0],  #18
    #                 # [38.6,23.1,-76],  #19
    #                 # [38.6,129.40,-76],  #20
    #             ], np.float32)
    
    world_points=np.array([[0,0,0],[274,0,0],[274,152.5,0],[0,152.5,0],
                    [0,76.25,0],[274,76.25,0],[137,152.5,0],[137,0,0],
                    [137,-15.25,0],[137,-15.25,15.25],[137,167.75,15.25],[137,167.75,0]], np.float32)
    
    #****************************************new points***********************************************************************  
    image_points = np.array([[1044,606],[1296,484],[1042,451],[744,554],
                            [885,580],[1164,467],[908,497],[1187,536],
                            [1218,541],[1218,508],[885,465],[885,494]], np.float32) #view4 overall error ---> 4.04 view4 [34.2 , 45.3 , 784.5] distance - 1565.96
    
    # image_points = np.array([[1038,526],[1272,410],[979,385],[677,484],
    #                         [849,504],[1120,397],[846,428],[1174,459],
    #                         [1208,461],[1208,427],[818,394],[818,428]], np.float32) #view5 overall error ---> 3.918 view5 [27.9 , -10.9 , 678.5] distance - 1641.65
    
    # image_points = np.array([[1026,534],[1341,425],[1034,407],[659,498],
    #                         [846,516],[1185,415],[869,445],[1205,471],
    #                         [1243,472],[1243,435],[839,407],[839,444]], np.float32) #view6 overall error ---> 5.37 view6 [22.2 , -4.9 , 616.9] distance - 1522.41
    
    # image_points = np.array([[1116,651],[1572,527],[990,508],[404,617],
    #                         [747,632],[1267,517],[741,554],[1373,582],
    #                         [1445,583],[1445,514],[680,487],[680,553]], np.float32) #view7 overall error ---> 23.92 view7 [26.4 , 30.9 , 285.7] distance - 1112.95
    
    # image_points = np.array([[1004,517],[1255,400],[866,388],[532,492],
    #                         [763,504],[1048,392],[716,433],[1146,448],
    #                         [1189,448],[1189,404],[678,386],[678,433]], np.float32) #view8 overall error ---> 8.77 view8 [14.5 , -1.16 , 507.8] distance 1519.71
    
    # image_points = np.array([[1100,570],[1429,445],[1108,415],[742,524],
    #                         [912,546],[1263,429],[940,466],[1278,500],
    #                         [1314,503],[1314,467],[910,428],[910,463]], np.float32) #view9 overall error ----> 5.68 view9 [44.7 , 17.2 , 618.7] distance 1490.38
    
    #**************************************************************************************************************  
    
    
 #****************************************view 3***********************************************************************  #error result -- point-by-point - 7.60 overall 5.99   
   
    # image_points = np.array([[1149,629],[1125,430],[796,427],[772,627],
    #                         [961,628],[960,429],[783,519],[1136,522],
    #                         [1171,522],[1171,490],[749,488],[749,519]], np.float32) #view2 according to research paper
        
 #***************************************************************************************************************    
    
 #****************************************view 2***********************************************************************     #error result -- point-by-point - 5.10 overall- 4.64
    
    # image_points = np.array([[1023,604],[1315,457],[1033,433],[692,563],
    #                         [848,583],[1168,443],[881,488],[1188,520],
    #                         [1225,521],[1224,487],[855,453],[858,485]], np.float32) #view2 according to research paper
        
 #***************************************************************************************************************    
    
 #****************************************view 1***********************************************************************    #error result -- point-by-point - 8.13 overall -6.06
    
    # world_points=np.array([[0,0,0],[274,0,0],[274,152.5,0],[0,152.5,0],
    #                 [0,76.25,0],[274,76.25,0],[137,-15.25,0],
    #                 [137,-15.25,15.25],[137,167.75,15.25]], np.float32)
    
    # image_points = np.array([[613,659],[1292,659],[1244,517],[676,518],
    #                         [647,583],[1266,582],[949,658],[949,644],
    #                         [959,477]], np.float32) #view1 according to research paper
    
 #******************************************************************************************************************************************    
    
    # image_points = np.array([
    #                 [1250, 719],      # Corresponding to Point 1
    #                 [1420, 295],      # Corresponding to Point 2
    #                 [897, 284],       # Corresponding to Point 3
    #                 [431, 660],       # Corresponding to Point 4
    #                 [812, 688],       # Corresponding to Point 5
    #                 [1153, 288],      # Corresponding to Point 6
    #                 [714, 426],       # Corresponding to Point 7
    #                 [1364, 450],      # Corresponding to Point 8
    #                 [1437, 453],      # Corresponding to Point 9
    #                 [1426, 387],      # Corresponding to Point 10
    #                 [659, 361],       # Corresponding to Point 11
    #                 [651, 419],       # Corresponding to Point 12
    #                 [1021, 445],       # Corresponding to Point 13
    #                 [1024, 381],       # Corresponding to Point 14
    #                 [612, 673],       # Corresponding to Point 15
    #                 [1020, 704],       # Corresponding to Point 16
    #                 [1283, 292],       # Corresponding to Point 17
    #                 [1023, 286],       # Corresponding to Point 18
    #                 # [1177, 973],       # Corresponding to Point 18
    #                 # [674, 910],       # Corresponding to Point 18
    #             ], dtype=np.float32)  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> custom view <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    images = loadimg.loadimage()
    
    #find the cameraMatrix and distCoeffs of the camera (Internal parameters which depends upon the camera)
    # cameraMatrix , distCoeffs, CM_error = cameramatrix.create_CM(images)
    
    cameraMatrix = np.array([[1920, 0, 960],
                             [0, 1080, 540],
                             [0, 0, 1]], dtype=np.float32)

#    Assuming no lens distortion for simplicity
    distCoeffs = np.zeros(4)
    
    rotation_matrix , rvec , tvecs = SolvePnp.implement_solvepnp(world_points , image_points , cameraMatrix , distCoeffs)
    
    extrensic_matrix = np.hstack((rotation_matrix, tvecs)) 
    
    mean_projected_error , error , image_points_reprojected = checkerror1.checkerror_result(extrensic_matrix , world_points , image_points , cameraMatrix , distCoeffs)
    
    verification = re_testing(cameraMatrix , extrensic_matrix ,world_points)
    
    verification = np.array(verification)
    
    new_wp = np.array(tvecs)
    new_wp_3d = np.append(new_wp, 1)

    result = matrix_multiply(cameraMatrix, extrensic_matrix, new_wp_3d)
    
    result = result/[result[2]]
    
    img_point_first = image_points[0]
    
    img_point_first = np.array(img_point_first)
    
    img_point_first_3d = np.append(img_point_first, 1)
    
    distance = distance_between_points(img_point_first_3d , result)
    
    actual_image_point = [np.array([x, y, 1]).reshape(3, 1) for x, y in image_points]
    actual_image_point = np.array(actual_image_point)

    print("***********************************************************************************")
    print("Intrensic matrix --> " , cameraMatrix)
    # print("Reprojection Error in cameraMatrix --> " ,CM_error)
    print("Extrensic matrix --> " , extrensic_matrix)
    print("POINT BY POINT Mean projected error --> ", mean_projected_error)
    print("OVERALL POSE ESTIMATION error --> ", error)
    print("distance from camera to origin point in world coordinate system --> ", distance)
    print("resultant image pointss --> " , verification)
    
    plot_error_vector_field(image_points, image_points_reprojected)
    
cameraCalibration()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


