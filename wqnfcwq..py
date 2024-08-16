import numpy as np

def distance_between_points(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    squared_diff = (point1 - point2) ** 2
    sum_squared_diff = np.sum(squared_diff)
    distance = np.sqrt(sum_squared_diff)
    return distance


# Points
point1 = [ 1.10156723e+03, -9.20383395e+02,  1.00000000e+00]
point2 = [1.1e+03, 5.7e+02, 1.0e+00]

# Calculate distance
distance = distance_between_points(point1, point2)
print(f"The distance between the points is: {distance}")