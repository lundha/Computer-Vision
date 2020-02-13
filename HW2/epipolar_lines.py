

# HW 2


import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2

homogenous_left = []
homogenous_right = []

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

image_left = cv2.imread('left.jpg')
image_right = cv2.imread('right.jpg')

plt.figure()
plt.imshow(image_left)
coordinates_left = plt.ginput(8, show_clicks=True)

plt.figure()
plt.imshow(image_right)
coordinates_right = plt.ginput(8, show_clicks=True)

homogenous_left = np.array(coordinates_left, np.int32)
homogenous_right = np.array(coordinates_right, np.int32)

print(homogenous_left)
print(homogenous_right)

F, mat = cv2.findFundamentalMat(np.array(homogenous_left), np.array(homogenous_right))
print(F)


def drawEpipolarLine(cartesian_coordinate, fundamental_matrix):

    def make_homogenous_coord(cartesian_cor):
        homogenous = []
        for i in range(0, len(cartesian_cor) + 1):
            homogenous.append(cartesian_cor[0][i])
        homogenous.append(1)
        return homogenous

    homogenous_coordinate = make_homogenous_coord(cartesian_coordinate)
    homogenous_coordinate = np.array(homogenous_coordinate)
    line = fundamental_matrix.dot(homogenous_coordinate)
    print(line)
    a, b, c = line.ravel()
    x = np.array([0, 400])
    y = -(x*a + c) / b
    
    plt.figure()
    plt.plot(x,y)
    plt.show()
    return 0


plt.figure()
plt.imshow(image_left)
epipolar_coordinates = plt.ginput(1, show_clicks=True)

drawEpipolarLine(epipolar_coordinates, F)

def make_homogenous_coord(cartesian_cor):
    homogenous = []
    for i in range(0, len(cartesian_cor)+1):
        homogenous.append(cartesian_cor[0][i])
    homogenous.append(1)
    return homogenous

#homogenous_right = make_homogenous_coord(coordinates_right)
#homogenous_left = make_homogenous_coord(coordinates_left)

