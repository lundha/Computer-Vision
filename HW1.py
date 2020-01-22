

# HW 1


# importing pyplot and image from matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2


def get_image_coordinates(first_image, second_image):

    im_left = cv2.imread(first_image)
    im_right = cv2.imread(second_image)

    plt.figure()
    plt.imshow(im_left)
    coordinates_left = plt.ginput(4, show_clicks=True)
    coordinates_left = (np.around(coordinates_left,2))

    print("Coordinates left image:")
    print(coordinates_left)
    print("\n")

    plt.figure()
    plt.imshow(im_right)
    coordinates_right = plt.ginput(4, show_clicks=True)
    coordinates_right = (np.around(coordinates_right,2))

    print("Coordinates right image:")
    print(coordinates_right)
    print("\n")

    return np.array(coordinates_left), np.array(coordinates_right)



def get_homography_matrix(coordinates_left, coordinates_right):

    row_a, col_a = 8, 8
    row_b, col_b = 8, 1

    A = [[0 for x in range(col_a)] for y in range(row_a)]
    B = [[0 for x in range(col_b)] for y in range(row_b)]

    A_conc = [[0 for x in range(col_a)] for y in range(row_a)]
    B_conc = [[0 for x in range(col_b)] for y in range(row_b)]

    for i in range(0, 4):
        x = int(coordinates_left[i][0])
        y = int(coordinates_left[i][1])

        x2 = int(coordinates_right[i][0])
        y2 = int(coordinates_right[i][1])


        A = np.matrix([[x, y, 1, 0, 0, 0, -x*x2, -y*x2], [0, 0, 0, x, y, 1, -x*y2, -y*y2]])
        B = np.matrix([[x2], [y2]])

        if i == 0:
            A_conc = A
            B_conc = B
        else:
            A_conc = np.concatenate((A_conc,A))
            B_conc = np.concatenate((B_conc,B))

    #print(A_conc)
    #print(B_conc)

    h, residuals, rank, s = np.linalg.lstsq(A_conc,B_conc, rcond=None)

    return h

def apply_homography_matrix(image, homography_matrix):

    size = image.shape
    im_new = cv2.warpPerspective(image, homography_matrix, (size[0], size[1]))
    plt.figure()
    plt.imshow(im_new)
    plt.imshow(image)


def perspectiveTransform(coordinates_left, coordinates_right, image):

    size = image.shape
    M = cv2.getPerspectiveTransform(coordinates_left, coordinates_right)
    im_new = cv2.warpPerspective(image, M, (size[0], size[1]))
    cv2.imshow("im_new", im_new)
    k = cv2.waitKey(10000)

    print("hh")


coordinates_left, coordinates_right = get_image_coordinates('left.jpg', 'right.jpg')

src = np.array(coordinates_left, np.float32)
dst = np.array(coordinates_right, np.float32)


homography_matrix = get_homography_matrix(coordinates_left, coordinates_right)

image = cv2.imread('left.jpg')

perspectiveTransform(src, dst, image)

#apply_homography_matrix(image, homography_matrix)

