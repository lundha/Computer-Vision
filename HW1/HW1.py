

# HW 1


# importing pyplot and image from matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2


image_src = cv2.imread('w_right.jpg')
image_dst = cv2.imread('w_left.jpg')
def get_image_coordinates(first_image, second_image):


    plt.figure()
    plt.imshow(first_image)
    coordinates_left = plt.ginput(4, show_clicks=True)
    coordinates_left = (np.around(coordinates_left,2))

    print("Coordinates left image:")
    print(coordinates_left)
    print("\n")

    plt.figure()
    plt.imshow(second_image)
    coordinates_right = plt.ginput(4, show_clicks=True)
    coordinates_right = (np.around(coordinates_right,2))


    print("Coordinates right image:")
    print(coordinates_right)
    print("\n")

    return np.array(coordinates_left, np.float32), np.array(coordinates_right, np.float32)

def get_homography_matrix(coordinate_src, coordinate_dst):

    row_a, col_a = 8, 8
    row_b, col_b = 8, 1


    A_conc = [[0 for x in range(col_a)] for y in range(row_a)]
    B_conc = [[0 for x in range(col_b)] for y in range(row_b)]

    for i in range(0, 4):
        x = int(coordinate_src[i][0])
        y = int(coordinate_src[i][1])

        x2 = int(coordinate_dst[i][0])
        y2 = int(coordinate_dst[i][1])


        A = np.matrix([[x, y, 1, 0, 0, 0, -x*x2, -y*x2], [0, 0, 0, x, y, 1, -x*y2, -y*y2]])
        B = np.matrix([[x2], [y2]])

        if i == 0:
            A_conc = A
            B_conc = B
        else:
            A_conc = np.concatenate((A_conc,A))
            B_conc = np.concatenate((B_conc,B))


    h, residuals, rank, s = np.linalg.lstsq(A_conc,B_conc, rcond=None)

    return h

def apply_homography_matrix(image, homography_matrix, filename):

    size = image.shape
    im_new = cv2.warpPerspective(image, homography_matrix, (size[1], size[0]))
    cv2.imwrite(filename, im_new)

def plot_points_on_image(src_coordinates, dst_coordinates, first_image, second_image, src_filename, dst_filename):

    red = (0, 0, 255)
    thickness = 2

    for i in range(0,4):
        x_src = int(src_coordinates[i][0])
        y_src = int(src_coordinates[i][1])

        x_dst = int(dst_coordinates[i][0])
        y_dst = int(dst_coordinates[i][1])

        cv2.circle(first_image, (x_src,y_src), 30, red, thickness)
        cv2.circle(second_image, (x_dst,y_dst), 30, red, thickness)


    cv2.imwrite(src_filename, first_image)
    cv2.imwrite(dst_filename, second_image)

src, dst = get_image_coordinates(image_src, image_dst)

homography_matrix = get_homography_matrix(src, dst)

with open('array_b_r_l.txt', 'w') as f:
    for item in homography_matrix:
        f.write("%s\n" % item)


with open('array_b_r_l.txt', 'w') as f:
    for item in homography_matrix:
        f.write("%s\n" % item)

homography_matrix = np.vstack((homography_matrix,1)).reshape((3,3))


print(homography_matrix)

apply_homography_matrix(image_src, homography_matrix, 'lol.jpg')

plot_points_on_image(src, dst, image_src, image_dst, 'lol2.jpg', 'lol3.jpg')


