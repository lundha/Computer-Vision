# Harris corner detection


import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from scipy import ndimage as ndi


# Convert images to greyscale
left = imread('left.jpg')
right = imread('right.jpg')

left_gray = rgb2gray(left)
right_gray = rgb2gray(right)


def corner_detection(img_gray, original):

    nrow, ncol = img_gray.shape

    def grad_x(img_gray):
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        I_x = sig.convolve2d(img_gray, kernel_x, mode='same')
        return I_x

    def grad_y(img_gray):
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        I_y = sig.convolve2d(img_gray, kernel_y, mode='same')
        return I_y


    I_x = grad_x(img_gray)
    I_y = grad_y(img_gray)


    Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
    Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
    Iyy = ndi.gaussian_filter(I_y**2, sigma=1)

    k = 0.05

    # determinant
    detA = Ixx * Iyy - Ixy ** 2

    # trace
    traceA = Ixx + Iyy

    harris_response = detA - k * traceA ** 2


    img_copy = np.copy(original)

    for rowindex, response in enumerate(harris_response):
        for colindex, r in enumerate(response):
            if r > 0 and colindex < ncol and rowindex < nrow:
                # this is a corner
                img_copy[rowindex, colindex] = [0, 0, 255]

    return img_copy


corners_left = corner_detection(left_gray, left)
corners_right = corner_detection(right_gray, right)


fig, (ax, ax2) = plt.subplots(1,2)
ax.set_title("corners found")
ax.imshow(corners_left)
ax2.set_title("corners found")
ax2.imshow(corners_right)
plt.show()


