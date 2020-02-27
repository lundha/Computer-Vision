# Harris corner detection


import cv2
import numpy as np
from scipy import signal as sig
import matplotlib.image as img
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from skimage.feature import corner_harris, corner_peaks


# Convert images to greyscale
left = imread('right.jpg')
left_gray = rgb2gray(left)
nrow, ncol  = left_gray.shape

# Spatial derivative calculation using the sobel filter

def grad_x(img_gray):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    I_x = sig.convolve2d(img_gray, kernel_x, mode='same')
    return I_x

def grad_y(img_gray):
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    I_y = sig.convolve2d(img_gray, kernel_y, mode='same')
    return I_y


I_x = grad_x(left_gray)
I_y = grad_y(left_gray)


#I_x, I_y = np.gradient(left_gray)

Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
Iyy = ndi.gaussian_filter(I_y**2, sigma=1)

k = 0.05

# determinant
detA = Ixx * Iyy - Ixy ** 2
# trace
traceA = Ixx + Iyy

harris_response = detA - k * traceA ** 2

# COPY PASTE FROM HERE

img_copy_for_corners = np.copy(left)

for rowindex, response in enumerate(harris_response):
    for colindex, r in enumerate(response):
        if r > 0.00001 and colindex < ncol and rowindex < nrow:
            # this is a corner
            img_copy_for_corners[rowindex, colindex] = [0, 0, 255]

# If r > 0.5 function works good.

#cv2.imshow('img', img_copy_for_corners)
#cv2.waitKey()



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
ax[0].set_title("corners found")
ax[0].imshow(img_copy_for_corners)
ax[1].set_title("corners found")
ax[1].imshow(img_copy_for_corners)
plt.show()


