

# HW 1


# importing pyplot and image from matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

# Read two image files

im = img.imread('eagle.png')
#im2 = img.imread('eagle.png')

plt.figure()
plt.imshow(im)
coordinates = plt.ginput(4, show_clicks=True)
formatted_coordinates = (np.around(coordinates,2))

print(formatted_coordinates)

# Identify 4 pairs of corresponding points

# Write funciton homography