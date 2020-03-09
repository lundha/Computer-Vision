import cv2
import numpy as np
import matplotlib.pyplot as plt


# Convert to greyscale

# img = Image.open('img.jpg').convert('L')
# img.save('img_gray.jpg')
img_gray = cv2.imread('img_gray.jpg')
 
# Crop image - Random 512x512 area in image
img_512 = img_gray[2500:3012,2500:3012]
cv2.imwrite('img512.jpg', img_512)


# Create Gaussian pyramid


def scaling(scale, img):

    width = int(img.shape[1] * scale/100)
    height = int(img.shape[0] * scale/100)
    dim = (width, height)
    return dim


img512 = cv2.GaussianBlur(img_512, (5,5), 0)


scale = 50

dim = scaling(scale, img512)

reso256 = cv2.resize(img512, dim, interpolation = cv2.INTER_AREA)
img256 = cv2.GaussianBlur(reso256, (5,5), 0)

dim = scaling(scale, img256)

reso128 = cv2.resize(img256, dim, interpolation = cv2.INTER_AREA)
img128 = cv2.GaussianBlur(reso128, (5,5), 0)

dim = scaling(scale, img128)

reso64 = cv2.resize(img128, dim, interpolation = cv2.INTER_AREA)
img64 = cv2.GaussianBlur(reso64, (5,5), 0)

# Print Gaussian pyramid

fix, (ax, ax2, ax3, ax4) = plt.subplots(1,4)
ax.set_title("512x512")
ax.imshow(img_512)
ax2.set_title("256x256")
ax2.imshow(reso256)
ax3.set_title("128x128")
ax3.imshow(reso128)
ax4.set_title("64x64")
ax4.imshow(reso64)

plt.show()


# Create Laplacian pyramid

scale = 200

dim = scaling(scale, img64)


reso128_lp = cv2.resize(img64, dim)
img128_smooth = cv2.GaussianBlur(reso128_lp, (5,5), 0)

dim = scaling(scale, img128)

reso256_lp = cv2.resize(img128, dim)
img256_smooth = cv2.GaussianBlur(reso256_lp, (5,5), 0)

dim = scaling(scale, img256)

reso512_lp = cv2.resize(img256, dim)
img512_smooth = cv2.GaussianBlur(reso512_lp, (5,5), 0)

laplace512 = np.subtract(img512,img512_smooth)
laplace256 = np.subtract(img256,img256_smooth)
laplace128 = np.subtract(img128,img128_smooth)
laplace64 = img64 # laplace 64x64 is the same as gaussian 64x64


# Print Laplacian pyramid

fix, (ax, ax2, ax3, ax4) = plt.subplots(1,4)

ax.set_title("512x512")
ax.imshow(laplace512)
ax2.set_title("256x256")
ax2.imshow(laplace256)
ax3.set_title("128x128")
ax3.imshow(laplace128)
ax4.set_title("64x64")
ax4.imshow(laplace64)

plt.show()
cv2.destroyAllWindows()


