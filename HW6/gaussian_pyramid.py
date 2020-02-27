


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image


# Convert to greyscale
# img = Image.open('img.jpg').convert('L')
# img.save('img_gray.jpg')


img_gray = cv2.imread('img_gray.jpg')

# Crop image
img_512 = img_gray[2500:3012,2500:3012]
cv2.imwrite('img512.jpg', img_512)
# Create Gaussian pyramid

# pyrDown256 - row/2 col/2 gives 256x256
img_lower_reso256 = cv2.pyrDown(img_512)
cv2.imwrite('img256.jpg', img_lower_reso256)

# pyrDown128 - row/2 col/2 gives 128x128
img_lower_reso128 = cv2.pyrDown(img_lower_reso256)
cv2.imwrite('img128.jpg', img_lower_reso128)

# pyrDown64 - row/2 col/2 gives 64x64
img_lower_reso64 = cv2.pyrDown(img_lower_reso128)
cv2.imwrite('img64.jpg', img_lower_reso128)

# pyrDown32 - row/2 col/2 gives 32x32
img_lower_reso32 = cv2.pyrDown(img_lower_reso64)
cv2.imwrite('img32.jpg', img_lower_reso32)


#fix, (ax, ax2, ax3, ax4) = plt.subplots(1,4)
#ax.set_title("512x512")
#ax.imshow(img_512)
#ax2.set_title("256x256")
#ax2.imshow(img_lower_reso256)
#ax3.set_title("128x128")
#ax3.imshow(img_lower_reso128)
#ax4.set_title("32x32")
#ax4.imshow(img_lower_reso32)

#plt.show()

# pyrUp and pyrDown convolves the image with the Gaussian kernel and then upsample/downsample


'''
img512_smooth = cv2.pyrUp(img_lower_reso256)
img256_smooth = cv2.pyrUp(img_lower_reso128)
img128_smooth = cv2.pyrUp(img_lower_reso64)
img64_smooth = cv2.pyrUp(img_lower_reso32)

laplace512 = np.subtract(img_512,img512_smooth)
laplace256 = np.subtract(img_lower_reso256,img256_smooth)
laplace128 = np.subtract(img_lower_reso128,img128_smooth)
laplace32 = img_lower_reso32 # laplace 32x32 is the same as gaussian 32x32

fix, (ax, ax2, ax3, ax4) = plt.subplots(1,4)

ax.set_title("512x512")
ax.imshow(laplace512)
ax2.set_title("256x256")
ax2.imshow(laplace256)
ax3.set_title("128x128")
ax3.imshow(laplace128)
ax4.set_title("32x32")
ax4.imshow(img_lower_reso32)

plt.show()
cv2.destroyAllWindows()
'''

