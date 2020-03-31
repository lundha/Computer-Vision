## Random dot stereo pair

import numpy as np
import random as rand
import matplotlib.pyplot as plt
from copy import copy, deepcopy


# make white blank image
blank_image = 255 * np.ones(shape=[512,512, 3], dtype=np.uint8)


def fill_image_w_black_dots(image, num_dots):

    height, width, color = image.shape
    black = (0,0,0)
    i = 0
    while (i < num_dots):
        pixel_height = rand.randint(0,height-1)
        pixel_width = rand.randint(0,width-1)
        image[pixel_height][pixel_width] = black
        i = i + 1
    return image

new_image = fill_image_w_black_dots(blank_image, 100000)
new_image2 = deepcopy(new_image)



def generate_image_stereopair(image):

    green = (0, 255,0)
    #image[200:400, 250:450] = green

    image[200:400, 300:500] = image[200:400, 250:450]

    return image


displaced_image = generate_image_stereopair(new_image)

fix, (ax, ax2) = plt.subplots(1,2)
ax.imshow(new_image2)
ax2.imshow(displaced_image)

plt.show()


