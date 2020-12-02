import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from array_utils.math import rescale


def plot_img(img):
    if isinstance(img, str):
        img = mpimg.imread(img)
    else:
        img_copy = img.copy()
        img_copy[np.isnan(img_copy)] = 0.
        img_copy = rescale(img_copy, 0, 255).astype(np.uint8)
        if img_copy.shape[0] == 3:
            img_copy = img_copy.swapaxes(0, 2).swapaxes(0, 1)
    imgplot = plt.imshow(img_copy)
    plt.show()
