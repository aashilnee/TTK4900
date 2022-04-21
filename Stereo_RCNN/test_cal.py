import calibration

import cv2
import torch
from torch.autograd import Variable
from lib.model.utils.config import cfg
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.roi_layers import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv, kpts_transform_inv, border_transform_inv
from lib.model.stereo_rcnn.resnet import resnet

import matplotlib.pyplot as plt

import _init_paths
import os
import numpy as np
import argparse
import time


# TODO test images
img_l_path = 'demo/29_left.bmp'  #'demo/left_no_offset.png'
img_r_path = 'demo/29_right.bmp'  #'demo/right_no_offset.png'

img_left = cv2.imread(img_l_path)
img_right = cv2.imread(img_r_path)

#img_left = cv2.resize(img_left, (3264, 1848)) # 3264x1848
#img_right = cv2.resize(img_right, (3264, 1848))

################## CALIBRATION #########################################################

img_right, img_left = calibration.undistortRectify(img_right, img_left)

########################################################################################

########## Offset #################################

import numpy as np
from scipy.signal import fftconvolve

'''
def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out


# gray-scale image used in cross-correlation
right_img = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
left_img = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

# cross-correlation
corr = normxcorr2(right_img, left_img)
# cross corr between equal image
corr_1 = normxcorr2(right_img, right_img)

# find match
y, x = np.unravel_index(np.argmax(corr), corr.shape)
# peak left image
y_1, x_1 = np.unravel_index(np.argmax(corr_1), corr_1.shape)
# offset
y_offset_value = y_1 - y

print("\nOffset value: ", y_offset_value)


#####################################################################################
'''

height, width, channels = img_left.shape

print(height,width)


### RESIZE ###########
#img_left = cv2.resize(img_left, (640, 360))
#img_right = cv2.resize(img_right, (640, 360))
#######################

'''
#### LINJE #####
start_point_1 = (170, 0)
end_point_1 = (170, 780)

start_point_2 = (400, 0)
end_point_2 = (400, 780)

start_point_3 = (600, 0)
end_point_3 = (600, 780)

start_point_4 = (800, 0)
end_point_4 = (800, 780)

# Green color in BGR
color = (0, 0, 255)

# Line thickness of 9 px
thickness = 2


img_left = cv2.resize(img_left, (544, 308)) # 3264x1848
img_right = cv2.resize(img_right, (544, 308)) # 3264x1848

# Using cv2.line() method
# Draw a diagonal green line with thickness of 9 px
image_L = cv2.line(img_left, start_point_1, end_point_1, color, thickness)
image_R = cv2.line(img_right, start_point_1, end_point_1, color, thickness)

image_L = cv2.line(img_left, start_point_2, end_point_2, color, thickness)
image_R = cv2.line(img_right, start_point_2, end_point_2, color, thickness)
##########################
'''



############ vis images ################################
img = np.hstack((img_left, img_right))




# Save image
path = 'results/'
cv2.imwrite(os.path.join(path, '3.jpg'), img)

cv2.imshow("img", img)
cv2.waitKey()
###########################################################




'''
############ vis images plot ###################################

# convert color image into grayscale image
img1 = cv2.cvtColor(img_left, cv2.COLOR_RGB2GRAY)
img2 = cv2.cvtColor(img_right, cv2.COLOR_RGB2GRAY)

# Draw the rectified images
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(img1,  cmap='gray')
axes[1].imshow(img2,  cmap='gray')
axes[0].axhline(220)
axes[1].axhline(220)

axes[0].axhline(350)
axes[1].axhline(350)

axes[0].axhline(439)
axes[1].axhline(439)


plt.suptitle("Rectified images")
plt.savefig("rectified_images.png")
plt.show()

cv2.waitKey(1)

########################################################################



'''