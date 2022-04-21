import sys
import cv2
import numpy as np
import time
from PIL import Image, ImageOps


def find_depth(disparity, frame_right, frame_left, baseline, alpha):
    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    # TODO
    # print("left image size: ", frame_left.shape)

    # print("right image size: ", frame_right.shape)

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)
        print("focal_length: ", f_pixel)


    else:
        print('Left and right camera frames do not have the same pixel width')

    # print("disparity: ", str(disparity))

    # CALCULATE DEPTH z:
    zDepth = (baseline * f_pixel) / disparity  # Depth in [cm]

    return zDepth





##########################################
'''
left_x_y = det_l[i]
left_x = left_x_y[0]
right_x_y = det_r[i]
right_x = right_x_y[0]
# Left y-coordinate of midpoint keypoints (origin: bottom left corner)
left_y = 720 - left_x_y[1]  # TODO 642 - left_x_y[1]    (720 when zero offset)

'''
'''
####################################################################################
# disparity, baseline and focal length
disparity = left_x - right_x  #right_x - left_x
baseline = 122  # mm
focal_length = 1.8  # mm
pixel_size = 0.00112  # 0.00112  # mm/pixel

# x, y coordinates of point
x_coord = (baseline * left_x) / (disparity)
y_coord = (baseline * left_y) / (disparity)

# z-coordinate Depth = f*b/disparity
depth = (focal_length * baseline) / (disparity * pixel_size)
####################################################################################
'''





