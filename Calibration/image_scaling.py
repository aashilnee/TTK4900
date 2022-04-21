import os

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageOps

#################### TIDLIGERE BRUKT FOR RESIZING ################################################
'''
img_path = "/Users/aashi/Documents/chess_frames_padding_2/"


for file in os.listdir(img_path):
    f_img = img_path + "/" + file

    down_width = 1088
    down_height = 616
    down_points = (down_width, down_height)

    img = cv2.imread(f_img)

    resize = cv2.resize(img, down_points, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(f_img, resize)

'''

################## EKSEMPEL 1, padding ######################################################
'''
img_path = "/Users/aashi/Documents/chess_frames_padding_2/"

for file in os.listdir(img_path):
    f_img = img_path + "/" + file
    img = cv2.imread(f_img)


# read image
#img = cv2.imread('test_images/Gull_portrait_ca_usa.jpg')
    old_image_height, old_image_width, channels = img.shape

    # create new image of desired size and color (blue) for padding
    new_image_width = 1280
    new_image_height = 720
    color = (0,0,0)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height,
    x_center:x_center+old_image_width] = img

    # view result
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# save result
#cv2.imwrite("lena_centered.jpg", result)

    cv2.imwrite(f_img, result)

'''
################### EKSEMPEL 2, PADDING #################################################################
'''



def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    # thumbnail: preserves the aspect of the image
    img.thumbnail((expected_size[0], expected_size[1]))
    print("image size: ", img.size)
    print("expected size: ", expected_size)
    delta_width = expected_size[0] - img.size[0]
    print("delta width: ", delta_width)
    delta_height = expected_size[1] - img.size[1]
    print("delta height: ", delta_height)
    pad_width = delta_width // 2
    print("pad_width: ", pad_width)
    pad_height = delta_height // 2
    print("pad_height: ", pad_height)
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    print("padding: ", padding)
    return ImageOps.expand(img, padding)


if __name__ == "__main__":

    img_path = "/Users/aashi/Documents/chess_frames_with_padding/"

    for file in os.listdir(img_path):
        f_img = img_path + "/" + file
        img = Image.open(f_img)
        print("image size", img.size)
        img_padding = resize_with_padding(img, (1280, 720))
        print(img_padding.size)
        img_padding.show()
        img_padding.save(f_img)
'''
#############################################

################# EKSEMPEL 3, Padding ##########################################


img_path = "/Users/aashi/Documents/chess_frames_padding/"


#desired_size = 368

desired_width = 1280
desired_height = 720

#im_pth = "/home/jdhao/test.jpg"

for file in os.listdir(img_path):
    f_img = img_path + "/" + file

    im = cv2.imread(f_img)
    old_size = im.shape[:2] # old_size is in (height, width) format

    #ratio = float(desired_size)/max(old_size)
    #ratio_width = float(desired_width)/float(old_size[1])
    #ratio_height = float(desired_height)/float(old_size[0])

    print("Old width ", old_size[1])
    print("Old height ", old_size[0])

    #print("ratio height ", ratio_height)
    #print("ratio width", ratio_width)

    #print("ratio: ", ratio_width) # 0.1127
    #new_size = tuple([int(x*ratio) for x in old_size])
    #print("new size: ", new_size) # (208, 368)


    #new_width = int(old_size[1] * ratio_width)
    #new_height = int(old_size[0] * ratio_height)

    new_width = 1272
    new_height = 720

    print("new width ", new_width)
    print("new height ", new_height)


    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_CUBIC)  # RESIZE TIL DET SOM PASSER

    delta_w = desired_width - new_width
    delta_h = desired_height - new_height
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    print("delta_w ", delta_w)
    print("delta_h ", delta_h)
    print("top ", top)
    print("bottom ", bottom)
    print("left ", left)
    print("right ", right)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    cv2.imshow("image", new_im)
    cv2.imwrite(f_img, new_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()