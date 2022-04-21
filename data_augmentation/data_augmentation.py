import imgaug.augmenters as iaa
import cv2
import glob
import os

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import csv
import json
import numpy as np

'''
# 1. Load dataset
images = []
images_path = glob.glob("images/*.png")
print(images_path)

for img_path in images_path:
    img = cv2.imread(img_path)
    images.append(img)  # put all images into images

# 2. Image augmentation
#put here all the images that we want to apply
augmentation = iaa.Sequential([

     # 1. Horizontal flip
    iaa.Fliplr(0.3), # mirrored images half of the time (0.5)
    
    #  Vertical flip, uaktuelt pga oppned fisk?
    #iaa.Flipud(0.5),



    # 3. Multiply (the cannels), make the image darker or brighter
    iaa.Multiply((0.8, 1.4)),  # standard: 0.8, 1.2

    # 4. Linear contrast, increasing the contrast
    iaa.LinearContrast((0.6, 1.4)),

    
    # Perform methods below only sometimes (half of the times)
    iaa.Sometimes(0.8,
                  # 5. GaussianBlur. in a dataset, the object that is in the foreground is in focus, while the rest
                  # is blurred. So if the dataset contains some blurring, it will improve the detection of the
                  # objects that are out of focus.
                  iaa.GaussianBlur((0, 3.0)),
                  iaa.Rotate((-15, 15)),

                  # 2. Affine (image moving to right/left hand side (20% of the image are gone) ), kan bare ta i x-retning,
                  # fjerne y, trenger ikke ha med rotate eller scale
                  #iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                #rotate=(-10, 10)),
                #scale=(0.5, 1.5)),
                  )

])


augmented_images = augmentation(images=images)


i = 1
# 3. Show images  # while true makes a loop where the images dont stop(?)
#while True:
while(i <= 3):
    augmented_images = augmentation(images=images)
    i = i+1
    for img in augmented_images:
        cv2.imshow("Image", img)
        cv2.waitKey(0)

        result_path = 'results'
        k = cv2.waitKey(0) & 0xFF
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
        elif k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite(os.path.join(result_path, 'result.png'), img)  #  img   TODO begge bilder
            cv2.destroyAllWindows()

'''
#####################################################################################################

########################################################################################################################
# The following example loads an image and places two bounding boxes on it. The image is then augmented to be          #
# brighter, slightly rotated and scaled. These augmentations are also applied to the bounding boxes. The image is then #
# shown before and after augmentation (with bounding boxes drawn on it).                                               #
#                                                                                                                      #
# KODE HENTET FRA: https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html                         #
########################################################################################################################
'''

ia.seed(1)
image_path = 'images/1.png'

image = cv2.imread(image_path) #ia.quokka(size=(256, 256))


# bounding box
bbs = BoundingBoxesOnImage([
    BoundingBox(x1=920, y1=210, x2=986, y2=270),
    BoundingBox(x1=400, y1=344, x2=490, y2=420)
], shape=image.shape)

# Data augmentation
seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
    iaa.Affine(
        translate_px={"x": 40, "y": 60},
        scale=(0.5, 0.7)
    ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
])

# Augment BBs and images.
image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

# print coordinates before/after augmentation (see below)
# use .x1_int, .y_int, ... to get integer coordinates
for i in range(len(bbs.bounding_boxes)):
    before = bbs.bounding_boxes[i]
    after = bbs_aug.bounding_boxes[i]
    print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        i,
        before.x1, before.y1, before.x2, before.y2,
        after.x1, after.y1, after.x2, after.y2)
    )

# image with BBs before/after augmentation (shown below)
image_before = bbs.draw_on_image(image, size=3, color=[0, 0, 255])
image_after = bbs_aug.draw_on_image(image_aug, size=3, color=[0, 0, 255])

cv2.imshow("img_before", image_before)
cv2.waitKey(0)

cv2.imshow("img_after", image_after)
cv2.waitKey(0)

result_path = 'results'


k = cv2.waitKey(0) & 0xFF
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite(os.path.join(result_path, 'image_before.png'), image_before)  #  img   TODO begge bilder
    cv2.destroyAllWindows()

'''

##################################################################################################
# Tests with format of dataset and Stereo R-CNN code
##################################################################################################


def compose_data(self, l_img, r_img, l_rois, r_rois):
  data_point = list()
  # left and right images
  im_shape = l_img.shape
  im_size_min = np.min(im_shape[0:2])
  im_scale = float(462) / float(im_size_min)
  l_img = cv2.resize(l_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
  r_img = cv2.resize(r_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
  l_img = l_img.astype(np.float32, copy=False)
  r_img = r_img.astype(np.float32, copy=False)
  l_img -= np.array([[[102.9801, 115.9465, 122.7717]]])
  r_img -= np.array([[[102.9801, 115.9465, 122.7717]]])
  info = np.array([l_img.shape[0], l_img.shape[1], 1.0], dtype=np.float32)
  l_img = np.moveaxis(l_img.copy(), -1, 0)
  r_img = np.moveaxis(r_img.copy(), -1, 0)
  data_point.append(l_img)
  data_point.append(r_img)
  # Image info
  data_point.append(info)
  # left and right ROIS
  l_temp = np.zeros([30, 5])
  l_rois[:, 2] = l_rois[:, 0] + l_rois[:, 2]
  l_rois[:, 3] = l_rois[:, 1] + l_rois[:, 3]
  l_rois = l_rois * im_scale
  l_temp[0:l_rois.shape[0], 0:4] = l_rois
  l_temp[0:l_rois.shape[0], 4] = 1
  r_temp = np.zeros([30, 5])
  r_rois[:, 2] = r_rois[:, 0] + r_rois[:, 2]
  r_rois[:, 3] = r_rois[:, 1] + r_rois[:, 3]
  r_rois = r_rois * im_scale
  r_temp[0:r_rois.shape[0], 0:4] = r_rois
  r_temp[0:r_rois.shape[0], 4] = 1
  data_point.append(l_temp.copy())
  data_point.append(r_temp.copy())
  # Merged ROIS
  merge = np.zeros([30, 5])
  for i in range(30):
    merge[i, 0] = np.min([l_temp[i, 0], r_temp[i, 0]])
    merge[i, 1] = np.min([l_temp[i, 1], r_temp[i, 1]])
    merge[i, 2] = np.max([l_temp[i, 2], r_temp[i, 2]])
    merge[i, 3] = np.max([l_temp[i, 3], r_temp[i, 3]])
  merge[0:r_rois.shape[0], 4] = 1
  data_point.append(merge.copy())
  data_point.append(np.zeros([30, 5]))  # Object dimension (3) and viewpoint angle (1), orientation (1)? (?)
  data_point.append(np.zeros([30, 6]))  # Object sparse key points (?)
  data_point.append(r_rois.shape[0])    # Num objects
  return data_point.copy()


# Function build_data_set ##############################################################################################

file_path = "tests_dataset/"


ia.seed(1)
image_path = 'images/1.png'

image = cv2.imread(image_path) #ia.quokka(size=(256, 256))

data = list()



with open(file_path + "labels.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')

    for i, row in enumerate(csv_reader):
        if row[5] != '{}' and i > 0:
            attributes_dict = json.loads(row[5])
            if attributes_dict['name'] == 'roi':
                print(row[0])


                img = cv2.imread(file_path + row[0])

                #cv2.imshow("Image", img)
                #cv2.waitKey(0)

                # Left and right image in dataset
                l_img = img[:, :int(img.shape[1] / 2), :]
                r_img = img[:, int(img.shape[1] / 2) - 1:-1, :]

                #TODO ########################

                #### Augmentation ####

                #####
                images_1 = []
                images_2 = []

                images_1.append(l_img)
                images_2.append(r_img)


                # 2. Image augmentation
                # put here all the images that we want to apply
                augmentation = iaa.Sequential([

                    # 1. Horizontal flip
                    iaa.Fliplr(0.6),  # mirrored images half of the time (0.5)

                    # 3. Multiply (the cannels), make the image darker or brighter
                    iaa.Multiply((0.8, 1.4)),  # standard: 0.8, 1.2

                    # 4. Linear contrast, increasing the contrast
                    iaa.LinearContrast((0.6, 1.4)),

                    # Perform methods below only sometimes (half of the times)
                    iaa.Sometimes(1,
                                  # 5. GaussianBlur. in a dataset, the object that is in the foreground is in focus, while the rest
                                  # is blurred. So if the dataset contains some blurring, it will improve the detection of the
                                  # objects that are out of focus.
                                  iaa.GaussianBlur((0, 3.0)),
                                  iaa.Rotate((-15, 15)),

                                  # 2. Affine (image moving to right/left hand side (20% of the image are gone) ), kan bare ta i x-retning,
                                  # fjerne y, trenger ikke ha med rotate eller scale
                                  # iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                  # rotate=(-10, 10)),
                                  # scale=(0.5, 1.5)),
                                  )

                ])

                l_rois = np.array(attributes_dict['left_rois'])
                r_rois = np.array(attributes_dict['right_rois'])

                print("l_rois: ", l_rois)


                def get_left_bounding_box(l_roi):

                    if len(l_roi) == 2:
                        print("2 present rois")
                        bbs_left = BoundingBoxesOnImage([
                            BoundingBox(x1=l_roi[0][0], y1=l_roi[0][1], x2=l_roi[0][0] + l_roi[0][2],
                                        y2=l_roi[0][1] + l_roi[0][3]),
                            BoundingBox(x1=l_roi[1][0], y1=l_roi[1][1], x2=l_roi[1][0] + l_roi[1][2],
                                        y2=l_roi[1][1] + l_roi[1][3])
                        ], shape=image.shape)

                    elif len(l_roi) == 3:
                        print("3 present rois")
                        bbs_left = BoundingBoxesOnImage([
                            BoundingBox(x1=l_roi[0][0], y1=l_roi[0][1], x2=l_roi[0][0] + l_roi[0][2],
                                        y2=l_roi[0][1] + l_roi[0][3]),
                            BoundingBox(x1=l_roi[1][0], y1=l_roi[1][1], x2=l_roi[1][0] + l_roi[1][2],
                                        y2=l_roi[1][1] + l_roi[1][3]),
                            BoundingBox(x1=l_roi[2][0], y1=l_roi[2][1], x2=l_roi[2][0] + l_roi[2][2],
                                        y2=l_roi[2][1] + l_roi[2][3])
                        ], shape=image.shape)

                    elif len(l_roi) == 4:
                        print("4 present rois")
                        bbs_left = BoundingBoxesOnImage([
                            BoundingBox(x1=l_roi[0][0], y1=l_roi[0][1], x2=l_roi[0][0] + l_roi[0][2],
                                        y2=l_roi[0][1] + l_roi[0][3]),
                            BoundingBox(x1=l_roi[1][0], y1=l_roi[1][1], x2=l_roi[1][0] + l_roi[1][2],
                                        y2=l_roi[1][1] + l_roi[1][3]),
                            BoundingBox(x1=l_roi[2][0], y1=l_roi[2][1], x2=l_roi[2][0] + l_roi[2][2],
                                        y2=l_roi[2][1] + l_roi[2][3]),
                            BoundingBox(x1=l_roi[3][0], y1=l_roi[3][1], x2=l_roi[3][0] + l_roi[3][2],
                                        y2=l_roi[3][1] + l_roi[3][3])
                        ], shape=image.shape)

                    elif len(l_roi) == 5:
                        print("5 present rois")
                        bbs_left = BoundingBoxesOnImage([
                            BoundingBox(x1=l_roi[0][0], y1=l_roi[0][1], x2=l_roi[0][0] + l_roi[0][2],
                                        y2=l_roi[0][1] + l_roi[0][3]),
                            BoundingBox(x1=l_roi[1][0], y1=l_roi[1][1], x2=l_roi[1][0] + l_roi[1][2],
                                        y2=l_roi[1][1] + l_roi[1][3]),
                            BoundingBox(x1=l_roi[2][0], y1=l_roi[2][1], x2=l_roi[2][0] + l_roi[2][2],
                                        y2=l_roi[2][1] + l_roi[2][3]),
                            BoundingBox(x1=l_roi[3][0], y1=l_roi[3][1], x2=l_roi[3][0] + l_roi[3][2],
                                        y2=l_roi[3][1] + l_roi[3][3]),
                            BoundingBox(x1=l_roi[4][0], y1=l_roi[4][1], x2=l_roi[4][0] + l_roi[4][2],
                                        y2=l_roi[4][1] + l_roi[4][3]),
                        ], shape=image.shape)

                    elif len(l_roi) == 6:
                        print("6 present rois")
                        bbs_left = BoundingBoxesOnImage([
                            BoundingBox(x1=l_roi[0][0], y1=l_roi[0][1], x2=l_roi[0][0] + l_roi[0][2],
                                        y2=l_roi[0][1] + l_roi[0][3]),
                            BoundingBox(x1=l_roi[1][0], y1=l_roi[1][1], x2=l_roi[1][0] + l_roi[1][2],
                                        y2=l_roi[1][1] + l_roi[1][3]),
                            BoundingBox(x1=l_roi[2][0], y1=l_roi[2][1], x2=l_roi[2][0] + l_roi[2][2],
                                        y2=l_roi[2][1] + l_roi[2][3]),
                            BoundingBox(x1=l_roi[3][0], y1=l_roi[3][1], x2=l_roi[3][0] + l_roi[3][2],
                                        y2=l_roi[3][1] + l_roi[3][3]),
                            BoundingBox(x1=l_roi[4][0], y1=l_roi[4][1], x2=l_roi[4][0] + l_roi[4][2],
                                        y2=l_roi[4][1] + l_roi[4][3]),
                            BoundingBox(x1=l_roi[5][0], y1=l_roi[5][1], x2=l_roi[5][0] + l_roi[5][2],
                                        y2=l_roi[5][1] + l_roi[5][3]),
                        ], shape=image.shape)

                    elif len(l_roi) == 7:
                        print("7 present rois")
                        bbs_left = BoundingBoxesOnImage([
                            BoundingBox(x1=l_roi[0][0], y1=l_roi[0][1], x2=l_roi[0][0] + l_roi[0][2],
                                        y2=l_roi[0][1] + l_roi[0][3]),
                            BoundingBox(x1=l_roi[1][0], y1=l_roi[1][1], x2=l_roi[1][0] + l_roi[1][2],
                                        y2=l_roi[1][1] + l_roi[1][3]),
                            BoundingBox(x1=l_roi[2][0], y1=l_roi[2][1], x2=l_roi[2][0] + l_roi[2][2],
                                        y2=l_roi[2][1] + l_roi[2][3]),
                            BoundingBox(x1=l_roi[3][0], y1=l_roi[3][1], x2=l_roi[3][0] + l_roi[3][2],
                                        y2=l_roi[3][1] + l_roi[3][3]),
                            BoundingBox(x1=l_roi[4][0], y1=l_roi[4][1], x2=l_roi[4][0] + l_roi[4][2],
                                        y2=l_roi[4][1] + l_roi[4][3]),
                            BoundingBox(x1=l_roi[5][0], y1=l_roi[5][1], x2=l_roi[5][0] + l_roi[5][2],
                                        y2=l_roi[5][1] + l_roi[5][3]),
                            BoundingBox(x1=l_roi[6][0], y1=l_roi[6][1], x2=l_roi[6][0] + l_roi[6][2],
                                        y2=l_roi[6][1] + l_roi[6][3]),
                        ], shape=image.shape)

                    elif len(l_roi) == 8:
                        print("8 present rois")
                        bbs_left = BoundingBoxesOnImage([
                            BoundingBox(x1=l_roi[0][0], y1=l_roi[0][1], x2=l_roi[0][0] + l_roi[0][2],
                                        y2=l_roi[0][1] + l_roi[0][3]),
                            BoundingBox(x1=l_roi[1][0], y1=l_roi[1][1], x2=l_roi[1][0] + l_roi[1][2],
                                        y2=l_roi[1][1] + l_roi[1][3]),
                            BoundingBox(x1=l_roi[2][0], y1=l_roi[2][1], x2=l_roi[2][0] + l_roi[2][2],
                                        y2=l_roi[2][1] + l_roi[2][3]),
                            BoundingBox(x1=l_roi[3][0], y1=l_roi[3][1], x2=l_roi[3][0] + l_roi[3][2],
                                        y2=l_roi[3][1] + l_roi[3][3]),
                            BoundingBox(x1=l_roi[4][0], y1=l_roi[4][1], x2=l_roi[4][0] + l_roi[4][2],
                                        y2=l_roi[4][1] + l_roi[4][3]),
                            BoundingBox(x1=l_roi[5][0], y1=l_roi[5][1], x2=l_roi[5][0] + l_roi[5][2],
                                        y2=l_roi[5][1] + l_roi[5][3]),
                            BoundingBox(x1=l_roi[6][0], y1=l_roi[6][1], x2=l_roi[6][0] + l_roi[6][2],
                                        y2=l_roi[6][1] + l_roi[6][3]),
                            BoundingBox(x1=l_roi[7][0], y1=l_roi[7][1], x2=l_roi[7][0] + l_roi[7][2],
                                        y2=l_roi[7][1] + l_roi[7][3]),
                        ], shape=image.shape)

                    elif len(l_roi) == 9:
                        print("9 present rois")
                        bbs_left = BoundingBoxesOnImage([
                            BoundingBox(x1=l_roi[0][0], y1=l_roi[0][1], x2=l_roi[0][0] + l_roi[0][2],
                                        y2=l_roi[0][1] + l_roi[0][3]),
                            BoundingBox(x1=l_roi[1][0], y1=l_roi[1][1], x2=l_roi[1][0] + l_roi[1][2],
                                        y2=l_roi[1][1] + l_roi[1][3]),
                            BoundingBox(x1=l_roi[2][0], y1=l_roi[2][1], x2=l_roi[2][0] + l_roi[2][2],
                                        y2=l_roi[2][1] + l_roi[2][3]),
                            BoundingBox(x1=l_roi[3][0], y1=l_roi[3][1], x2=l_roi[3][0] + l_roi[3][2],
                                        y2=l_roi[3][1] + l_roi[3][3]),
                            BoundingBox(x1=l_roi[4][0], y1=l_roi[4][1], x2=l_roi[4][0] + l_roi[4][2],
                                        y2=l_roi[4][1] + l_roi[4][3]),
                            BoundingBox(x1=l_roi[5][0], y1=l_roi[5][1], x2=l_roi[5][0] + l_roi[5][2],
                                        y2=l_roi[5][1] + l_roi[5][3]),
                            BoundingBox(x1=l_roi[6][0], y1=l_roi[6][1], x2=l_roi[6][0] + l_roi[6][2],
                                        y2=l_roi[6][1] + l_roi[6][3]),
                            BoundingBox(x1=l_roi[7][0], y1=l_roi[7][1], x2=l_roi[7][0] + l_roi[7][2],
                                        y2=l_roi[7][1] + l_roi[7][3]),
                            BoundingBox(x1=l_roi[8][0], y1=l_roi[8][1], x2=l_roi[8][0] + l_roi[8][2],
                                        y2=l_roi[8][1] + l_roi[8][3]),
                        ], shape=image.shape)

                    elif len(l_roi) == 10:
                        print("10 present rois")
                        bbs_left = BoundingBoxesOnImage([
                            BoundingBox(x1=l_roi[0][0], y1=l_roi[0][1], x2=l_roi[0][0] + l_roi[0][2],
                                        y2=l_roi[0][1] + l_roi[0][3]),
                            BoundingBox(x1=l_roi[1][0], y1=l_roi[1][1], x2=l_roi[1][0] + l_roi[1][2],
                                        y2=l_roi[1][1] + l_roi[1][3]),
                            BoundingBox(x1=l_roi[2][0], y1=l_roi[2][1], x2=l_roi[2][0] + l_roi[2][2],
                                        y2=l_roi[2][1] + l_roi[2][3]),
                            BoundingBox(x1=l_roi[3][0], y1=l_roi[3][1], x2=l_roi[3][0] + l_roi[3][2],
                                        y2=l_roi[3][1] + l_roi[3][3]),
                            BoundingBox(x1=l_roi[4][0], y1=l_roi[4][1], x2=l_roi[4][0] + l_roi[4][2],
                                        y2=l_roi[4][1] + l_roi[4][3]),
                            BoundingBox(x1=l_roi[5][0], y1=l_roi[5][1], x2=l_roi[5][0] + l_roi[5][2],
                                        y2=l_roi[5][1] + l_roi[5][3]),
                            BoundingBox(x1=l_roi[6][0], y1=l_roi[6][1], x2=l_roi[6][0] + l_roi[6][2],
                                        y2=l_roi[6][1] + l_roi[6][3]),
                            BoundingBox(x1=l_roi[7][0], y1=l_roi[7][1], x2=l_roi[7][0] + l_roi[7][2],
                                        y2=l_roi[7][1] + l_roi[7][3]),
                            BoundingBox(x1=l_roi[8][0], y1=l_roi[8][1], x2=l_roi[8][0] + l_roi[8][2],
                                        y2=l_roi[8][1] + l_roi[8][3]),
                            BoundingBox(x1=l_roi[9][0], y1=l_roi[9][1], x2=l_roi[9][0] + l_roi[9][2],
                                        y2=l_roi[9][1] + l_roi[9][3]),
                        ], shape=image.shape)

                    elif len(l_roi) == 11:
                        print("11 present rois")
                        bbs_left = BoundingBoxesOnImage([
                            BoundingBox(x1=l_roi[0][0], y1=l_roi[0][1], x2=l_roi[0][0] + l_roi[0][2],
                                        y2=l_roi[0][1] + l_roi[0][3]),
                            BoundingBox(x1=l_roi[1][0], y1=l_roi[1][1], x2=l_roi[1][0] + l_roi[1][2],
                                        y2=l_roi[1][1] + l_roi[1][3]),
                            BoundingBox(x1=l_roi[2][0], y1=l_roi[2][1], x2=l_roi[2][0] + l_roi[2][2],
                                        y2=l_roi[2][1] + l_roi[2][3]),
                            BoundingBox(x1=l_roi[3][0], y1=l_roi[3][1], x2=l_roi[3][0] + l_roi[3][2],
                                        y2=l_roi[3][1] + l_roi[3][3]),
                            BoundingBox(x1=l_roi[4][0], y1=l_roi[4][1], x2=l_roi[4][0] + l_roi[4][2],
                                        y2=l_roi[4][1] + l_roi[4][3]),
                            BoundingBox(x1=l_roi[5][0], y1=l_roi[5][1], x2=l_roi[5][0] + l_roi[5][2],
                                        y2=l_roi[5][1] + l_roi[5][3]),
                            BoundingBox(x1=l_roi[6][0], y1=l_roi[6][1], x2=l_roi[6][0] + l_roi[6][2],
                                        y2=l_roi[6][1] + l_roi[6][3]),
                            BoundingBox(x1=l_roi[7][0], y1=l_roi[7][1], x2=l_roi[7][0] + l_roi[7][2],
                                        y2=l_roi[7][1] + l_roi[7][3]),
                            BoundingBox(x1=l_roi[8][0], y1=l_roi[8][1], x2=l_roi[8][0] + l_roi[8][2],
                                        y2=l_roi[8][1] + l_roi[8][3]),
                            BoundingBox(x1=l_roi[9][0], y1=l_roi[9][1], x2=l_roi[9][0] + l_roi[9][2],
                                        y2=l_roi[9][1] + l_roi[9][3]),
                            BoundingBox(x1=l_roi[10][0], y1=l_roi[10][1], x2=l_roi[10][0] + l_roi[10][2],
                                        y2=l_roi[10][1] + l_roi[10][3]),
                        ], shape=image.shape)

                    elif len(l_roi) == 12:
                        print("12 present rois")
                        bbs_left = BoundingBoxesOnImage([
                            BoundingBox(x1=l_roi[0][0], y1=l_roi[0][1], x2=l_roi[0][0] + l_roi[0][2],
                                        y2=l_roi[0][1] + l_roi[0][3]),
                            BoundingBox(x1=l_roi[1][0], y1=l_roi[1][1], x2=l_roi[1][0] + l_roi[1][2],
                                        y2=l_roi[1][1] + l_roi[1][3]),
                            BoundingBox(x1=l_roi[2][0], y1=l_roi[2][1], x2=l_roi[2][0] + l_roi[2][2],
                                        y2=l_roi[2][1] + l_roi[2][3]),
                            BoundingBox(x1=l_roi[3][0], y1=l_roi[3][1], x2=l_roi[3][0] + l_roi[3][2],
                                        y2=l_roi[3][1] + l_roi[3][3]),
                            BoundingBox(x1=l_roi[4][0], y1=l_roi[4][1], x2=l_roi[4][0] + l_roi[4][2],
                                        y2=l_roi[4][1] + l_roi[4][3]),
                            BoundingBox(x1=l_roi[5][0], y1=l_roi[5][1], x2=l_roi[5][0] + l_roi[5][2],
                                        y2=l_roi[5][1] + l_roi[5][3]),
                            BoundingBox(x1=l_roi[6][0], y1=l_roi[6][1], x2=l_roi[6][0] + l_roi[6][2],
                                        y2=l_roi[6][1] + l_roi[6][3]),
                            BoundingBox(x1=l_roi[7][0], y1=l_roi[7][1], x2=l_roi[7][0] + l_roi[7][2],
                                        y2=l_roi[7][1] + l_roi[7][3]),
                            BoundingBox(x1=l_roi[8][0], y1=l_roi[8][1], x2=l_roi[8][0] + l_roi[8][2],
                                        y2=l_roi[8][1] + l_roi[8][3]),
                            BoundingBox(x1=l_roi[9][0], y1=l_roi[9][1], x2=l_roi[9][0] + l_roi[9][2],
                                        y2=l_roi[9][1] + l_roi[9][3]),
                            BoundingBox(x1=l_roi[10][0], y1=l_roi[10][1], x2=l_roi[10][0] + l_roi[10][2],
                                        y2=l_roi[10][1] + l_roi[10][3]),
                            BoundingBox(x1=l_roi[11][0], y1=l_roi[11][1], x2=l_roi[11][0] + l_roi[11][2],
                                        y2=l_roi[11][1] + l_roi[11][3]),
                        ], shape=image.shape)

                    elif len(l_roi) == 13:
                        print("13 present rois")
                        bbs_left = BoundingBoxesOnImage([
                            BoundingBox(x1=l_roi[0][0], y1=l_roi[0][1], x2=l_roi[0][0] + l_roi[0][2],
                                        y2=l_roi[0][1] + l_roi[0][3]),
                            BoundingBox(x1=l_roi[1][0], y1=l_roi[1][1], x2=l_roi[1][0] + l_roi[1][2],
                                        y2=l_roi[1][1] + l_roi[1][3]),
                            BoundingBox(x1=l_roi[2][0], y1=l_roi[2][1], x2=l_roi[2][0] + l_roi[2][2],
                                        y2=l_roi[2][1] + l_roi[2][3]),
                            BoundingBox(x1=l_roi[3][0], y1=l_roi[3][1], x2=l_roi[3][0] + l_roi[3][2],
                                        y2=l_roi[3][1] + l_roi[3][3]),
                            BoundingBox(x1=l_roi[4][0], y1=l_roi[4][1], x2=l_roi[4][0] + l_roi[4][2],
                                        y2=l_roi[4][1] + l_roi[4][3]),
                            BoundingBox(x1=l_roi[5][0], y1=l_roi[5][1], x2=l_roi[5][0] + l_roi[5][2],
                                        y2=l_roi[5][1] + l_roi[5][3]),
                            BoundingBox(x1=l_roi[6][0], y1=l_roi[6][1], x2=l_roi[6][0] + l_roi[6][2],
                                        y2=l_roi[6][1] + l_roi[6][3]),
                            BoundingBox(x1=l_roi[7][0], y1=l_roi[7][1], x2=l_roi[7][0] + l_roi[7][2],
                                        y2=l_roi[7][1] + l_roi[7][3]),
                            BoundingBox(x1=l_roi[8][0], y1=l_roi[8][1], x2=l_roi[8][0] + l_roi[8][2],
                                        y2=l_roi[8][1] + l_roi[8][3]),
                            BoundingBox(x1=l_roi[9][0], y1=l_roi[9][1], x2=l_roi[9][0] + l_roi[9][2],
                                        y2=l_roi[9][1] + l_roi[9][3]),
                            BoundingBox(x1=l_roi[10][0], y1=l_roi[10][1], x2=l_roi[10][0] + l_roi[10][2],
                                        y2=l_roi[10][1] + l_roi[10][3]),
                            BoundingBox(x1=l_roi[11][0], y1=l_roi[11][1], x2=l_roi[11][0] + l_roi[11][2],
                                        y2=l_roi[11][1] + l_roi[11][3]),
                            BoundingBox(x1=l_roi[12][0], y1=l_roi[12][1], x2=l_roi[12][0] + l_roi[12][2],
                                        y2=l_roi[12][1] + l_roi[12][3]),
                        ], shape=image.shape)


                    return bbs_left

                def get_right_bounding_box(r_roi):
                    if len(r_roi) == 2:
                        print("2 present rois")
                        bbs_right = BoundingBoxesOnImage([
                            BoundingBox(x1=r_roi[0][0], y1=r_roi[0][1], x2=r_roi[0][0] + r_roi[0][2],
                                        y2=r_roi[0][1] + r_roi[0][3]),
                            BoundingBox(x1=r_roi[1][0], y1=r_roi[1][1], x2=r_roi[1][0] + r_roi[1][2],
                                        y2=r_roi[1][1] + r_roi[1][3])
                        ], shape=image.shape)

                    elif len(r_roi) == 3:
                        print("3 present rois")
                        bbs_right = BoundingBoxesOnImage([
                            BoundingBox(x1=r_roi[0][0], y1=r_roi[0][1], x2=r_roi[0][0] + r_roi[0][2],
                                        y2=r_roi[0][1] + r_roi[0][3]),
                            BoundingBox(x1=r_roi[1][0], y1=r_roi[1][1], x2=r_roi[1][0] + r_roi[1][2],
                                        y2=r_roi[1][1] + r_roi[1][3]),
                            BoundingBox(x1=r_roi[2][0], y1=r_roi[2][1], x2=r_roi[2][0] + r_roi[2][2],
                                        y2=r_roi[2][1] + r_roi[2][3])
                        ], shape=image.shape)

                    elif len(r_roi) == 4:
                        print("4 present rois")
                        bbs_right = BoundingBoxesOnImage([
                            BoundingBox(x1=r_roi[0][0], y1=r_roi[0][1], x2=r_roi[0][0] + r_roi[0][2],
                                        y2=r_roi[0][1] + r_roi[0][3]),
                            BoundingBox(x1=r_roi[1][0], y1=r_roi[1][1], x2=r_roi[1][0] + r_roi[1][2],
                                        y2=r_roi[1][1] + r_roi[1][3]),
                            BoundingBox(x1=r_roi[2][0], y1=r_roi[2][1], x2=r_roi[2][0] + r_roi[2][2],
                                        y2=r_roi[2][1] + r_roi[2][3]),
                            BoundingBox(x1=r_roi[3][0], y1=r_roi[3][1], x2=r_roi[3][0] + r_roi[3][2],
                                        y2=r_roi[3][1] + r_roi[3][3])
                    ], shape=image.shape)
                    return bbs_right


                '''
                for roi in l_rois:
                    # get_bounding_box(l_roi[i])
                    #left_roi = get_left_bounding_box(l_rois[i])
                    #print(i)
                    #left_roi = get_left_bounding_box(i)
                    left_roi = get_left_bounding_box(roi)
                    print("bound: ",left_roi)
                '''

                
                bbs_left = get_left_bounding_box(l_rois)
                bbs_right = get_right_bounding_box(r_rois)




                i = 1
                # 3. Show images  # while true makes a loop where the images dont stop(?)
                # while True:
                while i <= 20:   #  20 images created from 1 image
                    aug_det = augmentation.to_deterministic()
                    augmented_images_left, aug_bbs_left = aug_det(images=images_1, bounding_boxes=bbs_left) #augmentation(images=images_1)
                    augmented_images_right, aug_bbs_right = aug_det(images=images_2, bounding_boxes=bbs_right) #augmentation(images=images_2)
                    i = i + 1



                    print("augmented left bounding boxes: ", aug_bbs_left)
                    print("AUGmented left bounding boxes: ", aug_bbs_left[1])
                    print("augmented right bounding boxes: ", aug_bbs_right)


                    l_rois_aug = []
                    ####
                    # print coordinates before/after augmentation (see below)
                    # use .x1_int, .y_int, ... to get integer coordinates
                    for i in range(len(bbs_left.bounding_boxes)):
                        before = bbs_left.bounding_boxes[i]
                        after = aug_bbs_left.bounding_boxes[i]
                        print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
                            i,
                            before.x1, before.y1, before.x2, before.y2,
                            after.x1, after.y1, after.x2, after.y2)
                              )


                        l_roiis = [after.x1, after.y1, after.x2, after.y2]
                        l_rois_aug.append(l_roiis)

                        #print("roiiiis", l_roiis)
                        #print("real rois ", l_rois)
                        #print("HAI: ", before.x1)  # TODO MANDAG

                    print("real rois ", l_rois)
                    l_rois_aug = np.array(l_rois_aug)
                    print("aug rois ", l_rois_aug)
                    ####

                    #for img in augmented_images_left:
                    for l_img, r_img in zip(augmented_images_left, augmented_images_right):
                        #################################

                        # Left and right RoIs in dataset
                        #l_rois = np.array(attributes_dict['left_rois'])
                        #r_rois = np.array(attributes_dict['right_rois'])




                        data_point = compose_data(None, l_img, r_img, l_rois_aug, r_rois)



                        data.append(data_point)

                        #################################

                        show_l_img = aug_bbs_left.draw_on_image(l_img, size=3, color=[0, 0, 255])

                        show_r_img = aug_bbs_right.draw_on_image(r_img, size=3, color=[0, 0, 255])

                        show_img = np.hstack((show_l_img, show_r_img))

                        #show_img = aug_bbs_left.draw_on_image(show_img, size=2, color=[0, 0, 255]) # TODO BOUNDING
                        #show_img = aug_bbs_right.draw_on_image(show_img, size=2, color=[0, 0, 255])  # TODO BOUNDING

                        cv2.imshow("show_image", show_img)

                        #cv2.imshow("left_image", l_img)
                        #cv2.waitKey(0)
                        #cv2.imshow("right_image", r_img)
                        #cv2.waitKey(0)

                        result_path = 'results'
                        k = cv2.waitKey(0) & 0xFF
                        if k == 27:  # wait for ESC key to exit
                            cv2.destroyAllWindows()

                        elif k == ord('s'):  # wait for 's' key to save and exit
                            val = input("Enter your value: ")
                            print("snap")
                            cv2.imwrite(os.path.join(result_path, val + '.png'), show_img)  #  img   TODO begge bilder
                            cv2.destroyAllWindows()

                ####################

                # TODO ########################

print("Amount of image pairs to train: ", len(data))


###################################################### TESTER ##########################################################
'''
file_path = "tests_dataset/"
image_path = file_path + "D20201113T153715_200.png"

image = cv2.imread(image_path)


# Lag en funksjon som henter boundingbokser EN boks om gangen. l_rois og r_rois består av hva?
# Lager en forloop. For hver roi i l_rois: hent boundingboks.

bbs = BoundingBoxesOnImage([
    BoundingBox(x1=65, y1=100, x2=200, y2=150),
    BoundingBox(x1=150, y1=80, x2=200, y2=130)
], shape=image.shape)

print(bbs)

def get_bounding_box(l_roi):
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=l_roi[0], y1=l_roi[1], x2=l_roi[2], y2=l_roi[3])
    ], shape=image.shape)
    return bbs

l_roi = [65, 100, 200, 150], [150, 80, 200, 130]


bbs_list = []
for i in l_roi:
    #get_bounding_box(l_roi[i])
    print(i)

    bbss = get_bounding_box(i)

    bbs_list.append(bbss)



print(bbs_list)
    #i += 1

'''