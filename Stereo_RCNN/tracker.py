from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

import _init_paths
import os
import numpy as np
import argparse
import time
import cv2
import torch
from torch.autograd import Variable
from lib.model.utils.config import cfg
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.roi_layers import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv, kpts_transform_inv, border_transform_inv
from lib.model.stereo_rcnn.resnet import resnet
from tqdm import tqdm
from kalman import kalman
import matplotlib.pyplot as plt
import calibration

from sort import *
from stereo_calibration import StereoManager, CameraParameters, StereoParameters

#create instance of SORT
mot_tracker_left = Sort()
#mot_tracker_right = Sort()

# create instance of stereo manager (calibration)
sm = StereoManager()
sm.load_calibration("/Users/aashi/Documents/stereoMaps/stereoMap_padding_4.pickle")

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Test the Stereo R-CNN network')

    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models_stereo",
                        type=str)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=12, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=6477, type=int)
    args = parser.parse_args()
    return args


def cropped_left_right_image(img):
    #img = cv2.imread(img)
    img_shape = img.shape

    img_l = img[:int(img_shape[0] / 2), :, :]
    img_r = img[int(img_shape[0] / 2) - 1:-1, :, :]

    y_offset_value = 75

    left_img_cropped = img_l[:-y_offset_value, :, :]
    right_img_cropped = img_r[y_offset_value:, :, :]

    #print("left image shape: ", left_img_cropped.shape)
    #print("Right image shape: ", right_img_cropped.shape)
    return left_img_cropped, right_img_cropped


def raw_left_right_image(img):
    #img = cv2.imread(img)
    img_shape = img.shape

    left_img_raw = img[:int(img_shape[0] / 2), :, :]
    right_img_raw = img[int(img_shape[0] / 2) - 1:-1, :, :]

    #print("left image shape: ", left_img_raw.shape)
    #print("Right image shape: ", right_img_raw.shape)

    return left_img_raw, right_img_raw


if __name__ == '__main__':

    args = parse_args()

    np.random.seed(cfg.RNG_SEED)

    input_dir = args.load_dir + "/"
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'stereo_rcnn_epoch_60_loss_-22.396568298339844.pth')  # stereo_rcnn_epoch_105_loss_-35.623756408691406.pth

    kitti_classes = np.asarray(['__background__', 'Car'])

    # initilize the network here.
    stereoRCNN = resnet(kitti_classes, 101, pretrained=False)
    stereoRCNN.create_architecture()

    print("load checkpoint %s" % load_name)
    checkpoint = torch.load(load_name)
    stereoRCNN.load_state_dict(checkpoint['model'])
    print('load model successfully!')

    with torch.no_grad():
        # initilize the tensor holder here.
        im_left_data = Variable(torch.FloatTensor(1).cpu())
        im_right_data = Variable(torch.FloatTensor(1).cpu())
        im_info = Variable(torch.FloatTensor(1).cpu())
        num_boxes = Variable(torch.LongTensor(1).cpu())
        gt_boxes = Variable(torch.FloatTensor(1).cpu())

        eval_thresh = 0.00
        vis_thresh = 0.01

        stereoRCNN.eval()


        video_in = "/Users/aashi/Documents/videos/D20201113T180117.mp4"

        #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        #video_out = cv2.VideoWriter("/Users/aashi/Documents/videos/out/test.mp4", fourcc, 25.0, (3264, 924))

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_out = cv2.VideoWriter('/Users/aashi/Documents/videos/out/test_sort.mp4', fourcc, 20.0, (2560, 720))

        cap = cv2.VideoCapture(video_in)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


        dictionary = {}
        img_nr = 1
        frame_num_1 = 1
        velocity_X = []
        velocity_Y = []
        velocity_Z = []
        #ID_xyz = list()
        path = 'results/3D_pos/'
        with open(path + '3D_pos.txt', 'w') as f:

            #for image in glob.glob(image_path):
            for frame_num in tqdm(range(1000, 2000)):

                print("frame num: ", frame_num_1)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, image = cap.read()
                #im_shape = image.shape

                l_img_cropped, r_img_cropped = cropped_left_right_image(image)

                # stack = np.hstack((l_img_cropped, r_img_cropped))

                # im_scale = 0.5
                # staaaack = cv2.resize(stack, None, None, fx=0.6, fy=0.8, interpolation=cv2.INTER_LINEAR)
                # cv2.imshow("stack", stack)
                # cv2.waitKey(0)

                # rgb -> bgr
                l_img_cropped = l_img_cropped.astype(np.float32, copy=False)
                r_img_cropped = r_img_cropped.astype(np.float32, copy=False)

                l_img_cropped -= cfg.PIXEL_MEANS
                r_img_cropped -= cfg.PIXEL_MEANS

                im_shape = l_img_cropped.shape
                im_size_min = np.min(im_shape[0:2])
                im_scale = float(cfg.TRAIN.SCALES[0]) / float(im_size_min)

                l_img_cropped = cv2.resize(l_img_cropped, None, None, fx=im_scale, fy=im_scale,
                                           interpolation=cv2.INTER_LINEAR)
                r_img_cropped = cv2.resize(r_img_cropped, None, None, fx=im_scale, fy=im_scale,
                                           interpolation=cv2.INTER_LINEAR)

                info = np.array([[l_img_cropped.shape[0], l_img_cropped.shape[1], im_scale]], dtype=np.float32)

                l_img_cropped = torch.from_numpy(l_img_cropped)
                l_img_cropped = l_img_cropped.permute(2, 0, 1).unsqueeze(0).contiguous()

                r_img_cropped = torch.from_numpy(r_img_cropped)
                r_img_cropped = r_img_cropped.permute(2, 0, 1).unsqueeze(0).contiguous()

                info = torch.from_numpy(info)

                im_left_data.resize_(l_img_cropped.size()).copy_(l_img_cropped)
                im_right_data.resize_(r_img_cropped.size()).copy_(r_img_cropped)
                im_info.resize_(info.size()).copy_(info)

                det_tic = time.time()
                rois_left, rois_right, cls_prob, bbox_pred, bbox_pred_dim, kpts_prob, \
                left_prob, right_prob, rpn_loss_cls, rpn_loss_box_left_right, \
                RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts, rois_label = \
                    stereoRCNN(im_left_data, im_right_data, im_info, gt_boxes, gt_boxes, gt_boxes, gt_boxes, gt_boxes,
                               num_boxes)

                scores = cls_prob.data
                boxes_left = rois_left.data[:, :, 1:5]
                boxes_right = rois_right.data[:, :, 1:5]

                bbox_pred = bbox_pred.data
                box_delta_left = bbox_pred.new(bbox_pred.size()[1], 4 * len(kitti_classes)).zero_()
                box_delta_right = bbox_pred.new(bbox_pred.size()[1], 4 * len(kitti_classes)).zero_()

                for keep_inx in range(box_delta_left.size()[0]):
                    box_delta_left[keep_inx, 0::4] = bbox_pred[0, keep_inx, 0::6]
                    box_delta_left[keep_inx, 1::4] = bbox_pred[0, keep_inx, 1::6]
                    box_delta_left[keep_inx, 2::4] = bbox_pred[0, keep_inx, 2::6]
                    box_delta_left[keep_inx, 3::4] = bbox_pred[0, keep_inx, 3::6]

                    box_delta_right[keep_inx, 0::4] = bbox_pred[0, keep_inx, 4::6]
                    box_delta_right[keep_inx, 1::4] = bbox_pred[0, keep_inx, 1::6]
                    box_delta_right[keep_inx, 2::4] = bbox_pred[0, keep_inx, 5::6]
                    box_delta_right[keep_inx, 3::4] = bbox_pred[0, keep_inx, 3::6]

                box_delta_left = box_delta_left.view(-1, 4)
                box_delta_right = box_delta_right.view(-1, 4)

                box_delta_left = box_delta_left * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cpu() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cpu()
                box_delta_right = box_delta_right * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cpu() \
                                  + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cpu()

                box_delta_left = box_delta_left.view(1, -1, 4 * len(kitti_classes))
                box_delta_right = box_delta_right.view(1, -1, 4 * len(kitti_classes))

                pred_boxes_left = bbox_transform_inv(boxes_left, box_delta_left, 1)
                pred_boxes_right = bbox_transform_inv(boxes_right, box_delta_right, 1)

                pred_boxes_left = clip_boxes(pred_boxes_left, im_info.data, 1)
                pred_boxes_right = clip_boxes(pred_boxes_right, im_info.data, 1)

                pred_boxes_left /= im_info[0, 2].data
                pred_boxes_right /= im_info[0, 2].data

                scores = scores.squeeze()[:, 1]
                pred_boxes_left = pred_boxes_left.squeeze()
                pred_boxes_right = pred_boxes_right.squeeze()

                det_toc = time.time()
                detect_time = det_toc - det_tic

                # TODO cropped bilder
                # l_img_cropped, r_img_cropped = cropped_left_right_image(image)
                # im2show_left = np.copy(l_img_cropped)  # im2show_left = np.copy(cv2.imread(img_l_path))
                # im2show_right = np.copy(r_img_cropped)  # im2show_right = np.copy(cv2.imread(img_r_path))
                ## TODO

                # TODO raw bilder
                l_img_raw, r_img_raw = raw_left_right_image(image)
                im2show_left = np.copy(l_img_raw)
                im2show_right = np.copy(r_img_raw)
                # TODO

                inds = torch.nonzero(scores > eval_thresh).view(-1)

                if inds.numel() > 0:
                    cls_scores = scores[inds]
                    _, order = torch.sort(cls_scores, 0, True)

                det_l = np.zeros([0, 2], dtype=np.int)
                det_r = np.zeros([0, 2], dtype=np.int)
                det_3d = np.zeros([0, 3], dtype=np.int)

                cls_boxes_left = pred_boxes_left[inds][:, 4:8]
                cls_boxes_right = pred_boxes_right[inds][:, 4:8]

                cls_dets_left = torch.cat((cls_boxes_left, cls_scores.unsqueeze(1)), 1)
                cls_dets_right = torch.cat((cls_boxes_right, cls_scores.unsqueeze(1)), 1)

                cls_dets_left = cls_dets_left[order]
                cls_dets_right = cls_dets_right[order]

                keep = nms(cls_boxes_left[order, :], cls_scores[order], cfg.TEST.NMS)
                keep = keep.view(-1).long()
                cls_dets_left = cls_dets_left[keep]
                cls_dets_right = cls_dets_right[keep]

                l_rois = cls_dets_left.cpu().numpy()
                r_rois = cls_dets_right.cpu().numpy()


                left_detections = list()
                right_detections = list()
                stored_xyz = list()

                for i, roi in enumerate(l_rois):
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))  # (0,0,128)
                    r_score = r_rois[i, -1]
                    l_score = l_rois[i, -1]
                    if l_score > vis_thresh and r_score > vis_thresh:

                        l_bbox = tuple(int(np.round(x)) for x in l_rois[i, :4])
                        r_bbox = tuple(int(np.round(x)) for x in r_rois[i, :4])
                        # Adjust detected boxes for offset in right image
                        offset_value = 75
                        r_bbox = (r_bbox[0], r_bbox[1] + offset_value, r_bbox[2], r_bbox[3] + offset_value)


                        ####### detection list for tracking
                        left_det = list(l_bbox)
                        left_det.append(l_score)
                        right_det = list(r_bbox)
                        right_det.append(r_score)

                        left_detections.append(left_det)
                        right_detections.append(right_det)

                        ########

                        # Visualize detected boxes
                        #im2show_left = cv2.rectangle(im2show_left, l_bbox[0:2], l_bbox[2:4], color, 5) # TODO O
                        im2show_right = cv2.rectangle(im2show_right, r_bbox[0:2], r_bbox[2:4], color, 5)  # TODO O

                        # Find mid point in left box
                        mid_left = np.array(
                            [l_bbox[0] + int((l_bbox[2] - l_bbox[0]) / 2),
                             l_bbox[1] + int((l_bbox[3] - l_bbox[1]) / 2)],
                            dtype=np.int)
                        # Find mid point in right box
                        mid_right = np.array(
                            [r_bbox[0] + int((r_bbox[2] - r_bbox[0]) / 2),
                             r_bbox[1] + int((r_bbox[3] - r_bbox[1]) / 2)],
                            dtype=np.int)

                        det_l = np.vstack((det_l, mid_left))
                        det_r = np.vstack((det_r, mid_right))

                        #print("det_l: ", det_l)
                        #print("det_r: ", det_r)

                        #print("mid_left: ", mid_left)
                        #print("mid_right: ", mid_right)

                        #im2show_left = cv2.circle(im2show_left, tuple(det_l[i]), 1, color, 10) # TODO O
                        im2show_right = cv2.circle(im2show_right, tuple(det_r[i]), 1, color, 10) # TODO O

                        disparity = mid_left[0] - mid_right[0]
                        #print("Disparity: ", disparity)

                        sl_key = np.array([mid_left], dtype=np.float32)
                        sr_key = np.array([mid_right], dtype=np.float32)

                        xyz, disparity_points = sm.stereopixel_to_real(sl_key, sr_key)

                        disparity_x = disparity_points[0][0]

                        xyz = xyz[0]
                        # change y-direction upwards
                        xyz[1] = -xyz[1]
                        stored_xyz.append(list(xyz))
                        #print("xyz: ", xyz)

                #print("stored xyz: ", stored_xyz)


                ############# TRACK
                #print("left dett: ", left_detections)
                #print("right dett: ", right_detections)

                left_detections = np.array(left_detections)
                right_detections = np.array(right_detections)

                track_bbs_ids_left = mot_tracker_left.update(left_detections)
                #track_bbs_ids_right = mot_tracker_right.update(right_detections)

                #print("track left: ", track_bbs_ids_left)
                #print("track right: ", track_bbs_ids_right)

                '''
                colors = {1: (0, 128, 0),
                          2: (255, 255, 255),
                          3: (0, 128, 128),
                          4: (0, 0, 0),
                          5: (128, 0, 0),
                          6: (0, 0, 255),
                          7: (0, 0, 128),
                          8: (128, 128, 128),
                          9: (128, 0, 128),
                          10: (0, 255, 255),
                          11: (0, 255, 0),
                          12: (255, 0, 255),
                          13: (255, 255, 0),
                          14: (255, 0, 0),
                          15: (128, 128, 0),
                          16: (192, 192, 192),
                          17: (0, 100, 180),
                          18: (20, 40, 112),
                          19: (92, 128, 92),
                          20: (30, 90, 255),
                          21: (110, 0, 0),
                          22: (0, 0, 240),
                          23: (0, 240, 128),
                          24: (112, 112, 0),
                          25: (112, 112, 112),
                          26: (74, 100, 200),
                          27: (32, 32, 128),
                          28: (200, 0, 200),
                          29: (255, 100, 0),
                          30: (255, 90, 90),
                          31: (128, 32, 100),
                          32: (100, 192, 80),
                          33: (0, 128, 0),
                          34: (255, 255, 255),
                          35: (0, 128, 128),
                          36: (0, 0, 0),
                          37: (128, 0, 0),
                          38: (0, 0, 255),
                          39: (0, 0, 128),
                          40: (128, 128, 128),
                          41: (128, 0, 128),
                          42: (0, 255, 255),
                          43: (0, 255, 0),
                          44: (255, 0, 255),
                          45: (255, 255, 0),
                          46: (255, 0, 0),
                          47: (128, 128, 0),
                          48: (192, 192, 192),
                          49: (0, 128, 0),
                          50: (255, 255, 255),
                          51: (0, 128, 128),
                          52: (0, 0, 0),
                          53: (128, 0, 0),
                          54: (0, 0, 255),
                          55: (0, 0, 128),
                          56: (128, 128, 128),
                          57: (128, 0, 128),
                          58: (0, 255, 255),
                          59: (0, 255, 0),
                          60: (255, 0, 255),
                          61: (255, 255, 0),
                          62: (255, 0, 0),
                          63: (128, 128, 0),
                          64: (192, 192, 192)}
                '''
                y1_det_left = list()
                for i in left_detections:
                    y1 = i[1]
                    y1_det_left.append(y1)

                y1_det_left = np.array(y1_det_left)

                colors = (255, 0, 0)

                for i, tracked_left in enumerate(track_bbs_ids_left):
                    tracked_l = tuple(int(np.round(x)) for x in track_bbs_ids_left[i, :5])
                    im2show_left = cv2.rectangle(im2show_left, tracked_l[0:2], tracked_l[2:4], colors, 5)  # FARGE: colors[tracked_l[4]]

                    #l_track = list(tracked_l)
                    im2show_left = cv2.putText(im2show_left, 'ID: {:}'.format(tracked_l[4]),
                                               (tracked_l[0], tracked_l[1] - 15),
                                               cv2.FONT_HERSHEY_DUPLEX,
                                               1, colors, 1, cv2.LINE_AA) # Farge: colors[tracked_l[4]]

                    #########
                    y1_tracked = tracked_left[1]

                    diff = np.absolute(y1_det_left - y1_tracked)
                    index = diff.argmin()
                    #print("Index in detections that corresponds with tracked object: ", index)

                    # xyz of detected object with corresponding ID
                    ID_xyz_item = [tracked_left[4], stored_xyz[index][0], stored_xyz[index][1], stored_xyz[index][2]]

                    if ID_xyz_item[0] in dictionary:
                        #print("add new item to existing key in dictionary")
                        dictionary[ID_xyz_item[0]].append(ID_xyz_item[1:4])
                    if not ID_xyz_item[0] in dictionary:
                        #print("add new key to dictionary")
                        dictionary[ID_xyz_item[0]] = []
                        dictionary[ID_xyz_item[0]].append(ID_xyz_item[1:4])

                print("dictionary: ", dictionary)
                #print("amount of samples in key 1", dictionary[1.0])
                    #print("amount of samples in key 1", dictionary[1.0])


                if frame_num_1 % 25 == 0:  # TODO 25
                    for key in dictionary.keys():
                        print("TEST HER")
                        print("length of values inside a key: ", len(dictionary[key]))
                        if len(dictionary[key]) == 25:  # TODO 25
                            x0, y0, z0, x1, y1, z1 = kalman(dictionary[key])
                            print("coordinates start, cordinates end: ", x0, y0, z0, x1, y1, z1)
                            t = 1
                            velocity_x = abs(x1 - x0)/t
                            velocity_y = abs(y1 - y0)/t
                            velocity_z = abs(z1 - z0)/t

                            velocity_X.append(velocity_x)
                            velocity_Y.append(velocity_y)
                            velocity_Z.append(velocity_z)

                            print("Estimated velocity (x) for ID {}:".format(key), velocity_x)
                            print("Estimated velocity (y) for ID {}:".format(key), velocity_y)
                            print("Estimated velocity (z) for ID {}:".format(key), velocity_z)
                        else:
                            print("Detected object with ID {} not tracked for the last 25 frames.".format(key))
                    if frame_num_1 > 1:
                        dictionary = {}


                ########################

                img_new = np.hstack((im2show_left, im2show_right))

                img_shape = img_new.shape
                print("IMAGE SHAPE: ", img_shape)
                img_write = img_new.copy()
                # Resize image
                im_scale = 0.5
                img_new = cv2.resize(img_new, None, None, fx=0.6, fy=0.8, interpolation=cv2.INTER_LINEAR)

                # Save image
                #cv2.imwrite(os.path.join(path, str(img_nr) + '.jpg'), img_new)

                cv2.imshow("img", img_new)
                if cv2.waitKey(1) == ord('q'):
                    break

                frame_num_1 += 1
                img_nr += 1
                video_out.write(img_write)

        cap.release()
        video_out.release()
        cv2.destroyAllWindows()


        # Plot histogram of velocity (x)
        plt.style.use('seaborn-deep')
        plt.hist(velocity_X, edgecolor='black')
        plt.xlabel('Velocity (mm/s)', fontsize=24)
        plt.ylabel('Number of velocity measurements', fontsize=24)
        plt.title('Velocity in x-direction', fontsize=28)
        plt.xticks(size=24)
        plt.yticks(size=24)
        plt.show()

        # Plot histogram of velocity (y)
        plt.hist(velocity_Y, edgecolor='black')
        plt.xlabel('Velocity (mm/s)', fontsize=24)
        plt.ylabel('Number of velocity measurements', fontsize=24)
        plt.title('Velocity in y-direction', fontsize=28)
        plt.xticks(size=24)
        plt.yticks(size=24)
        plt.show()

        # Plot histogram of velocity (z)
        plt.hist(velocity_Z, edgecolor='black')
        plt.xlabel('Velocity (mm/s)', fontsize=24)
        plt.ylabel('Number of velocity measurements', fontsize=24)
        plt.title('Velocity in z-direction', fontsize=28)
        plt.xticks(size=24)
        plt.yticks(size=24)
        plt.show()


