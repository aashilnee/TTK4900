from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

import triangulation as tri
import calibration

from stereo_calibration import StereoManager, CameraParameters, StereoParameters

sm = StereoManager()
sm.load_calibration("/Users/aashi/Documents/stereoMaps/stereoMap_padding_4.pickle")


####################### GPU ###########################
print(torch.cuda.is_available())

print(torch.cuda.current_device())

print(torch.cuda.device(0))

print(torch.cuda.device_count())

print(torch.cuda.get_device_name(0))

torch.backends.cudnn.Enabled = False
####################### GPU ###########################

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


if __name__ == '__main__':

    args = parse_args()

    np.random.seed(cfg.RNG_SEED)

    input_dir = args.load_dir + "/"
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir, 'stereo_rcnn_epoch_140_loss_-37.260189056396484.pth')  # stereo_rcnn_epoch_105_loss_-35.623756408691406.pth

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

        # TODO test images
        img_l_path = 'demo/left_det_200.png'
        img_r_path = 'demo/right_det_200.png'

        img_left = cv2.imread(img_l_path)
        img_right = cv2.imread(img_r_path)

        '''
        ################## CALIBRATION #########################################################

        # Resize images
        #img_left = cv2.resize(img_left, (3264, 1848))  # 3264x1848
        #img_right = cv2.resize(img_right, (3264, 1848))

        img_right, img_left = calibration.undistortRectify(img_right, img_left)


        # Resize images
        #img_left = cv2.resize(img_left, (1280, 642))
        #img_right = cv2.resize(img_right, (1280, 642))

        #cv2.imshow("img_left", img_left)
        #cv2.imshow("img_right", img_right)

        #cv2.imshow("Rectified images", np.concatenate((img_left, img_right), axis=1))

        #cv2.waitKey(0)
        ########################################################################################

        '''

        # rgb -> bgr
        img_left = img_left.astype(np.float32, copy=False)
        img_right = img_right.astype(np.float32, copy=False)

        img_left -= cfg.PIXEL_MEANS
        img_right -= cfg.PIXEL_MEANS

        im_shape = img_left.shape
        im_size_min = np.min(im_shape[0:2])
        im_scale = float(cfg.TRAIN.SCALES[0]) / float(im_size_min)

        img_left = cv2.resize(img_left, None, None, fx=im_scale, fy=im_scale,
                              interpolation=cv2.INTER_LINEAR)
        img_right = cv2.resize(img_right, None, None, fx=im_scale, fy=im_scale,
                               interpolation=cv2.INTER_LINEAR)

        info = np.array([[img_left.shape[0], img_left.shape[1], im_scale]], dtype=np.float32)

        img_left = torch.from_numpy(img_left)
        img_left = img_left.permute(2, 0, 1).unsqueeze(0).contiguous()

        img_right = torch.from_numpy(img_right)
        img_right = img_right.permute(2, 0, 1).unsqueeze(0).contiguous()

        info = torch.from_numpy(info)

        im_left_data.resize_(img_left.size()).copy_(img_left)
        im_right_data.resize_(img_right.size()).copy_(img_right)
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

        # TODO
        img_left = cv2.imread(img_l_path)
        img_right = cv2.imread(img_r_path)


        '''
        ################## CALIBRATION #########################################################
        # Resize images
        #img_left = cv2.resize(img_left, (3264, 1848))  # 3264x1848
        #img_right = cv2.resize(img_right, (3264, 1848))

        img_right, img_left = calibration.undistortRectify(img_right, img_left)

        # Resize images
        #img_left = cv2.resize(img_left, (1280, 642))
        #img_right = cv2.resize(img_right, (1280, 642))

        ########################################################################################
        
        '''

        im2show_left = np.copy(img_left)  #im2show_left = np.copy(cv2.imread(img_l_path))
        im2show_right = np.copy(img_right)  # im2show_right = np.copy(cv2.imread(img_r_path))
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

        for i, roi in enumerate(l_rois):
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))  # (0,0,128)
            r_score = r_rois[i, -1]
            l_score = l_rois[i, -1]
            if l_score > vis_thresh and r_score > vis_thresh:
                l_bbox = tuple(int(np.round(x)) for x in l_rois[i, :4])
                r_bbox = tuple(int(np.round(x)) for x in r_rois[i, :4])
                # Visualize detected boxes
                im2show_left = cv2.rectangle(im2show_left, l_bbox[0:2], l_bbox[2:4], color, 5)
                im2show_right = cv2.rectangle(im2show_right, r_bbox[0:2], r_bbox[2:4], color, 5)

                # Find mid point in left box
                mid_left = np.array(
                    [l_bbox[0] + int((l_bbox[2] - l_bbox[0]) / 2), l_bbox[1] + int((l_bbox[3] - l_bbox[1]) / 2)],
                    dtype=np.int)
                # Find mid point in right box
                mid_right = np.array(
                    [r_bbox[0] + int((r_bbox[2] - r_bbox[0]) / 2), r_bbox[1] + int((r_bbox[3] - r_bbox[1]) / 2)],
                    dtype=np.int)

                det_l = np.vstack((det_l, mid_left))
                det_r = np.vstack((det_r, mid_right))

                sl_key = np.array([mid_left], dtype=np.float32)
                sr_key = np.array([mid_right], dtype=np.float32)

                xyz = sm.stereopixel_to_real(sl_key, sr_key)

                print("---------------------------------------------------------", sl_key, sr_key)

                print("xyz: ", xyz)


                '''
                # Stereo camera setup parameters
                baseline = 122     # Distance between cameras [cm]
               # f = 1.8             # Camera lens focal length [mm]
                alpha = 70            # Camera field of view in the horizontal plane [degrees]

                for i in range(det_l.shape[0]):
                    # x-coordinates of midpoint keypoints
                    left_x_y = det_l[i]
                    left_x = left_x_y[0]
                    right_x_y = det_r[i]
                    right_x = right_x_y[0]
                    # Left y-coordinate of midpoint keypoints (origin: bottom left corner)
                    #left_y = 720 - left_x_y[1]  # TODO 642 - left_x_y[1]    (720 when zero offset)
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


                    #depth = tri.find_depth(right_x, left_x, img_right, img_left, baseline, alpha)
                    #x_coord = tri.find_x(left_x, right_x, left_x_y, baseline)
                    #y_coord = tri.find_y(left_x, right_x, left_x_y, baseline, img_right)





                #print("depth: ", str(depth))
                #print("x-coordinate: ", str(x_coord))
                #print("y-coordinate: ", str(y_coord))




                #print("x-coordinate: ", str(x_coord))
                #print("y-coordinate: ", str(y_coord))
                #print("disparity: ", str(disparity))
                #print("depth: ", str(depth))

                im2show_left = cv2.circle(im2show_left, tuple(det_l[i]), 1, color, 10)
                im2show_right = cv2.circle(im2show_right, tuple(det_r[i]), 1, color, 10)

                img = np.hstack((im2show_left, im2show_right))
                # Resize image
                im_scale = 0.5
                img = cv2.resize(img, None, None, fx=0.6, fy=0.8, interpolation=cv2.INTER_LINEAR)

                # Save image
                path = 'results/'
                cv2.imwrite(os.path.join(path, '200_calib.jpg'), img)

                cv2.imshow("img", img)
                cv2.waitKey()
