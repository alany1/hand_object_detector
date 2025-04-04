from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["PYTHONPATH"] = os.pathsep.join([os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib')),
                                           os.environ.get("PYTHONPATH", "")])

from tqdm import tqdm
from model.utils.config import cfg, cfg_from_file

import numpy as np
import cv2
import torch
import glob
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.resnet import resnet
from hands.ts_filtered import robust_binary_signal

MODEL_PATH = os.environ["HAND_OBJECT_MODEL_PATH"]
PASCAL_CLASSES = np.asarray(['__background__', 'targetobject', 'hand'])
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
TEST_SCALES = 600
MAX_TEST_SIZE = 1_000
BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
TEST_NMS = 0.3
THRESH_HAND = THRESH_OBJ = 0.5

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in [TEST_SCALES]:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > MAX_TEST_SIZE:
      im_scale = float(MAX_TEST_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def setup():
    cfg_from_file("cfgs/res101.yml")
    cfg.USE_GPU_NMS = True
    load_name = MODEL_PATH
    
    fasterRCNN = resnet(PASCAL_CLASSES, 101, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()
    
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if "pooling_mode" in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint["pooling_mode"]
    cfg.CUDA = True
    print('load model successfully!')
    
    return fasterRCNN

def run(image_paths, model):
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)

    # ship to cuda
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

    time_series = []
    for im_file in tqdm(image_paths):
        with torch.no_grad():
            model.cuda()
            model.eval()

            max_per_image = 100

            im_in = cv2.imread(im_file)

            # bgr
            im = im_in
            blobs, im_scales = _get_image_blob(im)

            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            with torch.no_grad():
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                box_info.resize_(1, 1, 5).zero_()

            rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, loss_list = model(im_data, im_info, gt_boxes, num_boxes, box_info)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            # extact predicted params
            contact_vector = loss_list[0][0] # hand contact state info
            offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
            lr_vector = loss_list[2][0].detach() # hand side info (left/right)

            # get hand contact 
            _, contact_indices = torch.max(contact_vector, 2)

            contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

            # get hand side 
            lr = torch.sigmoid(lr_vector) > 0.5
            lr = lr.squeeze(0).float()

            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4 * len(PASCAL_CLASSES))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

            pred_boxes /= im_scales[0]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            im2show = np.copy(im)
            obj_dets, hand_dets = None, None
            for j in range(1, len(PASCAL_CLASSES)):
                if PASCAL_CLASSES[j] == 'hand':
                    inds = torch.nonzero(scores[:,j]>THRESH_HAND).view(-1)
                elif PASCAL_CLASSES[j] == 'targetobject':
                    inds = torch.nonzero(scores[:,j]>THRESH_OBJ).view(-1)

                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], TEST_NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if PASCAL_CLASSES[j] == 'targetobject':
                        obj_dets = cls_dets.cpu().numpy()
                    if PASCAL_CLASSES[j] == 'hand':
                        hand_dets = cls_dets.cpu().numpy()

            is_contact = False
            if hand_dets is not None and obj_dets is not None:
                rh = hand_dets[0, -1] == 1
                hand_score = hand_dets[0, 4]
                obj_score = obj_dets[0, 4]
                is_contact = rh and (hand_score > THRESH_HAND) and (obj_score > THRESH_OBJ) and (hand_dets[0, 5] > 0)

            time_series.append(is_contact)

    return time_series

def hand_inference(image_paths):
    model = setup()
    time_series = run(image_paths, model)
    
    return time_series

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    image_paths = sorted(glob.glob("/home/exx/datasets/aria/real/kitchen_v2/processed/rgb/rgb_*.jpg"))
    hand_inference(image_paths)
    
