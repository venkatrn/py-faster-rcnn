#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__', # always index 0
		'glass_1', 'glass_2', 'glass_3', 'glass_4', 'glass_5', 
		'glass_6', 'glass_7', 'glass_8', 'wine_glass_1', 'wine_glass_2', 'wine_glass_3', 'wine_glass_4', 'wine_glass_5', 'wine_glass_6', 'cup_1', 'cup_2', 'cup_3', 'bowl_1', 'pitcher_1', 'kettle', 'all_detergent', 'all_detergent_small', 'brita_pitcher', 'clorox', 'milk_carton', 'milkjug',
		'odwalla_jug', 'orange_juice_jug', 'peroxide', 'tide',
		'pur_water_pitcher_filter', 'simple_green', 'red_mug', 'suave-3in1', 'tilex_spray', 'vf_paper_bowl')
def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, im_file, output_file):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.3 #0.8
    NMS_THRESH = 0.3 #0.3


    # Write output to file
    output = open(output_file, 'w')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
	cls_boxes = cls_boxes.transpose()
	for box_ind, box in enumerate(cls_boxes):
	    output.write(CLASSES[cls_ind] + '\n')
 	    output.write(str(cls_scores[box_ind]) + '\n')
            output.write(' '.join([str(x) for x in cls_boxes[:,box_ind]]) + '\n')
    output.close()

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
	

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices='', default='vgg16')
    parser.add_argument('--input', dest='input_file', help='Input image for detection',
                        default='')
    parser.add_argument('--output', dest='output_file', help='Output file in which detections are written',
                        default='', required=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    # prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'ZF', 'faster_rcnn_end2end',
    #                           'test.prototxt')
    # caffemodel = os.path.join(cfg.ROOT_DIR, 'output',
    #                         'faster_rcnn_end2end', 'willow_garage_2011_train',
    #                         'zf_faster_rcnn_iter_50000.caffemodel')
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'ZF', 'faster_rcnn_alt_opt',
                              'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output',
                            'faster_rcnn_alt_opt', 'willow_garage_2011_train',
                            'ZF_faster_rcnn_final.caffemodel')
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/' \
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg']
    #im_names = ['custom.png']
    im_names = [args.input_file]
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
    	im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', im_name)
        demo(net, im_file, args.output_file)

    plt.show()
