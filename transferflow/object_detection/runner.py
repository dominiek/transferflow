
import tensorflow as tf
import os
import json
from scipy.misc import imread, imresize

from trainer import build_forward
from utils.train_utils import calculate_rectangles, rescale_boxes
from . import DEFAULT_SETTINGS

import cv2
import argparse

class Runner(object):

    def __init__(self, checkpoint_file, options={}):

        H = DEFAULT_SETTINGS
        H['tau'] = 0.25
        H['min_confidence'] = 0.2
        H['show_suppressed'] = True
        for key in options:
            H[key] = options[key]

        tf.reset_default_graph()
        x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
        if H['use_rezoom']:
            pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
            grid_area = H['grid_height'] * H['grid_width']
            pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
            if H['reregress']:
                pred_boxes = pred_boxes + pred_boxes_deltas
        else:
            pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)

        self.pred_boxes = pred_boxes
        self.pred_logits = pred_logits
        self.pred_confidences = pred_confidences
        self.x_in = x_in
        self.H = H

        saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        saver.restore(self.sess, checkpoint_file)

    def run(self, image_path):

        pred_boxes = self.pred_boxes
        pred_logits = self.pred_logits
        pred_confidences = self.pred_confidences
        x_in = self.x_in
        H = self.H

        orig_img = imread(image_path)[:,:,:3]
        img = imresize(orig_img, (H["image_height"], H["image_width"]), interp='cubic')
        feed = {x_in: img}
        (np_pred_boxes, np_pred_confidences) = self.sess.run([pred_boxes, pred_confidences], feed_dict=feed)

        rects, raw_rects = calculate_rectangles(H, np_pred_confidences, np_pred_boxes,
                                        use_stitching=True, rnn_len=H['rnn_len'], tau=H['tau'])
        return img, rects, raw_rects
