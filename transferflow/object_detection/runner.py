
import tensorflow as tf
import os
import json
from scipy.misc import imread, imresize

from utils.train_utils import calculate_rectangles, rescale_boxes
from . import DEFAULT_SETTINGS
from transferflow.utils import *
from nnpack.engines._tensorflow import load_model_state

class Runner(object):

    def __init__(self, model_file, options={}):

        settings = DEFAULT_SETTINGS
        settings['tau'] = 0.25
        settings['min_confidence'] = 0.2
        settings['show_suppressed'] = True
        for key in options:
            settings[key] = options[key]

        tf.reset_default_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        load_model_state(self.sess, model_file)

        self.pred_boxes = self.sess.graph.get_tensor_by_name('decoder_2/pred_boxes_test:0')
        self.pred_confidences = self.sess.graph.get_tensor_by_name('decoder_2/pred_confidences_test:0')
        self.x_in = self.sess.graph.get_tensor_by_name('fifo_queue_1_DequeueMany:0')

        if settings['use_rezoom']:
            self._set_rezoom(settings)
        self.settings = settings

    def run(self, image_path):
        settings = self.settings

        orig_img = imread(image_path)[:,:,:3]
        img = imresize(orig_img, (settings["image_height"], settings["image_width"]), interp='cubic')

        feed = {self.x_in: [img]}
        (np_pred_boxes, np_pred_confidences) = self.sess.run([self.pred_boxes, self.pred_confidences], feed_dict=feed)

        rects, raw_rects = calculate_rectangles(settings, np_pred_confidences, np_pred_boxes,
                                        use_stitching=True, rnn_len=settings['rnn_len'], tau=settings['tau'])
        return img, rects, raw_rects

    def _set_rezoom(self, settings):
        pred_confs_deltas = self.sess.graph.get_tensor_by_name('decoder_2/pred_confs_deltas_test:0')
        pred_boxes_deltas = self.sess.graph.get_tensor_by_name('decoder_2/pred_boxes_deltas_test:0')

        grid_area = settings['grid_height'] * settings['grid_width']
        self.pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * settings['rnn_len'], 2])), [grid_area, settings['rnn_len'], 2])
        if settings['reregress']:
            self.pred_boxes = self.pred_boxes + pred_boxes_deltas
