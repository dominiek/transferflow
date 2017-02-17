
import os
import json
import numpy as np
import cv2
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import tensorflow as tf
import shutil
import time

import logging
logger = logging.getLogger("transferflow.utils")

def bounding_boxes_for_scaffold(path):
    if not os.path.isdir(path):
        raise Exception('Invalid model scaffold path: {}'.format(path))
    bounding_boxes_path = path + '/bounding_boxes.json'
    if not os.path.isfile(bounding_boxes_path):
        raise Exception('No bounding boxes found in model scaffold: {}'.format(bounding_boxes_path))
    with open(bounding_boxes_path, 'r') as f:
        bounding_boxes = json.load(f)
        for bounding_box in bounding_boxes:
            bounding_box['image_path'] = path + '/' + bounding_box['image_path']
        return bounding_boxes

def draw_rectangles(orig_image, rects, min_confidence=0.1, color=(0, 0, 255)):
    image = np.copy(orig_image)
    for rect in rects:
        if rect.confidence > min_confidence:
            cv2.rectangle(image,
                (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),
                (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),
                color,
                2)
    return image

def get_tensors(sess):
    layers = []
    for op in sess.graph.get_operations():
        layers.append(op.name)
    return layers

def get_tensor_namespaces(sess):
    namespaces = []
    for op in sess.graph.get_operations():
        path = op.name.split('/')
        if len(path) > 1 and path[0] not in namespaces:
            namespaces.append(path[0])
    return namespaces

def save_model(sess, path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    os.mkdir(path + '/state')

    #sub_graph = graph_util.extract_sub_graph(sess.graph.as_graph_def(add_shapes=True), tensors)
    #tf.train.export_meta_graph(path + '/state/model.meta', graph_def=sub_graph)
    logger.info('Saving model to {} (num_tensors={}, tensor_namespaces={})'.format(path, len(get_tensors(sess)), ','.join(get_tensor_namespaces(sess))))
    saver = tf.train.Saver()
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    tf.train.export_meta_graph(path + '/state/model.meta', graph_def=graph_def, as_text=True)
    saver.save(sess, path + '/state/model', write_meta_graph=False)

def load_model(sess, path, namespace=None, exclude_meta=False):
    start_ts = time.time()
    if not os.path.isdir(path):
        raise Exception('No model dir found at {}'.format(path))
    if exclude_meta:
        saver = tf.train.Saver()
    else:
        saver = tf.train.import_meta_graph(path + '/state/model.meta')
    saver.restore(sess, path + '/state/model')
    logger.info('Loaded model from {} (took={}s, num_tensors={}, tensor_namespaces={})'.format(path, time.time()-start_ts, len(get_tensors(sess)), ','.join(get_tensor_namespaces(sess))))
