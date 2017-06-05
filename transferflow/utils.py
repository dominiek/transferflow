
import os
import json
import glob
import re
import hashlib
import numpy as np
import cv2
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import tensorflow as tf
import shutil
import time
from distutils.version import LooseVersion

import logging
logger = logging.getLogger("transferflow.utils")

TENSORFLOW_VERSION = LooseVersion(tf.__version__)


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
      image_dir: String path to a folder containing subfolders of images.
      testing_percentage: Integer percentage of the images to reserve for tests.
      validation_percentage: Integer percentage of images reserved for validation.

    Returns:
      A dictionary containing an entry for each label subfolder, with images split
      into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        logger.warning("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in os.walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        logger.debug("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            logger.warning('No files found')
            continue
        label_name = re.sub(r'[^a-z0-9]+', '_', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        num_testing_images = int(round(len(file_list) * (testing_percentage / 100.0)))
        if num_testing_images < 1:
            num_testing_images = 1
        num_validation_images = int(round(len(file_list) * (validation_percentage / 100.0)))
        if num_validation_images < 1:
            num_validation_images = 1
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            if len(validation_images) < num_validation_images:
                validation_images.append(base_name)
                continue
            if len(testing_images) < num_testing_images:
                testing_images.append(base_name)
                continue
            training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def transfer_model_meta(source_path, destination_path):
    shutil.copyfile(source_path + '/nnscaffold.json', destination_path + '/nnpackage.json')
    shutil.copyfile(source_path + '/labels.json', destination_path + '/labels.json')


def tf_concat(axis, values, **kwargs):
    if TENSORFLOW_VERSION >= LooseVersion('1.0'):
        return tf.concat(values, axis, **kwargs)
    else:
        return tf.concat(axis, values, **kwargs)


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
