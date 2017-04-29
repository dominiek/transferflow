import tensorflow as tf
import os
from . import DEFAULT_SETTINGS
from transferflow.utils import *
from nnpack import load_labels
from copy import deepcopy
import logging
logger = logging.getLogger("transferflow.classification")
slim = tf.contrib.slim


class SlimRunner(object):
    def __init__(self, model_dir):
        pass

