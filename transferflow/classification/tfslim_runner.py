import tensorflow as tf
import os
from copy import deepcopy
# from . import DEFAULT_SETTINGS
import numpy as np
from models.slim.preprocessing import preprocessing_factory
from models.slim.datasets import imagenet
import importlib
from nnpack import load_labels
# from copy import deepcopy
import logging
logger = logging.getLogger("transferflow.classification")
slim = tf.contrib.slim


class SlimRunner(object):
    def __init__(self, model_dir):
        tf.reset_default_graph()
        self.model_dir = model_dir
        self.model_name = model_dir.split('/')[-1]  # ex inception_resnet_v2
        self.model_definition = importlib.import_module('models.slim.nets.' + self.model_name)
        self.preprocess = preprocessing_factory.get_preprocessing(self.model_name,
                                                                  is_training=False)
        labels = load_labels(model_dir)
        self.labels_by_node_id = {}
        for label_id in labels:
            label = labels[label_id]
            node_id = label.get('node_id', None)
            if node_id is None:
                raise Exception('No Softmax node_id is known for label {}, aborting'.format(label_id))
            self.labels_by_node_id[node_id] = label

    def run(self, image_path, num_predictions=10):

        image_size = getattr(self.model_definition, self.model_name).default_image_size

        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        image = tf.image.decode_jpeg(image_data, channels=3)
        processed_image = self.preprocess(image, image_size, image_size)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        # ex inception_resnet_v2_arg_scope
        arg_scope = getattr(self.model_definition, self.model_name + '_arg_scope')
        with slim.arg_scope(arg_scope()):
            model = getattr(self.model_definition, self.model_name)
            logits, _ = model(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        self.init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(self.model_dir, 'checkpoint', self.model_name + '.ckpt'),
            slim.get_model_variables())

        with tf.Session() as sess:
            self.init_fn(sess)
            np_image, probabilities = sess.run([image, probabilities])
            probabilities = np.squeeze(probabilities)
            top_k = probabilities.argsort()[-num_predictions:][::-1]

        labels = []
        for node_id in top_k:
            label = deepcopy(self.labels_by_node_id.get(node_id, None))
            score = probabilities[node_id]
            label['score'] = float(score)
            labels.append(label)
        return labels
