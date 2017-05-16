import sys
import tensorflow as tf
import os
from copy import deepcopy
# from . import DEFAULT_SETTINGS
import numpy as np
sys.path.append(os.path.abspath('models/slim'))
from models.slim.preprocessing import preprocessing_factory
from models.slim.nets import nets_factory
from nnpack import load_labels
import logging
logger = logging.getLogger("transferflow.classification")
slim = tf.contrib.slim


class SlimRunner(object):
    def __init__(self, model_dir, model_name):
        tf.reset_default_graph()
        self.model_dir = model_dir
        self.model_name = model_name  # ex inception_resnet_v2
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

        self.num_classes = len(self.labels_by_node_id)
        self.model_definition = nets_factory.get_network_fn(self.model_name, self.num_classes,
                                                            is_training=False)

    def get_checkpoint_path(self):
        checkpoint_dir = os.path.abspath(os.path.join(self.model_dir, 'state'))
        if tf.train.latest_checkpoint(checkpoint_dir) is None:
            return os.path.join(self.model_dir, 'state', self.model_name + '.ckpt')
        else:
            return tf.train.latest_checkpoint(checkpoint_dir)

    def run(self, image_path, num_predictions=10):

        image_size = self.model_definition.default_image_size

        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        image = tf.image.decode_jpeg(image_data, channels=3)
        processed_image = self.preprocess(image, image_size, image_size)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        logits, _ = self.model_definition(processed_images)
        probabilities = tf.nn.softmax(logits)

        self.init_fn = slim.assign_from_checkpoint_fn(
            self.get_checkpoint_path(),
            slim.get_model_variables())

        with tf.Session() as sess:
            self.init_fn(sess)
            _, probabilities = sess.run([image, probabilities])
            probabilities = np.squeeze(probabilities)
            top_k = probabilities.argsort()[-num_predictions:][::-1]

        labels = []
        for node_id in top_k:
            label = deepcopy(self.labels_by_node_id.get(node_id, None))
            score = probabilities[node_id]
            label['score'] = float(score)
            labels.append(label)
        return labels
