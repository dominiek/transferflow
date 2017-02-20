
import tensorflow as tf
import os

from . import DEFAULT_SETTINGS
from transferflow.utils import *

import logging
logger = logging.getLogger("transferflow.classification")

class Runner(object):

    def __init__(self, model_file, softmax_layer='retrained_layer:0', namespace='classification'):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.namespace = namespace
        with tf.gfile.FastGFile(os.path.join(model_file, 'state/model.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name=namespace)
        self.softmax_tensor = self.sess.graph.get_tensor_by_name(namespace + '/' + softmax_layer)

    def run(self, image_path, num_predictions=10):
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        #try:
        params = {self.namespace + '/DecodeJpeg/contents:0': image_data}
        predictions = self.sess.run(self.softmax_tensor, params)
        predictions = np.squeeze(predictions)
        #except Exception as e:
        #    logger.warning('Received error during DecodeJpeg/contents in Tensor: {}, returning empty labels'.format(e))
        #    raise Exception('Invalid image format ({}: {})'.format(type(e), e.message))
        top_k = predictions.argsort()[-num_predictions:][::-1]
        labels = []
        for node_id in top_k:
            #label_id = node_id_to_label.get(node_id, {'id': None})['id']
            score = predictions[node_id]
            labels.append({'node_id': node_id, 'score': float(score)})
            #labels.append({'score': float(score), 'id': label_id})
        return labels
