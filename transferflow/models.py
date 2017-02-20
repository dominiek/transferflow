
import os
import json
import shutil
import tensorflow as tf
from utils import *

import logging
logger = logging.getLogger("transferflow.models")

class InvalidModelError(Exception):
    pass

def validate_model(path):
    if not os.path.isdir(path):
        raise InvalidModelError('Invalid Model, should be a directory: {}'.format(path))
    if not os.path.isdir(path + '/state'):
        raise InvalidModelError('Invalid Model, expected state directory: {}'.format(path + '/state'))
    if not os.path.isfile(path + '/index.json'):
        raise InvalidModelError('Invalid Model, expected index.json')
    try:
        meta = load_meta(path)
    except Exception as e:
        raise InvalidModelError('Could not load Model meta data: {}'.format(str(e)))
    if not meta.has_key('id'):
        raise InvalidModelError('Invalid Model meta data, expected field `id` in index.json')
    if not meta.has_key('name'):
        raise InvalidModelError('Invalid Model meta data, expected field `name` in index.json')
    if not os.path.isfile(path + '/labels.jsons'):
        raise InvalidModelError('Invalid Model, expected labels.jsons')
    try:
        labels = load_labels(path)
    except Exception as e:
        raise InvalidModelError('Could not load Model labels: {}'.format(str(e)))
    for id in labels:
        if not labels[id].has_key('name'):
            raise InvalidModelError('Missing name attribute for label {}'.format(id))
    if len(labels) < 1:
        raise InvalidModelError('Expected at least 1 label in Model')

def create_empty_model(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    os.mkdir(path + '/state')

def transfer_model_meta(source_path, destination_path):
    shutil.copyfile(source_path + '/index.json', destination_path + '/index.json')
    shutil.copyfile(source_path + '/labels.jsons', destination_path + '/labels.jsons')

def save_model_state(sess, path):
    state_dir = path + '/state'
    if os.path.isdir(state_dir):
        shutil.rmtree(state_dir)
    os.mkdir(state_dir)

    #sub_graph = graph_util.extract_sub_graph(sess.graph.as_graph_def(add_shapes=True), tensors)
    #tf.train.export_meta_graph(path + '/state/model.meta', graph_def=sub_graph)
    logger.info('Saving model to {} (num_tensors={}, tensor_namespaces={})'.format(path, len(get_tensors(sess)), ','.join(get_tensor_namespaces(sess))))
    saver = tf.train.Saver()
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    tf.train.export_meta_graph(path + '/state/model.meta', graph_def=graph_def, as_text=True)
    saver.save(sess, path + '/state/model', write_meta_graph=False)

def load_model_state(sess, path, namespace=None, exclude_meta=False):
    start_ts = time.time()
    if not os.path.isdir(path):
        raise Exception('No model dir found at {}'.format(path))
    if exclude_meta:
        saver = tf.train.Saver()
    else:
        saver = tf.train.import_meta_graph(path + '/state/model.meta')
    saver.restore(sess, path + '/state/model')
    logger.info('Loaded model from {} (took={}s, num_tensors={}, tensor_namespaces={})'.format(path, time.time()-start_ts, len(get_tensors(sess)), ','.join(get_tensor_namespaces(sess))))

def save_model_benchmark_info(path, benchmark_info):
    state_dir = path + '/state'
    with open(state_dir + '/benchmark.json', 'w') as f:
        json.dump(benchmark_info, f)
