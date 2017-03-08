
import os
import shutil
import sys
import json
from datetime import datetime

import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from inception import *
from transferflow.utils import transfer_model_meta
from nnpack.models import create_empty_model, save_model_benchmark_info
from nnpack import load_labels

import logging
logger = logging.getLogger("transferflow.classification")

from . import DEFAULT_SETTINGS

class Trainer(object):

    def __init__(self, base_model_path, scaffold_path, **kwargs):
        self.base_model_path = base_model_path
        self.scaffold_path = scaffold_path
        self.settings = DEFAULT_SETTINGS
        for key in kwargs:
            self.settings[key] = kwargs[key]
        if not self.settings.has_key('base_graph_path'):
            self.settings['base_graph_path'] = base_model_path + '/state/model.pb'
        self.labels = load_labels(scaffold_path)


    def prepare(self):
        settings = self.settings
        sess = tf.Session()
        self.bottleneck_tensor, self.jpeg_data_tensor, self.resized_image_tensor = load_base_graph(sess, settings['base_graph_path'])

        image_dir = self.scaffold_path + '/images'
        if not os.path.isdir(self.scaffold_path + '/cache'):
            os.mkdir(self.scaffold_path + '/cache')
        bottleneck_dir = self.scaffold_path + '/cache/bottlenecks'

        self.image_lists = create_image_lists(image_dir, settings['testing_percentage'], settings['validation_percentage'])
        class_count = len(self.image_lists.keys())

        if class_count == 0:
            raise Exception('No valid folders of images found at ' + image_dir)
        if class_count == 1:
            raise Exception('Only one valid folder of images found at ' + image_dir + ', multiple classes are needed for classification')

        # Link labels to new softmax layer
        self._add_softmax_ids_to_labels()

        self.do_distort_images = should_distort_images(settings['flip_left_right'], settings['random_crop'], settings['random_scale'], settings['random_brightness'])

        if self.do_distort_images:
            logger.debug('Distorting images')
            self.distorted_jpeg_data_tensor, self.distorted_image_tensor = add_input_distortions(settings['flip_left_right'], settings['random_crop'], settings['random_scale'], settings['random_brightness'])
        else:
            cache_bottlenecks(sess, self.image_lists, image_dir, bottleneck_dir, self.jpeg_data_tensor, self.bottleneck_tensor)


    def train(self, output_model_path):
        settings = self.settings
        image_dir = self.scaffold_path + '/images'
        bottleneck_dir = self.scaffold_path + '/bottlenecks'

        sess = tf.Session()

        final_tensor_name = 'retrained_layer'
        (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_training_ops(len(self.image_lists.keys()), final_tensor_name, self.bottleneck_tensor, settings['learning_rate'])

        # Set up all our weights to their initial default values.
        init = tf.variables_initializer(tf.global_variables())
        sess.run(init)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)
        # Run the training for as many cycles as requested on the command line.
        for i in range(settings['num_steps']):
            # Get a catch of input bottleneck values, either calculated fresh every time
            # with distortions applied, or from the cache stored on disk.
            if self.do_distort_images:
                train_bottlenecks, train_ground_truth = get_random_distorted_bottlenecks(
                    sess, self.image_lists, settings['train_batch_size'], 'training',
                    image_dir, self.distorted_jpeg_data_tensor,
                    self.distorted_image_tensor, self.resized_image_tensor, self.bottleneck_tensor)
            else:
                train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                    sess, self.image_lists, settings['train_batch_size'], 'training',
                    bottleneck_dir, image_dir, self.jpeg_data_tensor,
                    self.bottleneck_tensor)
            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step.
            sess.run(train_step,
                    feed_dict={bottleneck_input: train_bottlenecks,
                              ground_truth_input: train_ground_truth})
            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == settings['num_steps'])
            if (i % settings['eval_step_interval']) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy], feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
                logger.debug('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i, train_accuracy * 100))
                logger.debug('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))

            validation_bottlenecks, validation_ground_truth = (
                get_random_cached_bottlenecks(
                    sess, self.image_lists, settings['validation_batch_size'], 'validation',
                    bottleneck_dir, image_dir, self.jpeg_data_tensor,
                    self.bottleneck_tensor))
            validation_accuracy = sess.run(
                evaluation_step,
                feed_dict={bottleneck_input: validation_bottlenecks,
                           ground_truth_input: validation_ground_truth})
            logger.debug('%s: Step %d: Validation accuracy = %.1f%%' %
                  (datetime.now(), i, validation_accuracy * 100))

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        test_bottlenecks, test_ground_truth = get_random_cached_bottlenecks(
            sess, self.image_lists, settings['test_batch_size'], 'testing',
            bottleneck_dir, image_dir, self.jpeg_data_tensor,
            self.bottleneck_tensor)
        test_accuracy = sess.run(
            evaluation_step,
            feed_dict={bottleneck_input: test_bottlenecks,
                       ground_truth_input: test_ground_truth})
        logger.info('Model trained, final test accuracy = %.1f%%' % (test_accuracy * 100))

        benchmark_info = {
            'validation_accuracy': float(validation_accuracy),
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'cross_entropy_value': float(cross_entropy_value)
        }

        create_empty_model(output_model_path)
        transfer_model_meta(self.scaffold_path, output_model_path)
        output_graph_path = output_model_path + '/state/model.pb'

        # Write out the trained graph and labels with the weights stored as constants.
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), [final_tensor_name])
        with gfile.FastGFile(output_graph_path, 'wb') as f:
          f.write(output_graph_def.SerializeToString())

        # Persist labels with softmax IDs
        with open(output_model_path + '/labels.json', 'w') as f:
            json.dump({'labels': self.labels.values()}, f)

        # Cleanup
        tf.reset_default_graph()
        sess.close()

        return benchmark_info

    def _add_softmax_ids_to_labels(self):
        i = 0
        for label_id in self.image_lists:
            if not self.labels.has_key(label_id):
                raise Exception('Label with ID {} does not appear in labels.json, bad scaffold'.format(label_id))
            label = self.labels[label_id]
            label['node_id'] = i
            i+=1
