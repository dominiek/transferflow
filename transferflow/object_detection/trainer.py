
import tensorflow.contrib.slim as slim
import time
import string
import os
import threading
import tensorflow as tf
import numpy as np
from . import DEFAULT_SETTINGS
from transferflow.utils import *
from nnpack.engines._tensorflow import save_model_state
from nnpack.models import create_empty_model, save_model_benchmark_info
from nnpack.scaffolds import bounding_boxes_for_scaffold
from tensorbox import *

import logging
logger = logging.getLogger("transferflow.object_detection")

from utils import train_utils
import math

class Trainer(object):

    def __init__(self, base_model_path, scaffold_path, **kwargs):
        self.scaffold_path = scaffold_path
        self.settings = DEFAULT_SETTINGS
        for key in kwargs:
            self.settings[key] = kwargs[key]
        self.settings['base_model_ckpt'] = base_model_path + '/state/model.ckpt'

    def prepare(self):
        bounding_boxes = bounding_boxes_for_scaffold(self.scaffold_path)
        num_total = len(bounding_boxes)
        num_test = int(math.floor(num_total * (1-self.settings['train_test_ratio'])))
        self.train_images = bounding_boxes[0:(num_total-num_test)]
        self.test_images = bounding_boxes[(num_total-num_test):]
        pass

    def train(self, output_model_path):
        train_images = self.train_images
        test_images = self.test_images
        settings = self.settings

        logger.info('Invoking train() with {} training images and {} test images)'.format(len(train_images), len(test_images)))

        tf.reset_default_graph()

        start_ts = time.time()

        num_steps = settings['num_steps']

        x_in = tf.placeholder(tf.float32)
        confs_in = tf.placeholder(tf.float32)
        boxes_in = tf.placeholder(tf.float32)
        queue = {}
        enqueue_op = {}
        for phase in ['train', 'test']:
            dtypes = [tf.float32, tf.float32, tf.float32]
            grid_size = settings['grid_width'] * settings['grid_height']
            shapes = (
                [settings['image_height'], settings['image_width'], 3],
                [grid_size, settings['rnn_len'], settings['num_classes']],
                [grid_size, settings['rnn_len'], 4],
            )
            queue[phase] = tf.FIFOQueue(capacity=30, dtypes=dtypes, shapes=shapes)
            enqueue_op[phase] = queue[phase].enqueue((x_in, confs_in, boxes_in))

        def make_feed(d):
            return {
                x_in: d['image'],
                confs_in: d['confs'],
                boxes_in: d['boxes'],
                learning_rate: settings['solver']['learning_rate']
            }

        def thread_loop(sess, enqueue_op, phase, gen, stop_event):
            for d in gen:
                if stop_event.is_set():
                    return
                sess.run(enqueue_op[phase], feed_dict=make_feed(d))

        (config, loss, accuracy, train_op,
         smooth_op, global_step, learning_rate) = build(settings, queue)

        saver = tf.train.Saver(max_to_keep=None)

        logger.info('train() initialization took {}s'.format(time.time() - start_ts))

        threads = []
        with tf.Session(config=config) as sess:
            for phase in ['train', 'test']:
                # enqueue once manually to avoid thread start delay
                if phase == 'train':
                    gen = train_utils.load_data_gen(settings, train_images, jitter=settings['solver']['use_jitter'])
                if phase == 'test':
                    gen = train_utils.load_data_gen(settings, test_images)
                d = gen.next()
                sess.run(enqueue_op[phase], feed_dict=make_feed(d))
                thread_stop_event = threading.Event()
                thread = threading.Thread(target=thread_loop,
                                     args=(sess, enqueue_op, phase, gen, thread_stop_event))
                thread.stop_event = thread_stop_event
                threads.append(thread)
                thread.daemon = True
                thread.start()

            tf.set_random_seed(settings['solver']['rnd_seed'])
            sess.run(tf.global_variables_initializer())
            init_fn = slim.assign_from_checkpoint_fn(
                  settings['base_model_ckpt'],
                  [x for x in tf.global_variables() if x.name.startswith(settings['base_name']) and settings['solver']['opt'] not in x.name])
            init_fn(sess)

            # train model for N iterations
            start = time.time()
            for i in xrange(num_steps):
                display_iter = settings['logging']['display_iter']
                adjusted_lr = (settings['solver']['learning_rate'] *
                               0.5 ** max(0, (i / settings['solver']['learning_rate_step']) - 2))
                lr_feed = {learning_rate: adjusted_lr}

                if i % display_iter != 0:
                    # train network
                    batch_loss_train, _ = sess.run([loss['train'], train_op], feed_dict=lr_feed)
                else:
                    # test network every N iterations; log additional info
                    if i > 0:
                        dt = (time.time() - start) / (settings['batch_size'] * display_iter)
                    start = time.time()
                    (train_loss, test_accuracy,
                        _, _) = sess.run([loss['train'], accuracy['test'],
                                          train_op, smooth_op,
                                         ], feed_dict=lr_feed)
                    print_str = string.join([
                        'Step: %d',
                        'lr: %f',
                        'Train Loss: %.2f',
                        'Softmax Test Accuracy: %.1f%%',
                        'Time/image (ms): %.1f'
                    ], ', ')
                    logger.info(print_str %
                          (i, adjusted_lr, train_loss,
                           test_accuracy * 100, dt * 1000 if i > 0 else 0))

            for thread in threads:
                thread.stop_event.set()

            create_empty_model(output_model_path)
            transfer_model_meta(self.scaffold_path, output_model_path)
            save_model_state(sess, output_model_path)

            benchmark_info = {
                'adjusted_lr': float(adjusted_lr),
                'train_loss': float(train_loss),
                'test_accuracy': float(test_accuracy)
            }
            save_model_benchmark_info(output_model_path, benchmark_info)
            return benchmark_info
