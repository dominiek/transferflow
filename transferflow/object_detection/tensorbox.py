
import random
import time
import string
import os
import tensorflow as tf
import numpy as np
from distutils.version import LooseVersion
if LooseVersion(tf.__version__) >= LooseVersion('1.0'):
    rnn_cell = tf.contrib.rnn
else:
    try:
        from tensorflow.models.rnn import rnn_cell
    except ImportError:
        rnn_cell = tf.nn.rnn_cell
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

random.seed(0)
np.random.seed(0)

from transferflow.utils import tf_concat
from utils import train_utils, googlenet_load


@ops.RegisterGradient("Hungarian")
def _hungarian_grad(op, *args):
    return map(array_ops.zeros_like, op.inputs)

def build_lstm_inner(settings, lstm_input):
    '''
    build lstm decoder
    '''
    lstm_cell = rnn_cell.BasicLSTMCell(settings['lstm_size'], forget_bias=0.0, state_is_tuple=False)
    if settings['num_lstm_layers'] > 1:
        lstm = rnn_cell.MultiRNNCell([lstm_cell] * settings['num_lstm_layers'], state_is_tuple=False)
    else:
        lstm = lstm_cell

    batch_size = settings['batch_size'] * settings['grid_height'] * settings['grid_width']
    state = tf.zeros([batch_size, lstm.state_size])

    outputs = []
    with tf.variable_scope('RNN', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
        for time_step in range(settings['rnn_len']):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            output, state = lstm(lstm_input, state)
            outputs.append(output)
    return outputs

def build_overfeat_inner(settings, lstm_input):
    '''
    build simple overfeat decoder
    '''
    if settings['rnn_len'] > 1:
        raise ValueError('rnn_len > 1 only supported with use_lstm == True')
    outputs = []
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('Overfeat', initializer=initializer):
        w = tf.get_variable('ip', shape=[settings['later_feat_channels'], settings['lstm_size']])
        outputs.append(tf.matmul(lstm_input, w))
    return outputs

def deconv(x, output_shape, channels):
    k_h = 2
    k_w = 2
    w = tf.get_variable('w_deconv', initializer=tf.random_normal_initializer(stddev=0.01),
                        shape=[k_h, k_w, channels[1], channels[0]])
    y = tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, k_h, k_w, 1], padding='VALID')
    return y

def rezoom(settings, pred_boxes, early_feat, early_feat_channels, w_offsets, h_offsets):
    '''
    Rezoom into a feature map at multiple interpolation points in a grid.

    If the predicted object center is at X, len(w_offsets) == 3, and len(h_offsets) == 5,
    the rezoom grid will look as follows:

    [o o o]
    [o o o]
    [o X o]
    [o o o]
    [o o o]

    Where each letter indexes into the feature map with bilinear interpolation
    '''


    grid_size = settings['grid_width'] * settings['grid_height']
    outer_size = grid_size * settings['batch_size']
    indices = []
    for w_offset in w_offsets:
        for h_offset in h_offsets:
            indices.append(train_utils.bilinear_select(settings,
                                                       pred_boxes,
                                                       early_feat,
                                                       early_feat_channels,
                                                       w_offset, h_offset))

    interp_indices = tf_concat(0, indices)
    rezoom_features = train_utils.interp(early_feat,
                                         interp_indices,
                                         early_feat_channels)
    rezoom_features_r = tf.reshape(rezoom_features,
                                   [len(w_offsets) * len(h_offsets),
                                    outer_size,
                                    settings['rnn_len'],
                                    early_feat_channels])
    rezoom_features_t = tf.transpose(rezoom_features_r, [1, 2, 0, 3])
    return tf.reshape(rezoom_features_t,
                      [outer_size,
                       settings['rnn_len'],
                       len(w_offsets) * len(h_offsets) * early_feat_channels])

def build_forward(settings, x, phase, reuse):
    '''
    Construct the forward model
    '''


    grid_size = settings['grid_width'] * settings['grid_height']
    outer_size = grid_size * settings['batch_size']
    input_mean = 117.
    x -= input_mean
    cnn, early_feat = googlenet_load.model(x, settings, reuse)
    early_feat_channels = settings['early_feat_channels']
    early_feat = early_feat[:, :, :, :early_feat_channels]

    if settings['deconv']:
        size = 3
        stride = 2
        pool_size = 5

        with tf.variable_scope("deconv", reuse=reuse):
            w = tf.get_variable('conv_pool_w', shape=[size, size, settings['later_feat_channels'], settings['later_feat_channels']],
                                initializer=tf.random_normal_initializer(stddev=0.01))
            cnn_s = tf.nn.conv2d(cnn, w, strides=[1, stride, stride, 1], padding='SAME')
            cnn_s_pool = tf.nn.avg_pool(cnn_s[:, :, :, :256], ksize=[1, pool_size, pool_size, 1],
                                        strides=[1, 1, 1, 1], padding='SAME')

            cnn_s_with_pool = tf_concat(3, [cnn_s_pool, cnn_s[:, :, :, 256:]])
            cnn_deconv = deconv(cnn_s_with_pool, output_shape=[settings['batch_size'], settings['grid_height'], settings['grid_width'], 256], channels=[settings['later_feat_channels'], 256])
            cnn = tf_concat(3, (cnn_deconv, cnn[:, :, :, 256:]))

    elif settings['avg_pool_size'] > 1:
        pool_size = settings['avg_pool_size']
        cnn1 = cnn[:, :, :, :700]
        cnn2 = cnn[:, :, :, 700:]
        cnn2 = tf.nn.avg_pool(cnn2, ksize=[1, pool_size, pool_size, 1],
                              strides=[1, 1, 1, 1], padding='SAME')
        cnn = tf_concat(3, [cnn1, cnn2])

    cnn = tf.reshape(cnn,
                     [settings['batch_size'] * settings['grid_width'] * settings['grid_height'], settings['later_feat_channels']])
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('decoder', reuse=reuse, initializer=initializer):
        scale_down = 0.01
        lstm_input = tf.reshape(cnn * scale_down, (settings['batch_size'] * grid_size, settings['later_feat_channels']))
        if settings['use_lstm']:
            lstm_outputs = build_lstm_inner(settings, lstm_input)
        else:
            lstm_outputs = build_overfeat_inner(settings, lstm_input)

        pred_boxes = []
        pred_logits = []
        for k in range(settings['rnn_len']):
            output = lstm_outputs[k]
            if phase == 'train':
                output = tf.nn.dropout(output, 0.5)
            box_weights = tf.get_variable('box_ip%d' % k,
                                          shape=(settings['lstm_size'], 4))
            conf_weights = tf.get_variable('conf_ip%d' % k,
                                           shape=(settings['lstm_size'], settings['num_classes']))

            pred_boxes_step = tf.reshape(tf.matmul(output, box_weights) * 50,
                                         [outer_size, 1, 4])

            pred_boxes.append(pred_boxes_step)
            pred_logits.append(tf.reshape(tf.matmul(output, conf_weights),
                                         [outer_size, 1, settings['num_classes']]))

        pred_boxes = tf_concat(1, pred_boxes, name='pred_boxes_{}'.format(phase))
        pred_logits = tf_concat(1, pred_logits, name='pred_logits_{}'.format(phase))
        pred_logits_squash = tf.reshape(pred_logits,
                                        [outer_size * settings['rnn_len'], settings['num_classes']])
        pred_confidences_squash = tf.nn.softmax(pred_logits_squash)
        pred_confidences = tf.reshape(pred_confidences_squash,
                                      [outer_size, settings['rnn_len'], settings['num_classes']], name='pred_confidences_{}'.format(phase))

        if settings['use_rezoom']:
            pred_confs_deltas = []
            pred_boxes_deltas = []
            w_offsets = settings['rezoom_w_coords']
            h_offsets = settings['rezoom_h_coords']
            num_offsets = len(w_offsets) * len(h_offsets)
            rezoom_features = rezoom(settings, pred_boxes, early_feat, early_feat_channels, w_offsets, h_offsets)
            if phase == 'train':
                rezoom_features = tf.nn.dropout(rezoom_features, 0.5)
            for k in range(settings['rnn_len']):
                delta_features = tf_concat(1, [lstm_outputs[k], rezoom_features[:, k, :] / 1000.])
                dim = 128
                delta_weights1 = tf.get_variable(
                                    'delta_ip1%d' % k,
                                    shape=[settings['lstm_size'] + early_feat_channels * num_offsets, dim])
                # TODO: add dropout here ?
                ip1 = tf.nn.relu(tf.matmul(delta_features, delta_weights1))
                if phase == 'train':
                    ip1 = tf.nn.dropout(ip1, 0.5)
                delta_confs_weights = tf.get_variable(
                                    'delta_ip2%d' % k,
                                    shape=[dim, settings['num_classes']])
                if settings['reregress']:
                    delta_boxes_weights = tf.get_variable(
                                        'delta_ip_boxes%d' % k,
                                        shape=[dim, 4])
                    pred_boxes_deltas.append(tf.reshape(tf.matmul(ip1, delta_boxes_weights) * 5,
                                                        [outer_size, 1, 4]))
                scale = settings.get('rezoom_conf_scale', 50)
                pred_confs_deltas.append(tf.reshape(tf.matmul(ip1, delta_confs_weights) * scale,
                                                    [outer_size, 1, settings['num_classes']]))
            pred_confs_deltas = tf_concat(1, pred_confs_deltas, name='pred_confs_deltas_{}'.format(phase))
            if settings['reregress']:
                pred_boxes_deltas = tf_concat(1, pred_boxes_deltas, name='pred_boxes_deltas_{}'.format(phase))
            return pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas

    return pred_boxes, pred_logits, pred_confidences

def build_forward_backward(settings, x, phase, boxes, flags):
    '''
    Call build_forward() and then setup the loss functions
    '''

    grid_size = settings['grid_width'] * settings['grid_height']
    outer_size = grid_size * settings['batch_size']
    reuse = {'train': None, 'test': True}[phase]
    if settings['use_rezoom']:
        (pred_boxes, pred_logits,
         pred_confidences, pred_confs_deltas, pred_boxes_deltas) = build_forward(settings, x, phase, reuse)
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(settings, x, phase, reuse)
    with tf.variable_scope('decoder', reuse={'train': None, 'test': True}[phase]):
        outer_boxes = tf.reshape(boxes, [outer_size, settings['rnn_len'], 4])
        outer_flags = tf.cast(tf.reshape(flags, [outer_size, settings['rnn_len']]), 'int32')
        if settings['use_lstm']:
            this_dir = os.path.dirname(os.path.realpath(__file__))
            hungarian_module = tf.load_op_library(this_dir + '/utils/hungarian/hungarian.so')
            assignments, classes, perm_truth, pred_mask = (
                hungarian_module.hungarian(pred_boxes, outer_boxes, outer_flags, settings['solver']['hungarian_iou']))
        else:
            classes = tf.reshape(flags, (outer_size, 1))
            perm_truth = tf.reshape(outer_boxes, (outer_size, 1, 4))
            pred_mask = tf.reshape(tf.cast(tf.greater(classes, 0), 'float32'), (outer_size, 1, 1))
        true_classes = tf.reshape(tf.cast(tf.greater(classes, 0), 'int64'),
                                  [outer_size * settings['rnn_len']])
        pred_logit_r = tf.reshape(pred_logits,
                                  [outer_size * settings['rnn_len'], settings['num_classes']])
        confidences_loss = (tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_logit_r, labels=true_classes))
            ) / outer_size * settings['solver']['head_weights'][0]
        residual = tf.reshape(perm_truth - pred_boxes * pred_mask,
                              [outer_size, settings['rnn_len'], 4])
        boxes_loss = tf.reduce_sum(tf.abs(residual)) / outer_size * settings['solver']['head_weights'][1]
        if settings['use_rezoom']:
            if settings['rezoom_change_loss'] == 'center':
                error = (perm_truth[:, :, 0:2] - pred_boxes[:, :, 0:2]) / tf.maximum(perm_truth[:, :, 2:4], 1.)
                square_error = tf.reduce_sum(tf.square(error), 2)
                inside = tf.reshape(tf.to_int64(tf.logical_and(tf.less(square_error, 0.2**2), tf.greater(classes, 0))), [-1])
            elif settings['rezoom_change_loss'] == 'iou':
                iou = train_utils.iou(train_utils.to_x1y1x2y2(tf.reshape(pred_boxes, [-1, 4])),
                                      train_utils.to_x1y1x2y2(tf.reshape(perm_truth, [-1, 4])))
                inside = tf.reshape(tf.to_int64(tf.greater(iou, 0.5)), [-1])
            else:
                assert settings['rezoom_change_loss'] == False
                inside = tf.reshape(tf.to_int64((tf.greater(classes, 0))), [-1])
            new_confs = tf.reshape(pred_confs_deltas, [outer_size * settings['rnn_len'], settings['num_classes']])
            delta_confs_loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=new_confs, labels=inside)) / outer_size * settings['solver']['head_weights'][0] * 0.1

            pred_logits_squash = tf.reshape(new_confs,
                                            [outer_size * settings['rnn_len'], settings['num_classes']])
            pred_confidences_squash = tf.nn.softmax(pred_logits_squash)
            pred_confidences = tf.reshape(pred_confidences_squash,
                                      [outer_size, settings['rnn_len'], settings['num_classes']])
            loss = confidences_loss + boxes_loss + delta_confs_loss
            if settings['reregress']:
                delta_residual = tf.reshape(perm_truth - (pred_boxes + pred_boxes_deltas) * pred_mask,
                                            [outer_size, settings['rnn_len'], 4])
                delta_boxes_loss = (tf.reduce_sum(tf.minimum(tf.square(delta_residual), 10. ** 2)) /
                               outer_size * settings['solver']['head_weights'][1] * 0.03)
                boxes_loss = delta_boxes_loss
                loss += delta_boxes_loss
        else:
            loss = confidences_loss + boxes_loss

    return pred_boxes, pred_confidences, loss, confidences_loss, boxes_loss

def build(settings, queue):
    '''
    Build full model for training, including forward / backward passes,
    optimizers, and summary statistics.
    '''
    solver = settings["solver"]

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(solver.get('gpu', ''))

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)

    learning_rate = tf.placeholder(tf.float32)
    if solver['opt'] == 'RMS':
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                        decay=0.9, epsilon=solver['epsilon'])
    elif solver['opt'] == 'Adam':
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                        epsilon=solver['epsilon'])
    elif solver['opt'] == 'SGD':
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        raise ValueError('Unrecognized opt type')
    loss, accuracy, confidences_loss, boxes_loss = {}, {}, {}, {}
    for phase in ['train', 'test']:
        # generate predictions and losses from forward pass
        x, confidences, boxes = queue[phase].dequeue_many(settings['batch_size'])
        flags = tf.argmax(confidences, 3)

        grid_size = settings['grid_width'] * settings['grid_height']

        (pred_boxes, pred_confidences,
         loss[phase], confidences_loss[phase],
         boxes_loss[phase]) = build_forward_backward(settings, x, phase, boxes, flags)
        pred_confidences_r = tf.reshape(pred_confidences, [settings['batch_size'], grid_size, settings['rnn_len'], settings['num_classes']])
        pred_boxes_r = tf.reshape(pred_boxes, [settings['batch_size'], grid_size, settings['rnn_len'], 4])


        # Set up summary operations for tensorboard
        a = tf.equal(tf.argmax(confidences[:, :, 0, :], 2), tf.argmax(pred_confidences_r[:, :, 0, :], 2))
        accuracy[phase] = tf.reduce_mean(tf.cast(a, 'float32'), name=phase+'/accuracy')

        if phase == 'train':
            global_step = tf.Variable(0, trainable=False)

            tvars = tf.trainable_variables()
            if settings['clip_norm'] <= 0:
                grads = tf.gradients(loss['train'], tvars)
            else:
                grads, norm = tf.clip_by_global_norm(tf.gradients(loss['train'], tvars), settings['clip_norm'])
            train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)
        elif phase == 'test':
            moving_avg = tf.train.ExponentialMovingAverage(0.95)
            smooth_op = moving_avg.apply([accuracy['train'], accuracy['test'],
                                          confidences_loss['train'], boxes_loss['train'],
                                          confidences_loss['test'], boxes_loss['test'],
                                          ])

    return (config, loss, accuracy, train_op,
            smooth_op, global_step, learning_rate)
