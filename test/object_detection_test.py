
import os
import sys
import unittest
import time
test_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(test_dir + '/../')
import tensorflow as tf
from scipy import misc
from transferflow.object_detection import trainer
from transferflow.object_detection.runner import Runner
from transferflow.utils import *
import logging
logger = logging.getLogger("transferflow")
logger.setLevel(logging.DEBUG)

class ObjectDetectionTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_1_train_faces(self):
        bounding_boxes = bounding_boxes_for_scaffold(test_dir + '/fixtures/scaffolds/faces')
        train_bounding_boxes = bounding_boxes[0:180]
        test_bounding_boxes = bounding_boxes[180:]
        trainer.train(train_bounding_boxes, test_bounding_boxes, test_dir + '/fixtures/tmp/faces_test.chkpt', {'num_steps': 1000})

    def test_2_run_faces(self):
        runner = Runner(test_dir + '/fixtures/tmp/faces_test.chkpt-1000')
        new_img, rects = runner.run(test_dir + '/fixtures/images/faces1.png')
	print('num bounding boxes: {}'.format(len(rects)))
        self.assertEqual(len(rects) >= 12, True)
        self.assertEqual(len(rects) <= 20, True)
        misc.imsave(test_dir + '/fixtures/tmp/faces_validation1.png', new_img)

    def test_3_run_faces(self):
        runner = Runner(test_dir + '/fixtures/models/faces_test.chkpt-1000')
        new_img, rects = runner.run(test_dir + '/fixtures/images/faces2.png')
        misc.imsave(test_dir + '/fixtures/tmp/faces_validation2.png', new_img)
        self.assertEqual(len(rects), 12)

    def test_4_train_faces_lstm(self):
        bounding_boxes = bounding_boxes_for_scaffold(test_dir + '/fixtures/scaffolds/faces')
        train_bounding_boxes = bounding_boxes[0:180]
        test_bounding_boxes = bounding_boxes[180:]
        trainer.train(train_bounding_boxes, test_bounding_boxes, test_dir + '/fixtures/tmp/faces_lstm_test.chkpt', {'num_steps': 1000, 'use_lstm': True, 'rnn_len': 5})

    def test_5_run_faces_lstm(self):
        runner = Runner(test_dir + '/fixtures/tmp/faces_lstm_test.chkpt-1000', {'use_lstm': True, 'rnn_len': 5})
        new_img, rects = runner.run(test_dir + '/fixtures/images/faces1.png')
	print('num bounding boxes: {}'.format(len(rects)))
        misc.imsave(test_dir + '/fixtures/tmp/faces_validation1_lstm.png', new_img)
        self.assertEqual(len(rects) >= 12, True)
        self.assertEqual(len(rects) <= 200, True)

    def test_6_train_faces_resnet(self):
        bounding_boxes = bounding_boxes_for_scaffold(test_dir + '/fixtures/scaffolds/faces')
        train_bounding_boxes = bounding_boxes[0:180]
        test_bounding_boxes = bounding_boxes[180:]
        options = {
            'num_steps': 1000,
            'slim_top_lname': 'resnet_v1_101/block4',
            'slim_attention_lname': 'resnet_v1_101/block1',
            'slim_basename': 'resnet_v1_101',
            'slim_ckpt': test_dir + '/../models/resnet_v1_101/model.ckpt'
        }
        trainer.train(train_bounding_boxes, test_bounding_boxes, test_dir + '/fixtures/tmp/faces_resnet_test.chkpt', options)

    def test_7_run_faces_resnet(self):
        runner = Runner(test_dir + '/fixtures/tmp/faces_resnet_test.chkpt-1000')
        new_img, rects = runner.run(test_dir + '/fixtures/images/faces1.png')
        misc.imsave(test_dir + '/fixtures/tmp/faces_resnet_validation1.png', new_img)
        self.assertEqual(len(rects) >= 15, True)
        self.assertEqual(len(rects) <= 20, True)

if __name__ == "__main__":
    unittest.main()
