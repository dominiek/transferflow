
import os
import sys
import unittest
import time
test_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(test_dir + '/../')
import tensorflow as tf
from scipy import misc
from transferflow.object_detection.trainer import Trainer
from transferflow.object_detection.runner import Runner
from transferflow.utils import *
from transferflow.object_detection.utils.rect import Rect
from scipy.misc import imread
from nnpack.models import validate_model

import logging
logger = logging.getLogger("transferflow")
logger.setLevel(logging.DEBUG)

if not os.path.isdir(test_dir + '/fixtures/tmp'):
    os.mkdir(test_dir + '/fixtures/tmp')

base_models_dir = test_dir + '/../models'

class ObjectDetectionTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_1_run_faces(self):
        runner = Runner(test_dir + '/fixtures/models/faces_test')
        resized_img, rects, raw_rects = runner.run(test_dir + '/fixtures/images/faces2.png')
        print('num bounding boxes: {}'.format(len(rects)))
        print('num raw bounding boxes: {}'.format(len(raw_rects)))
        new_img = draw_rectangles(resized_img, raw_rects, color=(255, 0, 0))
        new_img = draw_rectangles(new_img, rects, color=(0, 255, 0))
        misc.imsave(test_dir + '/fixtures/tmp/faces_validation2.png', new_img)
        self.assertEqual(len(rects), 16)

    def test_2_train_faces(self):
        base_model_path = base_models_dir + '/inception_v1'
        trainer = Trainer(base_model_path, test_dir + '/fixtures/scaffolds/faces', num_steps=10000)
        trainer.prepare()
        benchmark_info = trainer.train(test_dir + '/fixtures/tmp/faces_test')
        validate_model(test_dir + '/fixtures/tmp/faces_test')
        self.assertEqual(benchmark_info['test_accuracy'] >= 0.80, True)

    def test_3_run_faces(self):
        runner = Runner(test_dir + '/fixtures/tmp/faces_test')
        resized_img, rects, raw_rects = runner.run(test_dir + '/fixtures/images/faces1.png')
        print('num bounding boxes: {}'.format(len(rects)))
        print('num raw bounding boxes: {}'.format(len(raw_rects)))
        new_img = draw_rectangles(resized_img, raw_rects, color=(255, 0, 0))
        new_img = draw_rectangles(new_img, rects, color=(0, 255, 0))
        misc.imsave(test_dir + '/fixtures/tmp/faces_validation1.png', new_img)
        self.assertEqual(len(rects) >= 12, True)
        self.assertEqual(len(rects) <= 20, True)

    def test_4_train_faces_lstm(self):
        base_model_path = base_models_dir + '/inception_v1'
        trainer = Trainer(base_model_path, test_dir + '/fixtures/scaffolds/faces', num_steps=1000, use_lstm=True, rnn_len=5)
        trainer.prepare()
        trainer.train(test_dir + '/fixtures/tmp/faces_lstm_test')

    def test_5_run_faces_lstm(self):
        runner = Runner(test_dir + '/fixtures/tmp/faces_lstm_test', {'use_lstm': True, 'rnn_len': 5})
        resized_img, rects, raw_rects = runner.run(test_dir + '/fixtures/images/faces1.png')
        new_img = draw_rectangles(resized_img, rects, color=(255, 0, 0))
        new_img = draw_rectangles(new_img, raw_rects, color=(0, 255, 0))
        print('num bounding boxes: {}'.format(len(rects)))
        misc.imsave(test_dir + '/fixtures/tmp/faces_lstm_validation1.png', new_img)
        self.assertEqual(len(rects) >= 12, True)
        self.assertEqual(len(rects) <= 200, True)

    def test_6_train_faces_resnet(self):
        options = {
            'num_steps': 1000,
            'base_top_layer_name': 'resnet_v1_101/block4',
            'base_attention_layer_name': 'resnet_v1_101/block1',
            'base_name': 'resnet_v1_101'
        }
        base_model_path = base_models_dir + '/resnet_v1_101'
        trainer = Trainer(base_model_path, test_dir + '/fixtures/scaffolds/faces', **options)
        trainer.prepare()
        trainer.train(test_dir + '/fixtures/tmp/faces_resnet_test')

    def test_7_run_faces_resnet(self):
        options = {
            'base_top_layer_name': 'resnet_v1_101/block4',
            'base_attention_layer_name': 'resnet_v1_101/block1',
            'base_name': 'resnet_v1_101'
        }
        runner = Runner(test_dir + '/fixtures/tmp/faces_resnet_test', options)
        resized_img, rects, raw_rects = runner.run(test_dir + '/fixtures/images/faces1.png')
        new_img = draw_rectangles(resized_img, rects, color=(255, 0, 0))
        new_img = draw_rectangles(new_img, raw_rects, color=(0, 255, 0))
        print('num bounding boxes: {}'.format(len(rects)))
        misc.imsave(test_dir + '/fixtures/tmp/faces_resnet_validation1.png', new_img)
        self.assertEqual(len(rects) >= 12, True)
        self.assertEqual(len(rects) <= 200, True)

    def test_8_train_parking_lots(self):
        base_model_path = base_models_dir + '/inception_v1'
        trainer = Trainer(base_model_path, test_dir + '/fixtures/scaffolds/parking_lots', num_steps=1000)
        trainer.prepare()
        benchmark_info = trainer.train(test_dir + '/fixtures/tmp/parking_lots_test')
        validate_model(test_dir + '/fixtures/tmp/parking_lots_test')
        self.assertEqual(benchmark_info['test_accuracy'] >= 0.80, True)

    def test_9_run_parking_lots(self):
        runner = Runner(test_dir + '/fixtures/tmp/parking_lots_test')
        expected_lots = {
            '1': [424, 306],
            '2': [145, 496],
            '3': [428, 735],
            '4': [715, 668]
        }
        for i in range(1, 5):
            image_path = test_dir + '/fixtures/images/parking_lots/{}.png'.format(i)
            resized_img, rects, raw_rects = runner.run(image_path.format(i), adjust_rects=False)
            new_img = draw_rectangles(resized_img, raw_rects, color=(255, 0, 0))
            new_img = draw_rectangles(new_img, rects, color=(0, 255, 0))
            misc.imsave(test_dir + '/fixtures/tmp/parking_lots_validation{}.png'.format(i), new_img)
            resized_img, rects, raw_rects = runner.run(image_path.format(i), adjust_rects=True)
            orig_img_arr = imread(image_path)[:,:,:3]
            new_img = draw_rectangles(orig_img_arr, raw_rects, color=(255, 0, 0))
            new_img = draw_rectangles(new_img, rects, color=(0, 255, 0))
            misc.imsave(test_dir + '/fixtures/tmp/parking_lots_validation_adjusted{}.png'.format(i), new_img)
            self.assertEqual(len(rects) >= 1, True)
            expected_lot = expected_lots[str(i)]
            expected_rect = Rect(expected_lot[0], expected_lot[1], 10, 10, 1)
            found_lot = False
            for rect in rects:
                if rect.overlaps(expected_rect):
                    found_lot = True
            self.assertEqual(found_lot, True, 'Excpected lot for image {}'.format(i))

if __name__ == "__main__":
    unittest.main()
