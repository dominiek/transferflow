
import os
import sys
import unittest
import time
test_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(test_dir + '/../')
import tensorflow as tf
from scipy import misc
from transferflow.classification import trainer
from transferflow.utils import *
import logging
logger = logging.getLogger("transferflow")
logger.setLevel(logging.DEBUG)

class ClassificationTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_2_train_scene_type(self):
        scaffold_dir = test_dir + '/fixtures/scaffolds/scene_type'
        base_graph_path = test_dir + '/../models/inception_v3/model.pb'
        output_model_path = test_dir + '/fixtures/tmp/scene_type_test'
        trainer.train(scaffold_dir, base_graph_path, output_model_path, {'num_steps': 100})

    """
    def test_3_run_faces(self):
        runner = Runner(test_dir + '/fixtures/tmp/faces_test')
        resized_img, rects, raw_rects = runner.run(test_dir + '/fixtures/images/faces1.png')
        print('num bounding boxes: {}'.format(len(rects)))
        print('num raw bounding boxes: {}'.format(len(raw_rects)))
        new_img = draw_rectangles(resized_img, rects, color=(255, 0, 0))
        new_img = draw_rectangles(new_img, raw_rects, color=(0, 255, 0))
        misc.imsave(test_dir + '/fixtures/tmp/faces_validation1.png', new_img)
        self.assertEqual(len(rects) >= 12, True)
        self.assertEqual(len(rects) <= 20, True)
    """

if __name__ == "__main__":
    unittest.main()
