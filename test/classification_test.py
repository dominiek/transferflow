
import os
import sys
import unittest
import time
test_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(test_dir + '/../')
import tensorflow as tf
from scipy import misc
from transferflow.classification import trainer
from transferflow.classification.runner import Runner
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

    def test_3_run_scene_type(self):
        runner = Runner(test_dir + '/fixtures/tmp/scene_type_test')
        labels = runner.run(test_dir + '/fixtures/images/lake.jpg')
        self.assertEqual(labels[0]['node_id'], 1)

if __name__ == "__main__":
    unittest.main()
