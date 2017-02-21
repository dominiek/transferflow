
import os
import sys
import unittest
import time
test_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(test_dir + '/../')
import tensorflow as tf
from scipy import misc

import logging
logger = logging.getLogger("transferflow")
logger.setLevel(logging.DEBUG)

from transferflow.classification.trainer import Trainer
from transferflow.classification.runner import Runner
from transferflow.utils import *
from transferflow.models import validate_model
from transferflow.scaffolds import clear_scaffold_cache

class ClassificationTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_2_train_scene_type(self):
        scaffold_dir = test_dir + '/fixtures/scaffolds/scene_type'
        output_model_path = test_dir + '/fixtures/tmp/scene_type_test'
        trainer = Trainer(scaffold_dir, num_steps=100)
        clear_scaffold_cache(scaffold_dir)
        trainer.prepare()
        benchmark_info = trainer.train(output_model_path)
        self.assertEqual(benchmark_info['test_accuracy'] >= 0.80, True)
        validate_model(output_model_path)

    def test_3_run_scene_type(self):
        runner = Runner(test_dir + '/fixtures/tmp/scene_type_test')
        labels = runner.run(test_dir + '/fixtures/images/lake.jpg')
        self.assertEqual(labels[0]['node_id'], 1)

if __name__ == "__main__":
    unittest.main()
