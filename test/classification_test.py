
import os
import sys
import unittest

test_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(test_dir + '/../')

from transferflow.classification.trainer import Trainer
from transferflow.classification.runner import Runner
from transferflow.classification.tfslim_runner import SlimRunner
from transferflow.classification.tfslim_trainer import SlimTrainer
from transferflow.utils import *
from nnpack.models import validate_model
from nnpack.scaffolds import clear_scaffold_cache

import logging
logger = logging.getLogger("transferflow")
logger.setLevel(logging.DEBUG)

if not os.path.isdir(test_dir + '/fixtures/tmp'):
    os.mkdir(test_dir + '/fixtures/tmp')

base_models_dir = test_dir + '/../models'


class ClassificationTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_1_run_base_model(self):
        runner = Runner(base_models_dir + '/inception_v3', softmax_layer='softmax:0')
        labels = runner.run(test_dir + '/fixtures/images/lake.jpg')
        self.assertEqual(labels[0]['name'], 'boathouse')

    def test_2_train_scene_type(self):
        scaffold_dir = test_dir + '/fixtures/scaffolds/scene_type'
        output_model_path = test_dir + '/fixtures/tmp/scene_type_test'
        base_model_path = base_models_dir + '/inception_v3'
        trainer = Trainer(base_model_path, scaffold_dir, num_steps=100)
        clear_scaffold_cache(scaffold_dir)
        trainer.prepare()
        benchmark_info = trainer.train(output_model_path)
        self.assertEqual(benchmark_info['test_accuracy'] >= 0.80, True)
        validate_model(output_model_path)

    def test_3_run_scene_type(self):
        runner = Runner(test_dir + '/fixtures/tmp/scene_type_test')
        labels = runner.run(test_dir + '/fixtures/images/lake.jpg')
        self.assertEqual(labels[0]['node_id'], 1)
        self.assertEqual(labels[0]['name'], 'Outdoor')

    def test_4_run_inception_resnet_v2_base_model(self):
        runner = SlimRunner(base_models_dir + '/inception_resnet_v2', 'inception_resnet_v2')
        labels = runner.run(test_dir + '/fixtures/images/lake.jpg')
        self.assertEqual(labels[0]['name'], 'boathouse')

    def test_5_train_scene_type_resnet_v2(self):
        scaffold_dir = test_dir + '/fixtures/scaffolds/scene_type'
        output_model_path = test_dir + '/fixtures/tmp/scene_type_test_resnetv2'
        base_model_path = base_models_dir + '/inception_resnet_v2'
        trainer = SlimTrainer(base_model_path, scaffold_dir, num_steps=20, batch_size=32)
        clear_scaffold_cache(scaffold_dir)
        trainer.prepare()
        benchmark_info = trainer.train(output_model_path)
        self.assertEqual(benchmark_info['final_loss'] <= 0.80, True)
        validate_model(output_model_path)

    def test_6_run_scene_type_resnet_v2(self):
        runner = SlimRunner(test_dir + '/fixtures/tmp/scene_type_test_resnetv2',
                            'inception_resnet_v2')
        labels = runner.run(test_dir + '/fixtures/images/lake.jpg')
        self.assertEqual(labels[0]['node_id'], 1)
        self.assertEqual(labels[0]['name'], 'Outdoor')

    def test_7_train_coffee_roasts(self):
        scaffold_dir = test_dir + '/fixtures/scaffolds/coffee_roasts'
        output_model_path = test_dir + '/fixtures/tmp/coffee_roasts_test'
        base_model_path = base_models_dir + '/inception_v3'
        trainer = Trainer(base_model_path, scaffold_dir, num_steps=100)
        clear_scaffold_cache(scaffold_dir)
        trainer.prepare()
        benchmark_info = trainer.train(output_model_path)
        self.assertEqual(benchmark_info['test_accuracy'] >= 0.75, True)
        validate_model(output_model_path)

if __name__ == "__main__":
    unittest.main()
