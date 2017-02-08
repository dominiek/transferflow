
import os
import sys
import unittest
import time
test_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(test_dir + '/../')
from transferflow.object_detection import trainer
from transferflow.utils import *

class ObjectDetectionTrainTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_train_faces(self):
        bounding_boxes = bounding_boxes_for_scaffold(test_dir + '/fixtures/scaffolds/faces')
        train_bounding_boxes = bounding_boxes[0:180]
        test_bounding_boxes = bounding_boxes[180:]
        trainer.train(train_bounding_boxes, test_bounding_boxes)

if __name__ == "__main__":
    unittest.main()
