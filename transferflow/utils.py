
import os
import json

def bounding_boxes_for_scaffold(path):
    if not os.path.isdir(path):
        raise Exception('Invalid model scaffold path: {}'.format(path))
    bounding_boxes_path = path + '/bounding_boxes.json'
    if not os.path.isfile(bounding_boxes_path):
        raise Exception('No bounding boxes found in model scaffold: {}'.format(bounding_boxes_path))
    with open(bounding_boxes_path, 'r') as f:
        bounding_boxes = json.load(f)
        for bounding_box in bounding_boxes:
            bounding_box['image_path'] = path + '/' + bounding_box['image_path']
        return bounding_boxes
