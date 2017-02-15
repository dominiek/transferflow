
import os
import json
import numpy as np
import cv2

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

def draw_rectangles(orig_image, rects, min_confidence=0.1, color=(0, 0, 255)):
    image = np.copy(orig_image)
    for rect in rects:
        if rect.confidence > min_confidence:
            cv2.rectangle(image,
                (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),
                (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),
                color,
                2)
    return image
