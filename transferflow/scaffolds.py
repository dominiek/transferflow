
import os
import json
import shutil
from utils import load_labels, load_meta

class InvalidScaffoldError(Exception):
    pass

def validate_scaffold(path):
    if not os.path.isdir(path):
        raise InvalidScaffoldError('Invalid Scaffold, should be a directory: {}'.format(path))
    if not os.path.isfile(path + '/index.json'):
        raise InvalidScaffoldError('Invalid Scaffold, expected index.json')
    try:
        meta = load_meta(path)
    except Exception as e:
        raise InvalidScaffoldError('Could not load Scaffold meta data: {}'.format(str(e)))
    if not meta.has_key('id'):
        raise InvalidScaffoldError('Invalid Scaffold meta data, expected field `id` in index.json')
    if not meta.has_key('name'):
        raise InvalidScaffoldError('Invalid Scaffold meta data, expected field `name` in index.json')
    if not os.path.isfile(path + '/labels.jsons'):
        raise InvalidScaffoldError('Invalid Scaffold, expected labels.jsons')
    try:
        labels = load_labels(path)
    except Exception as e:
        raise InvalidScaffoldError('Could not load Scaffold labels: {}'.format(str(e)))
    for id in labels:
        if not labels[id].has_key('name'):
            raise InvalidScaffoldError('Missing name attribute for label {}'.format(id))
    if len(labels) < 1:
        raise InvalidScaffoldError('Expected at least 1 label in Scaffold')

def clear_scaffold_cache(scaffold_path):
    if os.path.isdir(scaffold_path + '/cache'):
        shutil.rmtree(scaffold_path + '/cache')

def bounding_boxes_for_scaffold(path):
    if not os.path.isdir(path):
        raise Exception('Invalid model scaffold path: {}'.format(path))
    labels = load_labels(path)
    all_bounding_boxes = []
    for id in labels:
        bounding_boxes_path = path + '/images/' + id + '/bounding_boxes.json'
        if not os.path.isfile(bounding_boxes_path):
            raise Exception('No bounding boxes found for label {}: {}'.format(id, bounding_boxes_path))
        with open(bounding_boxes_path, 'r') as f:
            bounding_boxes = json.load(f)
            for bounding_box in bounding_boxes:
                bounding_box['image_path'] = path + '/images/' + id + '/' + bounding_box['image_path']
                bounding_box['label'] = labels[id]
            all_bounding_boxes = all_bounding_boxes + bounding_boxes
    return bounding_boxes
