
_Warning: work in progress..._

# Transfer Learning for Tensorflow

* Object Detection based on [TensorBox - GoogleNet/Overfeat/Rezoom](https://github.com/TensorBox/TensorBox)
* Classification based on [Tensorflow - InceptionV3](https://www.tensorflow.org/how_tos/image_retraining/)

## Setup

```bash
pip -r requirements.txt
make
make download
```

## Example: Classification Transfer Learning

First, your training data needs to be formatted like a Standard Scaffold. See [Formats](FORMATS.md) for more details. In this example we'll use a pre-prepared Scaffold [test/fixtures/scaffolds/scene_type](test/fixtures/scaffolds/scene_type) that has two sets of images:

* 115 images depicting indoor scenes
* 158 images depicting outdoor scenes

We will use this to train a model for Scene Type detection (Indoor VS Outdoor).

```python
from transferflow.classification import trainer

# Our training data
scaffold_path = './test/fixtures/scaffolds/scene_type'

# The base Inception model that we'll use for training
base_graph_path = './models/inception_v3/model.pb'

# Output model path
output_model_path = './scene_type_model'

# Run training sequence for 1000 iterations
_, benchmark_info = trainer.train(scaffold_path, base_graph_path, output_model_path, {'num_steps': 1000})
```

We now have a newly created model in `./scene_type_model` which we can use as follows:

```python
from transferflow.classification.runner import Runner

# Our new model
runner = Runner('./scene_type_model')
predicted_labels = runner.run('./fixtures/images/lake.jpg')
print(predicted_labels)
```

This will output the predicted labels for this image ordered by score.

## Example: Object Detection Transfer Learning

First, your training data needs to be formatted like a Standard Scaffold. See [Formats](FORMATS.md) for more details. In this example we'll use a pre-prepared Scaffold [test/fixtures/scaffolds/faces](test/fixtures/scaffolds/faces) that has the following:

* 200 images that have people in them
* 1351 annotated bounding boxes for the faces of each person found in the pictures

We'll use this to train a face object detector.

```python
from transferflow.object_detection import trainer

# Our training data
scaffold_path = './test/fixtures/scaffolds/faces'

# Get bounding boxes, and specify train/test ratio
bounding_boxes = bounding_boxes_for_scaffold(test_dir + '/fixtures/scaffolds/faces')
train_bounding_boxes = bounding_boxes[0:180]
test_bounding_boxes = bounding_boxes[180:]

# Output model path
output_model_path = './faces_model'

test_dir + '/fixtures/tmp/faces_test', {'num_steps': 1000}

# Run training sequence for 1000 iterations
trainer.train(train_bounding_boxes, test_bounding_boxes, output_model_path, {'num_steps': 1000})
```

Now our object detection model will be stored in `./faces_model`. It can be used like so:

```python
from transferflow.object_detection.runner import Runner
from transferflow.utils import draw_rectangles

# Our new model
runner = Runner('./faces_model')

# Run, gets back predicted bounding boxes and a resized image
resized_img, rects, raw_rects = runner.run('./test/fixtures/images/faces1.png')

# Draw all predicted bounding box rectangles on the resized image and store
new_img = draw_rectangles(resized_img, rects, color=(255, 0, 0))
new_img = draw_rectangles(new_img, raw_rects, color=(0, 255, 0))
misc.imsave('./faces_validation1.png', new_image)
```

## Unit Tests

```bash
make test
```

## Todo

* Classification: Refactor
* Get rid of Tensorflow deprecation warnings
* Upgrade Tensorflow version
* Object Detection: Clean up settings
* Object Detection: Add non-face example
* Lot's of documentation
* Object Detection: Allow object detection of different image sizes
* Object Detection: Create validation set facility
* Object Detection: Slim down size of detection models (lots of unused nodes in there)
* Object Detection: Improve naming
* Object Detection: Refactor TensorBox originated code
* Make threads shut down gracefully
* Experiment with different base models
