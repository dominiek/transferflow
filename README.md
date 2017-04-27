
# Transfer Learning for Tensorflow

Transfer learning is the unhidden gem in the deep learning world. It allows model creation with significantly reduced training data and time by modifying existing rich deep learning models.

The goal of this framework is to collect best-of-breed approaches, make them developer friendly so they can be used for real-world applications.

Current capabilities:

* Object Detection based on [TensorBox - GoogleNet/Overfeat/Rezoom](https://github.com/TensorBox/TensorBox)
* Classification based on [Tensorflow - InceptionV3](https://www.tensorflow.org/how_tos/image_retraining/)

_Please note that this is still under active development. See the TODO list in the bottom._

## Dependencies

* Tensorflow >= 0.12.1 (Tested on `v0.12.1`, `v1.0` and `v1.1`)
* [nnpack](https://github.com/dominiek/nnpack) >= 0.1.0 (Tools & Data Portability for Neural Nets)

## Install using Pip

```bash
pip install transferflow
```

## Install by Source (Recommended)

```bash
pip install -r requirements.txt
make
make download
```

## Example: Classification

First, your training data needs to be formatted like a Standard Scaffold. See [NNPack](https://github.com/dominiek/nnpack) for more details. In this example we'll use a pre-prepared Scaffold [test/fixtures/scaffolds/scene_type](test/fixtures/scaffolds/scene_type) that has two sets of images:

* 115 images depicting indoor scenes
* 158 images depicting outdoor scenes

We will use this to train a model for Scene Type detection (Indoor VS Outdoor).

```python
from transferflow.classification.trainer import Trainer

# Instantiate Trainer with training data, base model and configuration
scaffold_path = './test/fixtures/scaffolds/scene_type'
base_model_path = './models/inception_v3'
trainer = Trainer(base_model_path, scaffold_path, num_steps=1000)

# Prepare session (calculates Bottleneck files)
trainer.prepare()

# Train new model and save
benchmark_info = trainer.train('./scene_type_model')
```

We now have a newly created model in `./scene_type_model` which we can use as follows:

```python
from transferflow.classification.runner import Runner

# Our new model
runner = Runner('./scene_type_model')
predicted_labels = runner.run('./test/fixtures/images/lake.jpg')
print(predicted_labels)
```

This will output the predicted labels for this image ordered by score.

## Example: Object Detection

First, your training data needs to be formatted like a Standard Scaffold. See [Formats](FORMATS.md) for more details. In this example we'll use a pre-prepared Scaffold [test/fixtures/scaffolds/faces](test/fixtures/scaffolds/faces) that has the following:

* 200 images that have people in them
* 1351 annotated bounding boxes for the faces of each person found in the pictures

We'll use this to train a face object detector.

```python
from transferflow.object_detection.trainer import Trainer

# Instantiate trainer with training data and configuration
scaffold_path = './test/fixtures/scaffolds/faces'
base_model_path = './models/inception_v1'
trainer = Trainer(base_model_path, scaffold_path, num_steps=1000)

# Prepare session (splits up test/train data)
trainer.prepare()

# Train new model and save
trainer.train('/faces_model')
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

* Classification: Refactor bottleneck creation phase
* Object Detection: Make hungarian and stitch examples into proper Python modules so they jive well with Pip
* Object Detection: Clean up settings
* Object Detection: Add non-face example
* Object Detection: Allow object detection of different image sizes
* Object Detection: Create validation set facility
* Object Detection: Slim down size of detection models (lots of unused nodes in there)
* Object Detection: Improve naming
* Object Detection: Refactor TensorBox originated code
* Make threads shut down gracefully
* Experiment with different base models
