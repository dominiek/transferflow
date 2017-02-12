
_Warning: work in progress.._

# Transfer Learning for Tensorflow

* Object Detection based on [TensorBox - GoogleNet/Overfeat/Rezoom](https://github.com/TensorBox/TensorBox)
* Classification based on [Tensorflow - InceptionV3](https://www.tensorflow.org/how_tos/image_retraining/)

## Setup

```bash
pip -r requirements.txt
cd transferflow/object_detection/utils; make; cd ../../../
```

## Unit Tests

```bash
python test/object_detection_test.py
```
## Todo

* Refactor TensorBox originated codebase
* Improve model persistence
* Untangle rectangle stitching and results from the image rendering
* Make threads shut down gracefully
* Bring in classification retrain code from TF
* Add unit tests for Classification
* Refactor
* Allow for more fine grained training controls + unit tests
* Experiment with different base models
