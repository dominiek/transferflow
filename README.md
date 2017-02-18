
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

## Unit Tests

```bash
make test
```

## Todo

* Classification: Bring in classification retrain code from TF
* Classification: Add unit tests for Classification
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
