
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

* Untangle rectangle stitching and results from the image rendering
* Improve model persistence
* Bring in classification retrain code from TF
* Add unit tests for Classification
* Refactor
* Get rid of Tensorflow deprecation warnings
* Upgrade Tensorflow version
* Create validation set facility
* Refactor TensorBox originated code
* Make threads shut down gracefully
* Experiment with different base models
