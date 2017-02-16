
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

* Improve model persistence: naming
* Improve model persistence: slim down model (only save what's needed)
* Bring in classification retrain code from TF
* Add unit tests for Classification
* Refactor
* Get rid of Tensorflow deprecation warnings
* Upgrade Tensorflow version
* Add non-face example
* Allow object detection of different image sizes
* Create validation set facility
* Refactor TensorBox originated code
* Make threads shut down gracefully
* Experiment with different base models
