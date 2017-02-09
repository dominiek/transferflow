
_Warning: work in progress.._

# Transfer Learning for Tensorflow

* Object Detection based on [TensorBox - GoogleNet/Overfeat/Rezoom](https://github.com/TensorBox/TensorBox)
* Classification based on [Tensorflow - InceptionV3](https://www.tensorflow.org/how_tos/image_retraining/)

## Unit Tests

```bash
python test/object_detection_test.py
```
## Todo

* Fix checkpoint saving
* Improve model persistence
* Get rid of more unnecessary dependencies
* Refactor TensorBox originated codebase
* Bring in classification retrain code from TF
* Add unit tests for Classification
* Refactor
* Allow for more fine grained training controls + unit tests
* Experiment with different base models
