all:
	cd transferflow/object_detection/utils; make; make hungarian

.PHONY: clean
clean:
	find . -iname '*.pyc' -delete
	find . -iname '*.so' -delete
	rm -rf nnpack.egg-info
	rm -rf dist
	rm -rf build
	rm -rf ve

.PHONY: download
download: download.resnet download.inception_v3 download.inception_resnet_v2 download.slim

.PHONY: download.slim
download.slim:
	cd models; \
		wget https://github.com/tensorflow/models/archive/f94f163726be25045ef86aebe17f69ca7c2703b9.zip -O slim_models.zip; \
		unzip slim_models.zip; \
		rm -f slim_models.zip; \
		mv models-f94f163726be25045ef86aebe17f69ca7c2703b9/slim .; \
		rm -rf models-f94f163726be25045ef86aebe17f69ca7c2703b9;

.PHONY: download.inception_resnet_v2
download.inception_resnet_v2:
	cd models; \
		wget --continue http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz; \
		tar xfzv inception_resnet_v2_2016_08_30.tar.gz; \
		rm -f inception_resnet_v2_2016_08_30.tar.gz; \
		mkdir -p inception_resnet_v2/state; \
		mv inception_resnet_v2_2016_08_30.ckpt inception_resnet_v2/state/inception_resnet_v2.ckpt;

.PHONY: download.inception_v3
download.inception_v3:
	cd models; \
		wget --continue http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz; \
		tar xfzv inception-2015-12-05.tgz; \
		mkdir -p inception_v3/state; \
		mv classify_image_graph_def.pb inception_v3/state/model.pb; \
		mv imagenet_2012_challenge_label_map_proto.pbtxt inception_v3/state/model.pbtxt; \
		mv LICENSE inception_v3/.; \
		rm *.jpg; rm *.txt; rm *.tgz

.PHONY: download.resnet
download.resnet:
	cd models; \
		wget --continue http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz; \
		rm -rf resnet_v1_101; \
		tar xfzv resnet_v1_101_2016_08_28.tar.gz; \
		rm -f resnet_v1_101_2016_08_28.tar.gz; \
		mkdir resnet_v1_101; \
		mkdir -p resnet_v1_101/state; \
		mv resnet_v1_101.ckpt resnet_v1_101/state/model.ckpt

.PHONY: test
test: ve
	. ve/bin/activate && find test -name '*_test.py' | xargs -n 1 python

.PHONY: test.classification
test.classification: ve
	. ve/bin/activate && find test -name 'classification_test.py' | xargs -n 1 python

.PHONY: package.build
package.build:
	make clean
	python setup.py sdist
	python setup.py bdist_wheel

.PHONY: package.release
package.release:
	twine upload dist/*

.PHONY: package.test
package.test:
	-pip uninstall -y transferflow
	pip install --no-cache-dir transferflow
	mv transferflow transferflow_
	-make test.classification
	mv transferflow_ transferflow
	pip uninstall -y transferflow

ve:
	virtualenv ve
	. ./ve/bin/activate && pip install -r requirements.txt

repl: ve
	. ve/bin/activate && python
