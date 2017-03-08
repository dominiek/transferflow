all:
	cd transferflow/object_detection/utils; make; make hungarian

.PHONY: clean
clean:
	find . -iname '*.pyc' -delete
	find . -iname '*.so' -delete
	rm -rf nnpack.egg-info
	rm -rf dist
	rm -rf build

.PHONY: download
download: download.resnet download.inception_v3

.PHONY: download.inception_v3
download.inception_v3:
	cd models; \
		wget --continue http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz; \
		tar xfzv inception-2015-12-05.tgz; \
		mkdir -p inception_v3/state; \
		mv classify_image_graph_def.pb inception_v3/state/model.pb; \
		mv imagenet_2012_challenge_label_map_proto.pbtxt inception_v3/state/model.pbtxt; \
		mv LICENSE inception_v3/.; \
		rm *.jpg; rm *.txt; rm *.tgz \

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

UNIT_TEST_FILES := $(wildcard test/*_test.py)
.PHONY: test
test: $(UNIT_TEST_FILES)
	$(foreach file,$(UNIT_TEST_FILES),python $(file);)

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
	-make test
	mv transferflow_ transferflow
	pip uninstall -y transferflow
