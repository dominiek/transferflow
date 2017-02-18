

all:
	cd transferflow/object_detection/utils; make; make hungarian

.PHONY: download
download: download.resnet download.inception

.PHONY: download.inception_v3
download.inception_v3:
	cd models; \
		wget --continue http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz; \
		tar xfzv inception-2015-12-05.tgz; \
		mv classify_image_graph_def.pb inception_v3/model.pb; \
		mv imagenet_2012_challenge_label_map_proto.pbtxt inception_v3/model.pbtxt; \
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
		mv resnet_v1_101.ckpt resnet_v1_101/model.ckpt


.PHONY: test
test:
	python test/object_detection_test.py
