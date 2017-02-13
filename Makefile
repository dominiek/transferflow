

all:
	cd transferflow/object_detection/utils; make; make hungarian

.PHONY: download
download:
	cd models; \
		wget --continue http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz; \
		rm -rf resnet_v1_101; \
		tar xfzv resnet_v1_101_2016_08_28.tar.gz; \
		rm -f resnet_v1_101_2016_08_28.tar.gz; \
		mkdir resnet_v1_101; \
		mv resnet_v1_101.ckpt resnet_v1_101/model.ckpt
