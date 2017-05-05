import os
import sys
import json
import tensorflow as tf
import random

from transferflow.utils import transfer_model_meta
from transferflow.utils import create_image_lists
from models.slim.preprocessing import preprocessing_factory
from models.slim.nets import nets_factory
from nnpack import load_labels
from nnpack.models import create_empty_model, save_model_benchmark_info

from . import DEFAULT_SETTINGS

slim = tf.contrib.slim


class SlimTrainer(object):
    def __init__(self, base_model_path, scaffold_path, **kwargs):
        self.base_model_path = base_model_path
        self.model_name = base_model_path.split('/')[-1]  # ex inception_resnet_v2
        self.scaffold_path = scaffold_path
        self.settings = DEFAULT_SETTINGS

        for key in kwargs:
            self.settings[key] = kwargs[key]
        if 'base_checkpoint_path' not in self.settings:
            self.settings['base_checkpoint_path'] = os.path.join(self.base_model_path,
                                                                 'checkpoint',
                                                                 self.model_name + '.ckpt')
        self.labels = load_labels(scaffold_path)
        self.num_classes = len(self.labels)
        self.preprocess = preprocessing_factory.get_preprocessing(self.model_name,
                                                                  is_training=True)
        self.model_definition = nets_factory.get_network_fn(self.model_name, self.num_classes,
                                                            is_training=True)

    def prepare(self):
        self.sess = tf.Session()

        self.image_dir = self.scaffold_path + '/images'
        if not os.path.isdir(self.scaffold_path + '/cache'):
            os.mkdir(self.scaffold_path + '/cache')

        self.image_lists = create_image_lists(self.image_dir, self.settings['testing_percentage'],
                                              self.settings['validation_percentage'])
        class_count = len(self.image_lists.keys())
        if class_count == 0:
            raise Exception('No valid folders of images found at ' + self.image_dir)
        if class_count == 1:
            raise Exception('Only one valid folder of images found at ' +
                            self.image_dir + ', multiple classes are needed for classification')
        if len(self.image_lists.keys()) != self.num_classes:
            raise Exception('wrong number of image classes detected')

        self._add_softmax_ids_to_labels()

    def get_init_fn(self):
        """Returns a function run by the chief worker to warm-start the training."""
        # TODO needs to not be hardcoded, need lookuptable for these scope names
        checkpoint_exclude_scopes = ["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"]
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

        return slim.assign_from_checkpoint_fn(
          os.path.join(self.base_model_path, 'checkpoint', self.model_name + '.ckpt'),
          variables_to_restore)

    def get_variables_to_train(self, tune=False):
        if tune is True:
            return None
        else:
            checkpoint_trainable_scopes = ["InceptionResnetV2/Logits",
                                           "InceptionResnetV2/AuxLogits"]
        trainable_scopes = [scope.strip() for scope in checkpoint_trainable_scopes]
        variables_to_train = []
        for scope in trainable_scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train

    def get_image_path(self, image_lists, label_name, index, image_dir, category):
        """"Returns a path to an image for a label at the given index.

        Args:
          image_lists: Dictionary of training images for each label.
          label_name: Label string we want to get an image for.
          index: Int offset of the image we want. This will be moduloed by the
          available number of images for the label, so it can be arbitrarily large.
          image_dir: Root folder string of the subfolders containing the training
          images.
          category: Name string of set to pull images from - training, testing, or
          validation.

        Returns:
          File system path string to an image that meets the requested parameters.

        """
        if label_name not in image_lists:
            tf.logging.fatal('Label does not exist %s.', label_name)
        label_lists = image_lists[label_name]
        if category not in label_lists:
            tf.logging.fatal('Category does not exist %s.', category)
        category_list = label_lists[category]
        if not category_list:
            tf.logging.fatal('Category has no images - %s.', category)
        mod_index = index % len(category_list)
        base_name = category_list[mod_index]
        sub_dir = label_lists['dir']
        full_path = os.path.join(image_dir, sub_dir, base_name)
        return full_path

    def load_batch_images(self, category, batch_size):
        image_size = self.model_definition.default_image_size

        label_list = list(self.image_lists.keys())
        images = []
        labels = []
        for i in xrange(batch_size):
            print('DOING BATCH')
            label_index = random.randrange(self.num_classes)
            label_name = label_list[label_index]
            image_index = random.randrange(65536)
            image_path = self.get_image_path(self.image_lists, label_name,
                                             image_index, self.image_dir, category)
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            image = tf.image.decode_jpeg(image_data, channels=3)
            processed_image = self.preprocess(image, image_size, image_size)
            # processed_images = tf.expand_dims(processed_image, 0)
            # images.append(processed_images)
            images.append(processed_image)
            labels.append(label_index)
        return tf.stack(images), labels

    def run(self, session, category, batch_size, image_lists,
            image_dir, tune_full_net, output_model_path):
        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)
            # Create the model, use the default arg scope to configure the batch norm parameters.
            processed_images, labels = self.load_batch_images(category, batch_size)

            # Create the model, use the default arg scope to configure the batch norm parameters.
            logits, _ = self.model_definition(processed_images)

            # Specify the loss function:
            one_hot_labels = slim.one_hot_encoding(labels, self.num_classes)
            slim.losses.softmax_cross_entropy(logits, one_hot_labels)
            total_loss = slim.losses.get_total_loss()

            # Create some summaries to visualize the training process:
            tf.summary.scalar('losses/Total Loss', total_loss)

            # Specify the optimizer and create the train op:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            train_op = slim.learning.create_train_op(total_loss=total_loss,
                                                     optimizer=optimizer,
                                                     variables_to_train=self.get_variables_to_train(
                                                         tune=tune_full_net))

            # Run the training:
            final_loss = slim.learning.train(
                train_op,
                logdir=output_model_path,
                init_fn=self.get_init_fn(),
                number_of_steps=10)
        print(final_loss)

    def train(self, output_model_path, tune_full_net=False):
        # catego
        self.run(self.sess, category='training', batch_size=1, image_lists=self.image_lists,
                 image_dir=self.image_dir, tune_full_net=tune_full_net,
                 output_model_path=output_model_path)

    def _add_softmax_ids_to_labels(self):
        i = 0
        for label_id in self.image_lists:
            if label_id not in self.labels:
                raise Exception('Label with ID {} does not appear in labels.json, bad scaffold'
                                .format(label_id))
            label = self.labels[label_id]
            label['node_id'] = i
            i += 1
