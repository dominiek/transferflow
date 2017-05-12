import os
import sys
import tensorflow as tf
import json

from transferflow.utils import transfer_model_meta
from transferflow.utils import create_image_lists
sys.path.append(os.path.abspath('models/slim'))
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
            checkpoint_dir = os.path.abspath(os.path.join(self.base_model_path, 'state'))
            if tf.train.latest_checkpoint(checkpoint_dir) is None:
                base_path = os.path.join(self.base_model_path, 'state', self.model_name + '.ckpt')
            else:
                base_path = tf.train.latest_checkpoint(checkpoint_dir)
            self.settings['base_checkpoint_path'] = base_path

        self.labels = load_labels(scaffold_path)
        self.num_classes = len(self.labels)
        self.preprocess = preprocessing_factory.get_preprocessing(self.model_name,
                                                                  is_training=True)
        self.model_definition = nets_factory.get_network_fn(self.model_name, self.num_classes,
                                                            is_training=True)

    def _add_softmax_ids_to_labels(self):
        i = 0
        for label_id in self.image_lists:
            if label_id not in self.labels:
                raise Exception('Label with ID {} does not appear in labels.json, bad scaffold'
                                .format(label_id))
            label = self.labels[label_id]
            label['node_id'] = i
            i += 1

    def prepare(self):
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

    @staticmethod
    def read_labeled_image_list(image_dir, image_lists, labels, category='training'):
        filenames = []
        training_labels = []
        for label in image_lists:
            for base_name in image_lists[label][category]:
                full_path = os.path.join(image_dir, image_lists[label]['dir'], base_name)
                filenames.append(full_path)
                training_labels.append(labels[label]['node_id'])
        return filenames, training_labels

    @staticmethod
    def load_image_from_queue(input_queue):
        """Consumes a single filename and label as a ' '-delimited string.
        Args:
          filename_and_label_tensor: A scalar string tensor.
        Returns:
          Two tensors: the decoded image, and the string label.
        """
        label = input_queue[1]
        file_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(file_contents, channels=3)
        return image, label

    def load_batch_images(self, category='training'):
        batch_size = self.settings['batch_size']
        image_size = self.model_definition.default_image_size
        num_preprocess_threads = 2
        min_queue_examples = 64

        filenames, labels = self.read_labeled_image_list(image_dir=self.image_dir,
                                                         image_lists=self.image_lists,
                                                         labels=self.labels,
                                                         category=category)
        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        # Makes an input queue
        input_queue = tf.train.slice_input_producer([filenames, labels],
                                                    shuffle=True)
        image, label = self.load_image_from_queue(input_queue)
        processed_image = self.preprocess(image, image_size, image_size)
        # processed_images = tf.expand_dims(processed_image, 0)
        images, labels = tf.train.shuffle_batch([processed_image, label],
                                                batch_size=batch_size,
                                                num_threads=num_preprocess_threads,
                                                capacity=min_queue_examples + 3 * batch_size,
                                                min_after_dequeue=min_queue_examples)
        return images, labels

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

        return slim.assign_from_checkpoint_fn(self.settings['base_checkpoint_path'],
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

    def run(self, category, tune_full_net, output_model_path):
        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)
            # Create the model, use the default arg scope to configure the batch norm parameters.
            processed_images, labels = self.load_batch_images(category)

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
                init_fn=self.get_init_fn(),
                logdir=os.path.join(output_model_path, 'state'),
                number_of_steps=self.settings['num_steps'])
        return final_loss

    def train(self, output_model_path, tune_full_net=False):
        create_empty_model(output_model_path)
        transfer_model_meta(self.scaffold_path, output_model_path)

        final_loss = self.run(category='training', tune_full_net=tune_full_net,
                              output_model_path=output_model_path)
        benchmark_info = {
            'final_loss': float(final_loss),
        }

        # Persist labels with softmax IDs
        with open(output_model_path + '/labels.json', 'w') as f:
            json.dump({'labels': self.labels.values()}, f)

        # Cleanup
        tf.reset_default_graph()

        return benchmark_info


