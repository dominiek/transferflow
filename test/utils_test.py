
import os
import sys
import unittest
import time
test_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(test_dir + '/../')
import tensorflow as tf
from transferflow.utils import *
import logging
logger = logging.getLogger("transferflow")
logger.setLevel(logging.DEBUG)
import time

def create_graph():
    state = tf.Variable(0, name='state')
    increment = tf.Variable(1, name='increment')
    add = tf.add(state, increment)
    update = tf.assign(state, add, name='update')
    return state, update, increment

class UtilsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_save_and_load_model_without_meta(self):
        tf.reset_default_graph()
        # Create TF graph
        state, update, increment = create_graph()

        # Create and init session
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)

        # Assert graph works as designed
        self.assertEqual(session.run(state), 0)
        session.run(update)
        self.assertEqual(session.run(state), 1)
        session.run(update, {increment: 2})
        self.assertEqual(session.run(state), 3)
        self.assertEqual(len(get_tensors(session)), 11)

        # Save state
        save_model(session, test_dir + '/fixtures/tmp/increment_without_meta')

        # Clear graph
        tf.reset_default_graph()
        session = tf.Session()
        self.assertEqual(len(get_tensors(session)), 0)

        # Restore model fully from file
        #state, update, increment = create_graph()
        state, update, increment = create_graph()
        init = tf.global_variables_initializer()
        session.run(init)
        load_model(session, test_dir + '/fixtures/tmp/increment_without_meta', exclude_meta=True)
        

        # Assert previous state
        self.assertEqual(session.run(state), 3)
        session.run(update, {increment: 3})
        self.assertEqual(session.run(state), 6)

    def test_save_and_load_model_with_meta(self):
        tf.reset_default_graph()
        # Create TF graph
        create_graph()

        # Create and init session
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)

        # Assert graph works as designed
        self.assertEqual(session.run('state:0'), 0)
        session.run('update')
        self.assertEqual(session.run('state:0'), 1)
        session.run('update', {'increment:0': 2})
        self.assertEqual(session.run('state:0'), 3)
        self.assertEqual(len(get_tensors(session)), 11)

        # Save state
        save_model(session, test_dir + '/fixtures/tmp/increment')

        # Clear graph
        tf.reset_default_graph()
        session = tf.Session()
        self.assertEqual(len(get_tensors(session)), 0)

        # Restore model fully from file
        load_model(session, test_dir + '/fixtures/tmp/increment')

        # Assert previous state
        self.assertEqual(session.run('state:0'), 3)
        session.run('update', {'increment:0': 3})
        self.assertEqual(session.run('state:0'), 6)


if __name__ == "__main__":
    unittest.main()
