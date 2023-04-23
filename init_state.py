import tensorflow as tf
tf.random.set_seed(42)
from tfdeterminism import patch
import numpy as np
import os

np.random.seed(42)
seed = 42
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def rnn_init_state(init_, batch_size, num_layers, num_units, rnn_network=None, initial_stddev=0.02):

    if init_ == "zero":
        initial_state = rnn_network.zero_state(batch_size, tf.float32)

    elif init_ == "random":
        initial_state = tf.random_normal(shape=(num_layers, 2, batch_size, num_units),
                                         mean=0.0, stddev=1.0, seed = seed)
        initial_state = tf.unstack(initial_state, axis=0)
        initial_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(initial_state[idx][0], initial_state[idx][1]) \
                               for idx in range(num_layers)])

    elif init_ == "variable":
        initial_state = []
        with tf.compat.v1.variable_scope("Discrete_discriminator",  reuse=tf.AUTO_REUSE):
            for i in range(num_layers):
                sub_initial_state1 = tf.get_variable("layer{}_initial_state1".format(i), (1, num_units),
                                                     initializer=tf.random_normal_initializer(stddev=initial_stddev, seed = seed),use_resource= False)
                sub_initial_state1 = tf.tile(sub_initial_state1, (batch_size, 1))
                sub_initial_state2 = tf.get_variable("layer{}_initial_state2".format(i), (1, num_units),
                                                    initializer=tf.random_normal_initializer(stddev=initial_stddev, seed = seed),use_resource= False)
                sub_initial_state2 = tf.tile(sub_initial_state2, (batch_size, 1))
                sub_initial_state = tf.nn.rnn_cell.LSTMStateTuple(sub_initial_state1, sub_initial_state2)
                initial_state.append(sub_initial_state)
            initial_state = tuple(initial_state)

    return initial_state
