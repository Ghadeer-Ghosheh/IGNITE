import tensorflow as tf2
#update
import warnings
warnings.filterwarnings("ignore")
tf2.random.set_seed(42)
from tfdeterminism import patch
import os 
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow.compat.v1 as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.disable_v2_behavior()
seed = 4
#from tensorflow.python import keras
from tensorflow.python.keras.regularizers import L2 as l2_regularizer
from init_state import rnn_init_state
class observed_only_vae(object):
    def __init__(self,
                  time_steps,
                 dim, z_dim,
                 enc_size=256, dec_size=256,
                 enc_layers=3, dec_layers=3,
                 keep_prob=1, l2scale=0.0001,
                 conditional=True, num_labels=0):

        self.time_steps = time_steps
      

        self.dim = dim
        self.z_dim = z_dim
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.keep_prob = keep_prob
        self.l2scale = l2scale
        self.conditional = conditional
        self.num_labels = num_labels
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers

    def build_vae(self, input_data, batch_size, conditions=None):
        tf2.random.set_seed(42)
        if self.conditional:
            assert not self.num_labels == 0
            # changed size
            repeated_encoding= conditions
            input_data_cond = tf.concat([input_data, repeated_encoding], axis=-1)
            input_enc = tf.unstack(input_data_cond, axis=1)
        else:
            input_enc = tf.unstack(input_data, axis=1)
 
        #self.batch_size = 200

        #batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        self.cell_enc = self.buildEncoder()
        self.cell_dec = self.buildDecoder()
        print(batch_size)
        enc_state = self.cell_enc.zero_state(batch_size, tf.float32)
        dec_state = self.cell_dec.zero_state(batch_size, tf.float32)

        self.e =tf2.random.normal((batch_size, self.z_dim))
        self.c, mu, logsigma, sigma, z = [0] * self.time_steps, [0] * self.time_steps, [0] * self.time_steps, \
                                         [0] * self.time_steps, [0] * self.time_steps 
        w_mu, b_mu, w_sigma, b_sigma, self.w_h_dec, self.b_h_dec = self.buildSampling()

        for t in range(self.time_steps):
            if t == 0:
                c_prev = tf.zeros((batch_size, self.dim))
            else:
                c_prev = self.c[t - 1]

            c_sigmoid = tf.sigmoid(c_prev)
                
            if self.conditional:
                x_hat = tf2.unstack(input_data, axis=1)[t] - c_sigmoid
            else:
                x_hat = input_enc[t] - c_sigmoid

            with tf.compat.v1.variable_scope('Encoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.compat.v1.AUTO_REUSE):
                tf2.random.set_seed(42)

                h_enc, enc_state = self.cell_enc(tf.concat([input_enc[t], x_hat], 1), enc_state)

            mu[t] = tf.matmul(h_enc, w_mu) + b_mu
            logsigma[t] = tf.matmul(h_enc, w_sigma) + b_sigma
            sigma[t] = tf.exp(logsigma[t])

            if self.conditional:
                z[t] = mu[t] + sigma[t] * self.e
                # changed shape
                z[t] = tf.concat([z[t], conditions[:,t,:]], axis=-1)
            else:
                z[t] = mu[t] + sigma[t] * self.e

            with tf.compat.v1.variable_scope('Decoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.compat.v1.AUTO_REUSE):
                tf2.random.set_seed(42)

                h_dec, dec_state = self.cell_dec(z[t], dec_state)

            self.c[t] = tf.matmul(h_dec, self.w_h_dec) + self.b_h_dec

        self.decoded = tf.stack(self.c, axis=1)

        return self.decoded, sigma, mu, logsigma, z

    def reconstruct_decoder(self, dec_input, batch_size, conditions=None):
        rec_decoded = [0] * self.time_steps
        rec_dec_state = self.cell_dec.zero_state(batch_size, dtype=tf.float32)
        for t in range(self.time_steps):
            if self.conditional:
                dec_input_with_c = tf.concat([dec_input[t], conditions[:,t,:]], axis=-1)
                rec_h_dec, rec_dec_state = self.cell_dec(dec_input_with_c, rec_dec_state)
            else:
                rec_h_dec, rec_dec_state = self.cell_dec(dec_input[t], rec_dec_state)

            rec_decoded[t] = tf.matmul(rec_h_dec, self.w_h_dec) + self.b_h_dec

        return tf.stack(rec_decoded, axis=1)

    def buildEncoder(self):
        cell_units = []
        for num_units in range(self.enc_layers-1):
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.enc_size, name="observed_only_VAE", reuse=tf.compat.v1.AUTO_REUSE)
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob,seed=seed)
            cell_units.append(cell)

        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.enc_size, name="Shared_VAE", reuse=tf.compat.v1.AUTO_REUSE)
        cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob,seed=seed)
        cell_units.append(cell)

        cell_enc = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_units)
        return cell_enc

    def buildDecoder(self):
        cell_units = []

        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.dec_size, name="Shared_VAE", reuse=tf.compat.v1.AUTO_REUSE)
        cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob,seed=seed)
        cell_units.append(cell)

        for num_units in range(self.dec_layers-1):
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.dec_size, name="observed_only_VAE", reuse=tf.compat.v1.AUTO_REUSE)
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob,seed=seed)
            cell_units.append(cell)

        cell_dec = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_units)
        return cell_dec

    def buildSampling(self):
        w_mu = self.weight_variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='w_mu')   
        b_mu = self.bias_variable([self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='b_mu')
        w_sigma = self.weight_variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='w_sigma')
        b_sigma = self.bias_variable([self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='b_sigma')

        w_h_dec = self.weight_variable([self.dec_size, self.dim], scope_name='Decoder/Linear/observed_only_VAE', name='w_h_dec')
        b_h_dec = self.bias_variable([self.dim], scope_name='Decoder/Linear/observed_only_VAE', name='b_h_dec')

        return w_mu, b_mu, w_sigma, b_sigma, w_h_dec, b_h_dec

    def weight_variable(self, shape, scope_name, name):
        with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            # initial = tf.truncated_normal(shape, stddev=0.1)
            tf2.random.set_seed(42)

            wv = tf.compat.v1.get_variable(name=name, shape=shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed = seed),use_resource= False)
        return wv

    def bias_variable(self, shape, scope_name, name=None):
        with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            tf2.random.set_seed(42)

            # initial = tf.constant(0.1, shape=shape)
            bv = tf.compat.v1.get_variable(name=name, shape=shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed = seed),use_resource= False)
        return bv

class IMM_vae(object):
    def __init__(self,
                 time_steps,
                 dim, z_dim,
                 enc_size=256, dec_size=256,
                 enc_layers=3, dec_layers=3,
                 keep_prob=1, l2scale=0.0001,
                 conditional=True, num_labels=0):
        

        self.time_steps = time_steps
        self.dim = dim
        self.z_dim = z_dim
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.keep_prob = keep_prob
        self.l2scale = l2scale
        self.conditional = conditional
        self.num_labels = num_labels

    def build_vae(self, input_data, batch_size, conditions=None):
        tf2.random.set_seed(42)
        if self.conditional:
            assert not self.num_labels == 0
            repeated_encoding = conditions
            # repeated_encoding = tf.reshape(conditions, [batch_size, self.time_steps, -1])
            # repeated_encoding = tf.stack([conditions] * self.time_steps, axis=1)
            input_data_cond = tf.concat([input_data, repeated_encoding], axis=-1)
            print(input_data_cond.get_shape(), batch_size)
            input_enc = tf.unstack(input_data_cond, axis=1)
        else:
            input_enc = tf.unstack(input_data, axis=1)
            
     

        self.cell_enc = self.buildEncoder()
        self.cell_dec = self.buildDecoder()
        enc_state = self.cell_enc.zero_state(batch_size, tf.float32)
        dec_state = self.cell_dec.zero_state(batch_size, tf.float32)

        self.e = tf2.random.normal((batch_size, self.z_dim))
        self.c, mu, logsigma, sigma, z = [0] * self.time_steps, [0] * self.time_steps, \
                                        [0] * self.time_steps, [0] * self.time_steps, \
                                         [0]* self.time_steps

        w_mu, b_mu, w_sigma, b_sigma, self.w_h_dec, self.b_h_dec = self.buildSampling()

        for t in range(self.time_steps):
            if t == 0:
                c_prev = tf.zeros((batch_size, self.dim))
            else:
                c_prev = self.c[t - 1]

            c_sigmoid = tf.sigmoid(c_prev)

            if self.conditional:
                x_hat = tf.unstack(input_data, axis=1)[t] - c_sigmoid
            else:
                x_hat = input_enc[t] - c_sigmoid

            with tf.compat.v1.variable_scope('Encoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.compat.v1.AUTO_REUSE):
                tf2.random.set_seed(42)

                h_enc, enc_state = self.cell_enc(tf.concat([input_enc[t], x_hat], 1), enc_state)

            mu[t] = tf.matmul(h_enc, w_mu) + b_mu 
            logsigma[t] = tf.matmul(h_enc, w_sigma) + b_sigma 
            sigma[t] = tf.exp(logsigma[t])

            if self.conditional:
                z[t] = mu[t] + sigma[t] * self.e
                z[t] = tf.concat([z[t], conditions[:,t,:]], axis=-1)
            
            else:
                z[t] = mu[t] + sigma[t] * self.e

            with tf.compat.v1.variable_scope('Decoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.compat.v1.AUTO_REUSE):
                tf2.random.set_seed(42)

                h_dec, dec_state = self.cell_dec(z[t], dec_state)
            self.c[t] = tf.matmul(h_dec, self.w_h_dec) + self.b_h_dec

        self.decoded = tf.stack(self.c, axis=1)
        print(len(z))


        return self.decoded, sigma, mu, logsigma, z

    def reconstruct_decoder(self, dec_input,batch_size, conditions=None):
        rec_decoded = [0] * self.time_steps
        rec_dec_state = self.cell_dec.zero_state(batch_size, dtype=tf.int32)
        for t in range(self.time_steps):
            if self.conditional:
                dec_input_with_c = tf.concat([dec_input[t], conditions[:,t,:]], axis=-1)
                rec_h_dec, rec_dec_state = self.cell_dec(dec_input_with_c, rec_dec_state)
            else:
                rec_h_dec, rec_dec_state = self.cell_dec(dec_input[t], rec_dec_state)

            rec_decoded[t] = tf.matmul(rec_h_dec, self.w_h_dec) + self.b_h_dec

        return tf.stack(rec_decoded, axis=1)

    def buildEncoder(self):
        cell_units = []

        for num_units in range(self.enc_layers-1):
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.enc_size, name="IMM_VAE", reuse=tf.compat.v1.AUTO_REUSE)
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob,seed=seed)
            cell_units.append(cell)

        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.enc_size, name="Shared_VAE", reuse=tf.compat.v1.AUTO_REUSE)
        cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob,seed=seed)
        cell_units.append(cell)

        cell_enc = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_units)
        return cell_enc

    def buildDecoder(self):
        cell_units = []

        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.dec_size, name="Shared_VAE", reuse=tf.compat.v1.AUTO_REUSE)
        cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob,seed=seed)
        cell_units.append(cell)

        for num_units in range(self.dec_layers-1):
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.dec_size, name="IMM_VAE", reuse=tf.compat.v1.AUTO_REUSE)
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob,seed=seed)
            cell_units.append(cell)

        cell_dec = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_units)

        return cell_dec


    def buildSampling(self):
        w_mu = self.weight_variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='w_mu')
        b_mu = self.bias_variable([self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='b_mu')
        w_sigma = self.weight_variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='w_sigma')
        b_sigma = self.bias_variable([self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='b_sigma')

        w_h_dec = self.weight_variable([self.dec_size, self.dim], scope_name='Decoder/Linear/', name='w_h_dec')
        b_h_dec = self.bias_variable([self.dim], scope_name='Decoder/Linear/IMM_VAE', name='b_h_dec')

        return w_mu, b_mu, w_sigma, b_sigma, w_h_dec, b_h_dec

    def weight_variable(self, shape, scope_name, name):
        with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            tf2.random.set_seed(42)

            # initial = tf.truncated_normal(shape, stddev=0.1)
            wv = tf.compat.v1.get_variable(name=name, shape=shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed = 42),use_resource= False)
        return wv

    def bias_variable(self, shape, scope_name, name=None):
        with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            tf2.random.set_seed(42)

            # initial = tf.constant(0.1, shape=shape)
            bv = tf.compat.v1.get_variable(name=name, shape=shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform",seed = 42),use_resource= False)
        return bv
         
    