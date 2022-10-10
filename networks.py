import tensorflow.compat.v1 as tf
#update
tf.disable_v2_behavior()
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.regularizers import L2 as l2_regularizer
from init_state import rnn_init_state

class observed_only_vae(object):
    def __init__(self,
                 batch_size, time_steps,
                 dim, z_dim,
                 enc_size=128, dec_size=128,
                 enc_layers=3, dec_layers=3,
                 keep_prob=0.8, l2scale=0.001,
                 conditional=False, num_labels=0):

        self.batch_size = batch_size
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

    def build_vae(self, input_data, conditions=None):
        if self.conditional:
            assert not self.num_labels == 0
            # changed size
            repeated_encoding= conditions
            input_data_cond = tf.concat([input_data, repeated_encoding], axis=-1)
            input_enc = tf.unstack(input_data_cond, axis=1)
        else:
            input_enc = tf.unstack(input_data, axis=1)

        self.cell_enc = self.buildEncoder()
        self.cell_dec = self.buildDecoder()
        enc_state = self.cell_enc.zero_state(self.batch_size, tf.float32)
        dec_state = self.cell_dec.zero_state(self.batch_size, tf.float32)

        self.e = tf.random.normal((self.batch_size, self.z_dim))
        self.c, mu, logsigma, sigma, z = [0] * self.time_steps, [0] * self.time_steps, [0] * self.time_steps, \
                                         [0] * self.time_steps, [0] * self.time_steps 
        w_mu, b_mu, w_sigma, b_sigma, self.w_h_dec, self.b_h_dec = self.buildSampling()

        for t in range(self.time_steps):
            if t == 0:
                c_prev = tf.zeros((self.batch_size, self.dim))
            else:
                c_prev = self.c[t - 1]

            c_sigmoid = tf.sigmoid(c_prev)

            if self.conditional:
                x_hat = tf.unstack(input_data, axis=1)[t] - c_sigmoid
            else:
                x_hat = input_enc[t] - c_sigmoid

            with tf.compat.v1.variable_scope('Encoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.compat.v1.AUTO_REUSE):
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
                h_dec, dec_state = self.cell_dec(z[t], dec_state)

            self.c[t] = tf.matmul(h_dec, self.w_h_dec) + self.b_h_dec

        self.decoded = tf.stack(self.c, axis=1)

        return self.decoded, sigma, mu, logsigma, z

    def reconstruct_decoder(self, dec_input, conditions=None):
        rec_decoded = [0] * self.time_steps
        rec_dec_state = self.cell_dec.zero_state(self.batch_size, dtype=tf.float32)
        for t in range(self.time_steps):
            if self.conditional:
                print("reconstruct_decoder")
                print(conditions.get_shape(), dec_input[t].get_shape())
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
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            cell_units.append(cell)

        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.enc_size, name="Shared_VAE", reuse=tf.compat.v1.AUTO_REUSE)
        cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        cell_units.append(cell)

        cell_enc = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_units)
        return cell_enc

    def buildDecoder(self):
        cell_units = []

        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.dec_size, name="Shared_VAE", reuse=tf.compat.v1.AUTO_REUSE)
        cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        cell_units.append(cell)

        for num_units in range(self.dec_layers-1):
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.dec_size, name="observed_only_VAE", reuse=tf.compat.v1.AUTO_REUSE)
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
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
            wv = tf.compat.v1.get_variable(name=name, shape=shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        return wv

    def bias_variable(self, shape, scope_name, name=None):
        with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            # initial = tf.constant(0.1, shape=shape)
            bv = tf.compat.v1.get_variable(name=name, shape=shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        return bv

class IMM_vae(object):
    def __init__(self,
                 batch_size, time_steps,
                 dim, z_dim,
                 enc_size=128, dec_size=128,
                 enc_layers=3, dec_layers=3,
                 keep_prob=0.8, l2scale=0.001,
                 conditional=False, num_labels=0):

        self.batch_size = batch_size
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

    def build_vae(self, input_data, conditions=None):
        if self.conditional:
            assert not self.num_labels == 0
            repeated_encoding = conditions
            # repeated_encoding = tf.reshape(conditions, [self.batch_size, self.time_steps, -1])
            # repeated_encoding = tf.stack([conditions] * self.time_steps, axis=1)
            input_data_cond = tf.concat([input_data, repeated_encoding], axis=-1)
            print(input_data_cond.get_shape())
            input_enc = tf.unstack(input_data_cond, axis=1)
        else:
            input_enc = tf.unstack(input_data, axis=1)

        self.cell_enc = self.buildEncoder()
        self.cell_dec = self.buildDecoder()
        enc_state = self.cell_enc.zero_state(self.batch_size, tf.float32)
        dec_state = self.cell_dec.zero_state(self.batch_size, tf.float32)

        self.e = tf.random.normal((self.batch_size, self.z_dim))
        self.c, mu, logsigma, sigma, z = [None] * self.time_steps, [None] * self.time_steps, \
                                         [None] * self.time_steps, [None] * self.time_steps, \
                                         [None] * self.time_steps

        w_mu, b_mu, w_sigma, b_sigma, self.w_h_dec, self.b_h_dec = self.buildSampling()

        for t in range(self.time_steps):
            if t == 0:
                c_prev = tf.zeros((self.batch_size, self.dim))
            else:
                c_prev = self.c[t - 1]

            c_sigmoid = tf.sigmoid(c_prev)

            if self.conditional:
                x_hat = tf.unstack(input_data, axis=1)[t] - c_sigmoid
            else:
                x_hat = input_enc[t] - c_sigmoid

            with tf.compat.v1.variable_scope('Encoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.compat.v1.AUTO_REUSE):
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
                h_dec, dec_state = self.cell_dec(z[t], dec_state)
            self.c[t] = tf.matmul(h_dec, self.w_h_dec) + self.b_h_dec

        self.decoded = tf.stack(self.c, axis=1)

        return self.decoded, sigma, mu, logsigma, z

    def reconstruct_decoder(self, dec_input, conditions=None):
        rec_decoded = [0] * self.time_steps
        rec_dec_state = self.cell_dec.zero_state(self.batch_size, dtype=tf.float32)
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
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            cell_units.append(cell)

        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.enc_size, name="Shared_VAE", reuse=tf.compat.v1.AUTO_REUSE)
        cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        cell_units.append(cell)

        cell_enc = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_units)
        return cell_enc

    def buildDecoder(self):
        cell_units = []

        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.dec_size, name="Shared_VAE", reuse=tf.compat.v1.AUTO_REUSE)
        cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        cell_units.append(cell)

        for num_units in range(self.dec_layers-1):
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.dec_size, name="IMM_VAE", reuse=tf.compat.v1.AUTO_REUSE)
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
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
            # initial = tf.truncated_normal(shape, stddev=0.1)
            wv = tf.compat.v1.get_variable(name=name, shape=shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        return wv

    def bias_variable(self, shape, scope_name, name=None):
        with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            # initial = tf.constant(0.1, shape=shape)
            bv = tf.compat.v1.get_variable(name=name, shape=shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        return bv
         
    def build_Discriminator(self, input_discriminator):
       with tf.variable_scope("Discrete_discriminator", regularizer=l2_regularizer(self.l2scale)):
           cell_units = []
           for num_units in range(self.enc_layers):
               cell = tf.nn.rnn_cell.LSTMCell(self.batch_size, reuse=tf.AUTO_REUSE)
               cell =  tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
               cell_units.append(cell)
           d_rnn_network  = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_units)
           initial_state = rnn_init_state(init_="variable", batch_size=self.batch_size,
                                          num_layers=self.enc_layers, num_units=self.batch_size)
           inputs = tf.unstack(input_discriminator, axis=1)
           outputs, _ = tf.nn.static_rnn(cell=d_rnn_network, inputs=inputs, initial_state=initial_state)
           outputs = tf.stack(outputs, axis=1) 
           result = tf.layers.dense(tf.layers.flatten(outputs), 1, reuse=tf.AUTO_REUSE, activation=None) 
       return result