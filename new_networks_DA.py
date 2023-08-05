# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:37:23 2023

@author: gghos
"""
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
tf.config.run_functions_eagerly(True)
import os 
import tensorflow as tf2
import warnings
warnings.filterwarnings("ignore")
tf2.random.set_seed(42)
seed = 42
import keras.backend as K
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
from tensorflow.keras.layers import (LSTM,  Permute)
from tensorflow.python.keras.regularizers import L2 as l2_regularizer
tf.compat.v1.disable_v2_behavior()

class observed_only_vae(object):
    def __init__(self,
                  time_steps,
                 dim, z_dim,keep_prob,l2scale,
                 enc_size, dec_size,
                 conditional=True):

        self.time_steps = time_steps
        self.dim = dim
        self.z_dim = z_dim
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.keep_prob = keep_prob
        self.l2scale = l2scale
        self.conditional = conditional
     

    def build_vae(self, input_data, conditions=True):
         tf2.random.set_seed(seed)
     
         self.cell_enc = LSTM(self.enc_size, dropout =self.keep_prob, return_state=True,recurrent_dropout = 0.0)
         self.cell_dec = LSTM(self.dec_size,dropout =self.keep_prob,  return_state=True,recurrent_dropout = 0.0)
         batch_size = K.shape(input_data)[0]

         self.e =tf2.random.normal((batch_size, self.z_dim))
         self.c, mu, logsigma, sigma, z, Alpha_t = [0] * self.time_steps, [0] * self.time_steps, [0] * self.time_steps, \
                                            [0] * self.time_steps, [0] * self.time_steps , [0] * self.time_steps 
         self.w_mu, self.b_mu, self.w_sigma, self.b_sigma, self.w_h_dec, self.b_h_dec = self.buildSampling()
         self.w_W_e,self.b_W_e,self.w_U_e,self.b_U_e, self.w_W_d,self.b_W_d,self.w_U_d,self.b_U_d, self.w_v_d,self.b_v_d, self.w_v_e,self.b_v_e= self.Dual_Attention()
     
         self.enc_state = tf.zeros((batch_size, self.enc_size))

         self.cell_state_enc = tf.zeros((batch_size, self.enc_size))
         
               
         for t in range(self.time_steps):
              with tf.compat.v1.variable_scope('observed_only_VAE_Feature_Attention', reuse=tf.compat.v1.AUTO_REUSE):

                  Alpha_t[t] = self.FeatureAttention(self.enc_state, self.cell_state_enc,input_data)

              X_tilde_t = tf.multiply( Alpha_t[t], input_data[:, None, t, :])
              if self.conditional:

                  X_tilde_t = tf.concat([X_tilde_t, conditions[:, None, t, :]], axis=-1)
              with tf.compat.v1.variable_scope('observed_only_VAE_Encoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.compat.v1.AUTO_REUSE):
                  tf2.random.set_seed(seed)
              
                  self.enc_state, _, self.cell_state_enc = self.cell_enc(X_tilde_t, initial_state=[self.enc_state, self.cell_state_enc])

              mu[t] = tf.matmul( self.enc_state, self.w_mu) + self.b_mu
              logsigma[t] = tf.matmul( self.enc_state, self.w_sigma) + self.b_sigma
              sigma[t] = tf.exp(logsigma[t])
           
              z[t] = mu[t] + sigma[t] * self.e

         self.cell_state_dec = tf.zeros((batch_size, self.dec_size))
         self.dec_state = tf.zeros((batch_size, self.dec_size))


         x_encoded = tf.stack(z, axis=1)
         for t in range(self.time_steps):
            with tf.compat.v1.variable_scope('observed_only_VAE_Temporal_Attention', reuse=tf.compat.v1.AUTO_REUSE):

                Beta_t = self.TemporalAttention( self.dec_state,self.cell_state_dec, x_encoded)
                context_vector= tf.multiply( Beta_t[t], x_encoded[:, None, t, :])
            if self.conditional:
                context_vector = tf.concat([context_vector, conditions], axis=-1)

            with tf.compat.v1.variable_scope('observed_only_VAE_Decoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.compat.v1.AUTO_REUSE):
                  tf2.random.set_seed(seed)
                  self.dec_state, _, self.cell_state_dec = self.cell_dec(context_vector, initial_state=[self.dec_state, self.cell_state_dec])
                  activation= tf.keras.layers.Activation("sigmoid")

            self.c[t] = activation(tf.matmul(self.dec_state, self.w_h_dec) + self.b_h_dec)
     
         self.decoded = tf.stack(self.c, axis=1)
         
         return self.decoded, sigma, mu, logsigma, z
     
    def buildSampling(self):
        w_mu = self.variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/observed_only_VAE', name='w_mu')   
        b_mu = self.variable([self.z_dim], scope_name='Sampling_layer/observed_only_VAE', name='b_mu')
        w_sigma = self.variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/observed_only_VAE', name='w_sigma')
        b_sigma = self.variable([self.z_dim], scope_name='Sampling_layer/observed_only_VAE', name='b_sigma')

        w_h_dec = self.variable([self.dec_size, self.dim], scope_name='Decoder/Linear/observed_only_VAE', name='w_h_dec')
        b_h_dec = self.variable([self.dim], scope_name='Decoder/Linear/observed_only_VAE', name='b_h_dec')

        return w_mu, b_mu, w_sigma, b_sigma, w_h_dec, b_h_dec
    
    def Dual_Attention(self):
        w_W_e = self.variable([self.enc_size*2,self.time_steps], scope_name='observed_only_VAE/FeatureAttention', name='w_W_e')
        b_W_e = self.variable([self.time_steps], scope_name='observed_only_VAE/FeatureAttention', name='b_W_e')
        
        w_U_e = self.variable([self.time_steps,self.time_steps], scope_name='observed_only_VAE/FeatureAttention', name='w_U_e')
        b_U_e = self.variable([self.time_steps], scope_name='observed_only_VAE/FeatureAttention', name='b_U_e')
        
        w_v_e = self.variable([self.time_steps*2,1], scope_name='observed_only_VAE/FeatureAttention', name='w_v_e')
        b_v_e = self.variable([1], scope_name='observed_only_VAE/FeatureAttention', name='b_v_e')
        
        w_W_d = self.variable([self.dec_size*2,self.dec_size], scope_name='observed_only_VAE/TemporalAttention', name='w_W_d')
        b_W_d= self.variable([self.dec_size], scope_name='observed_only_VAE/TemporalAttention', name='b_W_d')
        
        w_U_d = self.variable([self.z_dim,self.dec_size], scope_name='observed_only_VAE/TemporalAttention', name='w_U_d')
        b_U_d = self.variable([self.dec_size], scope_name='observed_only_VAE/TemporalAttention', name='b_U_d')
        
        w_v_d = self.variable([self.enc_size*2,1], scope_name='observed_only_VAE/TemporalAttention', name='w_v_d')
        b_v_d = self.variable([1], scope_name='observed_only_VAE/TemporalAttention', name='b_v_d')
        

        return w_W_e,b_W_e,w_U_e,b_U_e, w_W_d,b_W_d,w_U_d,b_U_d, w_v_d, b_v_d, w_v_e,b_v_e

    def variable(self, shape, scope_name, name):
       with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
           tf2.random.set_seed(seed)
           v = tf.compat.v1.get_variable(name=name, shape=shape, initializer=tf2.keras.initializers.GlorotUniform(seed = seed))
       return v

    def Discriminator(self,input_discriminator,IMM):
        batch_size = K.shape(input_discriminator)[0]
        inputs= tf.concat([input_discriminator, IMM], axis = -1)
        w_d = self.variable([self.dec_size, self.dim], scope_name='observed_only_VAE/Discriminator_oo', name='w_d')
        b_d = self.variable([self.dim], scope_name='observed_only_VAE/Discriminator_oo', name='b_d')
        self.discrim =  LSTM(self.enc_size, return_sequences=True,recurrent_dropout = 0.0)
        enc_state = tf.zeros((batch_size, self.time_steps, self.enc_size))
        with tf.variable_scope("observed_only_VAE/Discriminator_oo", regularizer=l2_regularizer(self.l2scale), reuse=tf.compat.v1.AUTO_REUSE):
             enc_state = self.discrim(inputs)
             x= tf.matmul(enc_state, w_d) + b_d
             layer = tf.keras.layers.Activation('sigmoid')
             out= layer(x)
        return(out)

    def FeatureAttention(self, hidden_state, cell_state, input_data):
              n = input_data.shape[2]
              hs = K.repeat(
                  tf.concat([hidden_state, cell_state], axis=-1), n)
              W_e= tf.matmul(hs, self.w_W_e) + self.b_W_e
              U_e= tf.matmul(Permute((2, 1))(input_data), self.w_U_e) + self.b_U_e
              tanh = tf.math.tanh(tf.concat([W_e,U_e ], axis=-1))
              v_e= tf.matmul(tanh, self.w_v_e) + self.b_v_e
              return tf.nn.softmax(Permute((2, 1))(v_e))
             
    def TemporalAttention(self,hidden_state,cell_state,X_encoded):
            W_d= tf.matmul( K.repeat(tf.concat([hidden_state, cell_state], axis=-1), X_encoded.shape[1]), self.w_W_d) + self.b_W_d
            U_d= tf.matmul(X_encoded, self.w_U_d) + self.b_U_d
            l= tf.matmul(tf.math.tanh(tf.concat([W_d ,U_d], axis=-1)), self.w_v_d) + self.b_v_d
            return tf.nn.softmax(l, axis=1)
      

         
class IMM_vae(object):
    def __init__(self,
                 time_steps,dim, z_dim,keep_prob,l2scale,
                 enc_size, dec_size,
                 conditional=True):
        

        self.time_steps = time_steps
        self.dim = dim
        self.z_dim = z_dim
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.keep_prob = keep_prob
        self.l2scale = l2scale
        self.conditional = conditional

    def build_vae(self, input_data, conditions=True):
         tf2.random.set_seed(seed)
         batch_size = K.shape(input_data)[0]

         self.cell_enc = LSTM(self.enc_size, dropout =self.keep_prob, return_state=True,recurrent_dropout = 0.0)
         self.cell_dec = LSTM(self.dec_size,dropout =self.keep_prob,  return_state=True,recurrent_dropout = 0.0)

         self.e =tf2.random.normal((batch_size, self.z_dim))
         self.c, mu, logsigma, sigma, z, Alpha_t = [0] * self.time_steps, [0] * self.time_steps, [0] * self.time_steps, \
                                            [0] * self.time_steps, [0] * self.time_steps , [0] * self.time_steps 
         self.w_mu, self.b_mu, self.w_sigma, self.b_sigma, self.w_h_dec, self.b_h_dec = self.buildSampling()
         self.w_W_e,self.b_W_e,self.w_U_e,self.b_U_e, self.w_W_d,self.b_W_d,self.w_U_d,self.b_U_d, self.w_v_d,self.b_v_d, self.w_v_e,self.b_v_e= self.Dual_Attention()
         self.enc_state = tf.zeros((batch_size, self.enc_size))
         self.cell_state_enc = tf.zeros((batch_size, self.enc_size))
         
               
         for t in range(self.time_steps):
              Alpha_t[t] = self.FeatureAttention(self.enc_state, self.cell_state_enc,input_data)
              X_tilde_t = tf.multiply(Alpha_t[t], input_data[:, None, t, :])
              if self.conditional:
                  X_tilde_t = tf.concat([X_tilde_t, conditions[:, None, t, :]], axis=-1)
              with tf.compat.v1.variable_scope('IMM_VAE_Encoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.compat.v1.AUTO_REUSE):
                  tf2.random.set_seed(seed)
              
                  self.enc_state, _, self.cell_state_enc = self.cell_enc(X_tilde_t, initial_state=[self.enc_state, self.cell_state_enc])

              mu[t] = tf.matmul( self.enc_state, self.w_mu) + self.b_mu
              logsigma[t] = tf.matmul( self.enc_state, self.w_sigma) + self.b_sigma
              sigma[t] = tf.exp(logsigma[t])
           
              z[t] = mu[t] + sigma[t] * self.e

         self.cell_state_dec = tf.zeros((batch_size, self.dec_size))
         self.dec_state = tf.zeros((batch_size, self.dec_size))


         x_encoded = tf.stack(z, axis=1)
         for t in range(self.time_steps):
            Beta_t = self.TemporalAttention( self.dec_state, self.cell_state_dec,x_encoded)
            context_vector= tf.multiply( Beta_t[t], x_encoded[:, None, t, :])
            if self.conditional:
                context_vector = tf.concat([context_vector, conditions], axis=-1)

            with tf.compat.v1.variable_scope('IMM_VAE_Decoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.compat.v1.AUTO_REUSE):
                  tf2.random.set_seed(seed)
                  self.dec_state, _, self.cell_state_dec = self.cell_dec(context_vector, initial_state=[self.dec_state, self.cell_state_dec])
            activation= tf.keras.layers.Activation("sigmoid")
            self.c[t] = activation(tf.matmul(self.dec_state, self.w_h_dec) + self.b_h_dec)
            
         self.decoded = tf.stack(self.c, axis=1)
         return self.decoded, sigma, mu, logsigma, z
     

    def buildSampling(self):
        w_mu = self.variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/IMM_VAE', name='w_mu')
        b_mu = self.variable([self.z_dim], scope_name='Sampling_layer/IMM_VAE', name='b_mu')
        w_sigma = self.variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/IMM_VAE', name='w_sigma')
        b_sigma = self.variable([self.z_dim], scope_name='Sampling_layer/IMM_VAE', name='b_sigma')

        w_h_dec = self.variable([self.dec_size, self.dim], scope_name='Decoder/Linear/IMM_VAE', name='w_h_dec')
        b_h_dec = self.variable([self.dim], scope_name='Decoder/Linear/IMM_VAE', name='b_h_dec')

        return w_mu, b_mu, w_sigma, b_sigma, w_h_dec, b_h_dec

    def variable(self, shape, scope_name, name):
        with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            tf2.random.set_seed(seed)
            v = tf.compat.v1.get_variable(name=name, shape=shape, initializer=tf2.keras.initializers.GlorotUniform(seed = seed))
        return v
       
    def Discriminator(self,input_discriminator, IMM):
         batch_size = K.shape(input_discriminator)[0]
         inputs= tf.concat([input_discriminator, IMM], axis = -1)

         w_d = self.variable([self.dec_size, self.dim], scope_name='IMM_VAE/Discriminator_IMM', name='w_d')
         b_d = self.variable([self.dim], scope_name='IMM_VAE/Discriminator_IMM', name='b_d')
         self.discrim =  LSTM(self.enc_size, return_sequences=True,recurrent_dropout = 0.0)
         enc_state = tf.zeros((batch_size, self.time_steps, self.enc_size))
         with tf.variable_scope("IMM_VAE/Discriminator_IMM", regularizer=l2_regularizer(self.l2scale), reuse=tf.compat.v1.AUTO_REUSE):

              enc_state = self.discrim(inputs)
              x= tf.matmul(enc_state, w_d) + b_d
              layer = tf.keras.layers.Activation('sigmoid')
              out= layer(x)
         return(out)
    def Dual_Attention(self):
        w_W_e = self.variable([self.enc_size*2,self.time_steps], scope_name='IMM_VAE/FeatureAttention', name='w_W_e')
        b_W_e = self.variable([self.time_steps], scope_name='IMM_VAE/FeatureAttention', name='b_W_e')
        
        w_U_e = self.variable([self.time_steps,self.time_steps], scope_name='IMM_VAE/FeatureAttention', name='w_U_e')
        b_U_e = self.variable([self.time_steps], scope_name='IMM_VAE/FeatureAttention', name='b_U_e')
        
        w_v_e = self.variable([self.time_steps*2,1], scope_name='IMM_VAE/FeatureAttention', name='w_v_e')
        b_v_e = self.variable([1], scope_name='IMM_VAE/FeatureAttention', name='b_v_e')
        
        w_W_d = self.variable([self.dec_size*2,self.dec_size], scope_name='IMM_VAE/TemporalAttention', name='w_W_d')
        b_W_d= self.variable([self.dec_size], scope_name='IMM_VAE/TemporalAttention', name='b_W_d')
        
        w_U_d = self.variable([self.z_dim,self.dec_size], scope_name='IMM_VAE/TemporalAttention', name='w_U_d')
        b_U_d = self.variable([self.dec_size], scope_name='IMM_VAE/TemporalAttention', name='b_U_d')
        
        w_v_d = self.variable([self.enc_size*2,1], scope_name='IMM_VAE/TemporalAttention', name='w_v_d')
        b_v_d = self.variable([1], scope_name='IMM_VAE/TemporalAttention', name='b_v_d')
        

        return w_W_e,b_W_e,w_U_e,b_U_e, w_W_d,b_W_d,w_U_d,b_U_d, w_v_d, b_v_d, w_v_e,b_v_e
    
    def FeatureAttention(self, hidden_state, cell_state, input_data):
              n = input_data.shape[2]
              hs = K.repeat(
                  tf.concat([hidden_state, cell_state], axis=-1), n)
              W_e= tf.matmul(hs, self.w_W_e) + self.b_W_e
              U_e= tf.matmul(Permute((2, 1))(input_data), self.w_U_e) + self.b_U_e
              tanh = tf.math.tanh(tf.concat([W_e,U_e ], axis=-1))
              v_e= tf.matmul(tanh, self.w_v_e) + self.b_v_e
              return tf.nn.softmax(Permute((2, 1))(v_e))
             
    def TemporalAttention(self,hidden_state,cell_state,X_encoded):
            W_d= tf.matmul( K.repeat(tf.concat([hidden_state, cell_state], axis=-1), X_encoded.shape[1]), self.w_W_d) + self.b_W_d
            U_d= tf.matmul(X_encoded, self.w_U_d) + self.b_U_d
            l= tf.matmul(tf.math.tanh(tf.concat([W_d ,U_d], axis=-1)), self.w_v_d) + self.b_v_d
            return tf.nn.softmax(l, axis=1)
      
        
