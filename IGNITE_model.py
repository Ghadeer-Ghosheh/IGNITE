# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 17:25:30 2022

@author: gghos
"""
import tensorflow.compat.v1 as tf

tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
import tensorflow as tf2

import numpy as np
import keras.backend as K

import os
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "True"
import os
tf2.keras.utils.set_random_seed(42)
tf2.config.experimental.enable_op_determinism()
import math
from downstream_eval import get_results_2
from Contrastivelosslayer import nt_xent_loss

from prep_inputs import input_impute
tf.compat.v1.disable_v2_behavior()

class IGNITE(object):
    def __init__(self,
                 # -- shared params:
                 batch_size, time_steps,
                 num_epochs, 
                 # -- params for observed-only-vae
                 oo_vae_dim,  checkpoint_dir,
                 z_size,observed_only_data_sample,
                 observed_only_vae,
                 # -- params for imm-vae
                enc_size, dec_size,
                IMM_mask,

                 imm_vae_dim, IMM_data_sample, indicating_mask_sample,
                 IMM_vae, 
                 outcomes,
                 alpha_re, alpha_kl,  alpha_discrim, alpha_semantic, 
                          alpha_contrastive, alpha_matching,

                  alpha_MIT, IGNITE_lr,
                 binary_mask_data_sample,experiment_name,keep_prob,
                 conditional=True, num_labels=0,
                 interventions= None):

        self.sess = tf.InteractiveSession()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.num_epochs = num_epochs
        self.interventions = interventions
        self.experiment_name = experiment_name
        self.dec_size = dec_size
        self.checkpoint_dir = checkpoint_dir

        self.enc_size = enc_size

        # params for observed_only-VAE
        self.oo_vae_dim = oo_vae_dim
        self.z_size = z_size
        self.observed_only_data_sample =observed_only_data_sample
        self.observed_only_vae_net = observed_only_vae

        # params for IMM-VAE
        self.imm_vae_dim = imm_vae_dim
        self.IMM_data_sample = IMM_data_sample
        self.IMM_vae_net = IMM_vae

        self.binary_mask_data_sample = binary_mask_data_sample
        self.indicating_mask_sample = indicating_mask_sample

        # params for interventions information
        self.num_labels = num_labels

        self.conditional = conditional
        self.name = "logs/neww/"+self.experiment_name
        self.keep_prob = keep_prob

        self.alpha_re = alpha_re
        self.IGNITE_lr = IGNITE_lr

        self.alpha_kl = alpha_kl
        self.alpha_MIT = alpha_MIT
        self.alpha_matching = alpha_matching
        self.alpha_contrastive = alpha_contrastive
        self.alpha_discrim = alpha_discrim
        self.alpha_semantic = alpha_semantic
        self.outcomes = outcomes
        self.IMM_mask= IMM_mask
        
    def build(self):
        self.build_tf_graph()
        self.build_loss()
        self.build_summary()
        self.saver = tf.train.Saver()

        
    def save(self, global_id, model_name=None, checkpoint_dir=None):
        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, model_name), global_step=global_id)

    def load(self, model_name=None, checkpoint_dir=None):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        global_id = int(ckpt_name[len(model_name) + 1:])
        return global_id


    def build_tf_graph(self):

        self.binary_mask_data_pl = tf.placeholder(
            dtype=bool, shape=[None, self.time_steps, self.oo_vae_dim], name="observed_only_real_binary_mask")
       
        self.indicating_mask_sample_pl = tf.placeholder(
            dtype=bool, shape=[None, self.time_steps, self.oo_vae_dim], name="indicating_mask_sample")
        self.IMM_mask_pl = tf.placeholder(
         dtype=float, shape=[None, self.time_steps, self.imm_vae_dim], name="IMM_mask_pl")
        
        self.outcome_pl = tf.placeholder(
         dtype=bool, shape=[None], name="outcomes")
        if self.conditional:
            self.real_data_label_pl = tf.placeholder(
                    dtype=float, shape=[None,self.time_steps, self.num_labels], name="conditions")


        self.observed_only_real_data_pl = tf.placeholder(
            dtype=float, shape=[None, self.time_steps, self.oo_vae_dim], name="observed_only_real_data")
 
        if self.conditional:
            self.observed_only_decoded_output, self.observed_only_vae_sigma, self.observed_only_vae_mu, self.observed_only_vae_logsigma, self.c_enc_z = \
                self.observed_only_vae_net.build_vae(self.observed_only_real_data_pl, self.real_data_label_pl)
            
        else:
            self.observed_only_decoded_output, self.observed_only_vae_sigma, self.observed_only_vae_mu, self.observed_only_vae_logsigma, self.c_enc_z = \
                self.observed_only_vae_net.build_vae(self.observed_only_real_data_pl)

     
        self.IMM_real_data_pl = tf.placeholder(
            dtype=float, shape=[None, self.time_steps, self.imm_vae_dim], name="IMM__Input_real_data")
     
        if self.conditional:
            self.IMM_decoded_output, self.IMM_vae_sigma, self.IMM_vae_mu, self.IMM_vae_logsigma, self.d_enc_z = \
                self.IMM_vae_net.build_vae(self.IMM_real_data_pl, self.real_data_label_pl)
        else:
            self.IMM_decoded_output, self.IMM_vae_sigma, self.IMM_vae_mu, self.IMM_vae_logsigma, self.d_enc_z = \
                self.IMM_vae_net.build_vae(self.IMM_real_data_pl)
        
        Hat_X = self.observed_only_decoded_output*tf.cast(self.binary_mask_data_pl, tf.float32) + \
            self.observed_only_real_data_pl * (1-tf.cast(self.binary_mask_data_pl, tf.float32))
        Hat_X_imm = self.IMM_decoded_output*tf.cast(self.binary_mask_data_pl, tf.float32) +\
            self.IMM_real_data_pl * (1-tf.cast(self.binary_mask_data_pl, tf.float32))

        self.prob_real = self.observed_only_vae_net.Discriminator(Hat_X, self.IMM_mask_pl)
        self.prob_real_imm = self.IMM_vae_net.Discriminator(Hat_X_imm,self.IMM_mask_pl)
        

    def build_loss(self):

        #################
        # (0) Latent loss  #
        #################
     
        x_latent_1 = tf.stack(self.c_enc_z, axis=1)
        x_latent_2 = tf.stack(self.d_enc_z, axis=1)
        
        with tf.variable_scope("observed_only_VAE/semantic_classifier"):
            vae_flatten_input1 = tf.compat.v1.layers.flatten(x_latent_1)
            vae_flatten_input2 = tf.compat.v1.layers.flatten(x_latent_2)
            vae_hidden_layer = tf.layers.dense(tf.concat([vae_flatten_input1, vae_flatten_input2], axis=-1), units=24, activation=tf.nn.relu)
            vae_logits = tf.layers.dense(vae_hidden_layer, units=1, activation=tf.nn.tanh)
            self.vae_semantics_loss = self.alpha_semantic*tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.squeeze(tf.cast(self.outcome_pl, dtype=tf.float32)), logits= vae_logits))
 
        self.vae_matching_loss = self.alpha_matching*(tf.losses.mean_squared_error(x_latent_1, x_latent_2))
       
       
        self.vae_contra_loss = self.alpha_contrastive*(nt_xent_loss(tf.reshape(x_latent_1, [K.shape(x_latent_1)[0], -1]),
                                 tf.reshape(x_latent_2, [K.shape(x_latent_1)[0], -1]), K.shape(x_latent_1)[0]))
   
        #################
         # (1) OO loss  #
        #################
        binary_masked_observed_only_decoded_output = tf.where(self.binary_mask_data_pl, self.observed_only_decoded_output, tf.zeros_like(self.observed_only_decoded_output))  
        binary_masked_observed_only_real_data = tf.where(self.binary_mask_data_pl, self.observed_only_real_data_pl, tf.zeros_like(self.observed_only_real_data_pl))  

        MIT_masked_observed_only_real_output = tf.where(self.indicating_mask_sample_pl, self.observed_only_real_data_pl, tf.zeros_like(self.observed_only_real_data_pl))  

        MIT_masked_observed_only_decoded_output = tf.where(self.indicating_mask_sample_pl, self.observed_only_decoded_output, tf.zeros_like(self.observed_only_decoded_output))  

        self.observed_only_re_loss = self.alpha_re*(tf.losses.mean_squared_error(binary_masked_observed_only_real_data,binary_masked_observed_only_decoded_output))
        observed_only_kl_loss = [0] * self.time_steps 
        for t in range(self.time_steps):
            observed_only_kl_loss[t] = 0.5 * (tf.reduce_sum(self.observed_only_vae_sigma[t], 1) + tf.reduce_sum(
                tf.square(self.observed_only_vae_mu[t]), 1) - tf.reduce_sum(self.observed_only_vae_logsigma[t] + 1, 1))
        
        self.observed_only_kl_loss =self.alpha_kl*( tf.reduce_mean(tf.add_n(observed_only_kl_loss)))
        self.MIT_loss = self.alpha_MIT*(tf.losses.mean_squared_error(MIT_masked_observed_only_real_output,MIT_masked_observed_only_decoded_output))


        self.discriminator_loss_oo =self.alpha_discrim* (-tf.reduce_mean(tf.cast(self.binary_mask_data_pl, tf.float32) * tf.log(self.prob_real + 1e-8) + \
                                                    (1-tf.cast(self.binary_mask_data_pl, tf.float32)) * tf.log(1. - self.prob_real + 1e-8))) 

        self.observed_only_vae_loss = self.observed_only_re_loss + \
                            self.observed_only_kl_loss  +self.vae_semantics_loss +self.discriminator_loss_oo + \
                            self.vae_matching_loss +self.MIT_loss + self.vae_contra_loss


         #################
         # (2) IMM loss  #
         #################
        MIT_masked_IMM_real_output = tf.where(self.indicating_mask_sample_pl, self.IMM_real_data_pl, tf.zeros_like(self.IMM_real_data_pl))  

        MIT_masked_IMM_decoded_output = tf.where(self.indicating_mask_sample_pl, self.IMM_decoded_output, tf.zeros_like(self.IMM_decoded_output))  
         
        self.IMM_re_loss =  self.alpha_re*(tf.losses.mean_squared_error(self.IMM_real_data_pl, self.IMM_decoded_output))
        IMM_kl_loss = [0] * self.time_steps
        for t in range(self.time_steps):
            IMM_kl_loss[t] = 0.5 * (tf.reduce_sum(self.IMM_vae_sigma[t], 1) + tf.reduce_sum(
                tf.square(self.IMM_vae_mu[t]), 1) - tf.reduce_sum(self.IMM_vae_logsigma[t] + 1, 1))
        self.IMM_kl_loss = self.alpha_kl*(tf.reduce_mean(tf.add_n(IMM_kl_loss)))
        self.IMM_MIT_loss = self.alpha_re*(tf.losses.mean_squared_error(MIT_masked_IMM_real_output,MIT_masked_IMM_decoded_output))

        self.discriminator_loss_imm =self.alpha_discrim*(-tf.reduce_mean(tf.cast(self.binary_mask_data_pl, tf.float32) * tf.log(self.prob_real_imm + 1e-8) + \
                                                     (1-tf.cast(self.binary_mask_data_pl, tf.float32)) * tf.log(1. - self.prob_real_imm + 1e-8)))

        self.IMM_vae_loss =self.IMM_re_loss + self.IMM_kl_loss + self.discriminator_loss_imm + self.IMM_MIT_loss 

        
        #######################
        # Optimizers           #
        #######################
        t_vars = tf.trainable_variables()
        observed_only_vae_vars = [var for var in t_vars if 'observed_only_VAE' in var.name]
        IMM_vae_vars = [var for var in t_vars if 'IMM_VAE' in var.name]

        self.oo_v_op_pre = tf.train.AdamOptimizer(learning_rate=self.IGNITE_lr)\
            .minimize(self.observed_only_vae_loss, var_list=observed_only_vae_vars)

        self.imm_v_op_pre = tf.train.AdamOptimizer(self.IGNITE_lr)\
            .minimize(self.IMM_vae_loss, var_list=IMM_vae_vars)
      
     
    def build_summary(self):
        self.observed_only_vae_summary = []
        self.observed_only_vae_summary.append(tf.summary.scalar("observed_only_loss/reconstruction_loss", self.observed_only_re_loss))
        self.observed_only_vae_summary.append(tf.summary.scalar("observed_only_loss/kl_divergence_loss", self.observed_only_kl_loss))
        self.observed_only_vae_summary.append(tf.summary.scalar("observed_only_loss/MIT_loss", self.MIT_loss))
        self.observed_only_vae_summary.append(tf.summary.scalar("observed_only_loss/matching_loss", self.vae_matching_loss))
        self.observed_only_vae_summary.append(tf.summary.scalar("observed_only_loss/discriminator_loss_oo", self.discriminator_loss_oo))
        self.observed_only_vae_summary.append(tf.summary.scalar("observed_only_loss/semantic_loss", self.vae_semantics_loss))
        self.observed_only_vae_summary.append(tf.summary.scalar("observed_only_loss/contrastive_loss", self.vae_contra_loss))
        self.observed_only_vae_summary.append(tf.summary.scalar("observed_only_loss/vae_loss", self.observed_only_vae_loss))
        self.observed_only_vae_summary = tf.summary.merge(self.observed_only_vae_summary)

        self.IMM_vae_summary = []
        self.IMM_vae_summary.append(tf.summary.scalar("IMM_VAE_loss/reconstruction_loss", self.IMM_re_loss))
        self.IMM_vae_summary.append(tf.summary.scalar("IMM_VAE_loss/kl_divergence_loss", self.IMM_kl_loss))
        self.IMM_vae_summary.append(tf.summary.scalar("IMM_VAE_loss/IMM_MIT_loss", self.IMM_MIT_loss))
        self.IMM_vae_summary.append(tf.summary.scalar("IMM_VAE_loss/matching_loss", self.vae_matching_loss))
        self.IMM_vae_summary.append(tf.summary.scalar("IMM_VAE_loss/discriminator_loss_imm", self.discriminator_loss_imm))
        #self.IMM_vae_summary.append(tf.summary.scalar("IMM_VAE_loss/vae_semantics_loss_IMM", self.vae_semantics_loss_IMM))
        self.IMM_vae_summary.append(tf.summary.scalar("IMM_VAE_loss/vae_loss", self.IMM_vae_loss))
        self.IMM_vae_summary = tf.summary.merge(self.IMM_vae_summary)

                
    def train(self):
        # training IGNITE model and updating losses

        self.summary_writer = tf.summary.FileWriter(self.name, self.sess.graph)

        num_batches = math.ceil(self.observed_only_data_sample.shape[0] / self.batch_size)
        #tf.global_variables_initializer().run()
        self.sess.run(tf.global_variables_initializer())

        print('start training')
        for epoch in range(self.num_epochs):
            print("Epoch %d" % epoch)

            IMM_rec_data_lst,observed_only_rec_data_lst = [],[]
            
            for batch_index in range(num_batches):

                feed_dict = {}
                feed_dict[self.observed_only_real_data_pl] =  self.observed_only_data_sample[batch_index* self.batch_size: (batch_index +1) * self.batch_size]
                feed_dict[self.IMM_real_data_pl] = self.IMM_data_sample[batch_index* self.batch_size: (batch_index +1) * self.batch_size]
                feed_dict[self.binary_mask_data_pl] = self.binary_mask_data_sample[batch_index* self.batch_size: (batch_index +1) * self.batch_size]
                feed_dict[self.IMM_mask_pl] = self.IMM_mask[batch_index* self.batch_size: (batch_index +1) * self.batch_size]
                feed_dict[self.indicating_mask_sample_pl] =  np.where( self.indicating_mask_sample[batch_index* self.batch_size: batch_index * self.batch_size + self.batch_size],\
                                                                      feed_dict[self.observed_only_real_data_pl], np.zeros_like( feed_dict[self.observed_only_real_data_pl]))  
                feed_dict[self.outcome_pl] = self.outcomes[batch_index* self.batch_size: (batch_index +1) * self.batch_size]

                if self.conditional:
                    feed_dict[self.real_data_label_pl] = self.interventions[batch_index* self.batch_size: (batch_index +1) * self.batch_size]
     

                summary_result_observed_only, _,observed_only_rec_data= self.sess.run([self.observed_only_vae_summary, self.oo_v_op_pre,self.observed_only_decoded_output], feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_result_observed_only, epoch)
                #wandb.tensorflow.log(summary_result_observed_only,epoch)
                observed_only_rec_data_lst.append(observed_only_rec_data)

    
                summary_result_IMM, _ , IMM_rec_data= self.sess.run([self.IMM_vae_summary, self.imm_v_op_pre, self.IMM_decoded_output], feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_result_IMM, epoch)
                IMM_rec_data_lst.append(IMM_rec_data)

                #wandb.tensorflow.log(summary_result_IMM,epoch)

             
            observed_only_rec_data_lst_rec=np.vstack(observed_only_rec_data_lst)

            oo_ours =(self.binary_mask_data_sample * self.observed_only_data_sample)+ ((1-self.binary_mask_data_sample)*observed_only_rec_data_lst_rec)
            auc, auprc,test_f1,test_balanced_accuracy,f, g=get_results_2(["results"],[oo_ours],self.outcomes)
           # wandb.log({"aucs_oo": auc,"aurpcs_oo": auprc,"test_f1_oo": test_f1,"test_balanced_accuracy_oo": test_balanced_accuracy, "epoch": epoch})
            
            IMM_rec=np.vstack(IMM_rec_data_lst)
            imputed_ours =(self.binary_mask_data_sample *self.observed_only_data_sample)+ ((1-self.binary_mask_data_sample)*IMM_rec)
            auc, auprc,test_f1,test_balanced_accuracy, f, g=get_results_2(["results"],[imputed_ours],self.outcomes)
            #wandb.log({"aucs": auc,"aurpcs": auprc,  "epoch": epoch, "test_f1": test_f1,"test_balanced_accuracy": test_balanced_accuracy})
        np.savez('data/'+self.experiment_name+'.npz', observed_only_real=self.observed_only_data_sample, observed_only_rec=observed_only_rec_data_lst_rec,
                                     IMM_real=self.IMM_data_sample, IMM_rec=IMM_rec)
        self.save_path = os.path.join(self.checkpoint_dir, "pretrain_vae_{}".format(epoch))
        if not os.path.exists(self.save_path):
             os.makedirs(self.save_path)
        self.save(global_id=epoch - 1, model_name='IGNITE', checkpoint_dir=self.save_path)
     
        print('finished model training')
        
        
    def test(self, test_data, test_condition= None):
        # impute the test set without retraining (to be used for the MCAR reconstruction task)
        miss, mask_personalized, zero, noise_input, IMM_input=input_impute(test_data)

        num_batches = math.ceil(test_data.shape[0] / self.batch_size)
        imputed_data,imputed_data_IMM = [], []
        for batch_index in range(num_batches):
            feed_dict = {}
            feed_dict[self.observed_only_real_data_pl] = zero[batch_index* self.batch_size: (batch_index +1) * self.batch_size]
            feed_dict[self.real_data_label_pl] = test_condition[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
            feed_dict[self.IMM_real_data_pl] = IMM_input[batch_index* self.batch_size: (batch_index +1) * self.batch_size]

            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))


            imputed= self.sess.run(self.observed_only_decoded_output, feed_dict=feed_dict)            
            imputed_data.append(imputed)            

            imputed_IMM= self.sess.run(self.IMM_decoded_output, feed_dict=feed_dict)          
            imputed_data_IMM.append(imputed_IMM)
        print('finished imputing test')

        np.savez('data/imputed'+self.experiment_name+'.npz', imputed_data_oo=np.vstack(imputed_data), imputed_data_IMM =  np.vstack(imputed_data_IMM))
        
    
    def test_full(self, test_data, test_condition= None):
        # generate the full imputation for the downstream task
        miss, mask_personalized, zero, noise_input, IMM_input=input_impute(test_data)

        num_batches = math.ceil(test_data.shape[0] / self.batch_size)
        imputed_data,imputed_data_IMM = [], []
        for batch_index in range(num_batches):
            feed_dict = {}
            feed_dict[self.observed_only_real_data_pl] = zero[batch_index* self.batch_size: (batch_index +1) * self.batch_size]
            feed_dict[self.real_data_label_pl] = test_condition[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
            feed_dict[self.IMM_real_data_pl] = IMM_input[batch_index* self.batch_size: (batch_index +1) * self.batch_size]

            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))


            imputed= self.sess.run(self.observed_only_decoded_output, feed_dict=feed_dict)            
            imputed_data.append(imputed)            

            imputed_IMM= self.sess.run(self.IMM_decoded_output, feed_dict=feed_dict)          
            imputed_data_IMM.append(imputed_IMM)
        print('finished imputing ALl')

        np.savez('data/imputedfull'+self.experiment_name+'.npz', imputed_data_oo=np.vstack(imputed_data), imputed_data_IMM =  np.vstack(imputed_data_IMM))