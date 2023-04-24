# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:15:34 2023

@author: gghos
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 17:25:30 2022

@author: gghos
"""
import tensorflow as tf2
tf2.random.set_seed(42)
print(tf2. __version__) 
from tfdeterminism import patch
import numpy as np
import os


np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import os

tf2.keras.utils.set_random_seed(42)
tf2.config.experimental.enable_op_determinism()

from Contrastivelosslayer import nt_xent_loss, ConLoss
from utils import ones_target, zeros_target

import math
import wandb
tf2.compat.v1.disable_v2_behavior()

import tensorflow.compat.v1 as tf

from downstream_eval import *


class IGNITE(object):
    def __init__(self, sess,
                 # -- shared params:
                 batch_size, time_steps,
                 num_epochs, 
                 checkpoint_dir,
                 # -- params for observed-only-vae
                 oo_vae_dim, 
                #  c_noise_dim,
                 z_size,observed_only_data_sample,
                 observed_only_vae,
                 # -- params for imm-vae
                 imm_vae_dim, IMM_data_sample,
                 IMM_vae, 
                 outcomes,
                 alpha_re, alpha_kl, alpha_mt, 
                alpha_ct, IGNITE_lr,
                 binary_mask_data_sample,experiment_name,keep_prob,
                 temperature, hidden_norm,
                 conditional=False, num_labels=0,
                 interventions=None):

        self.sess = sess
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.interventions = interventions
        self.temperature = temperature
        self.hidden_norm = hidden_norm
        self.experiment_name = experiment_name

        # params for observed_only-VAE
        self.oo_vae_dim = oo_vae_dim
        self.z_size = z_size
        self.observed_only_data_sample =observed_only_data_sample
        self.observed_only_vae_net = observed_only_vae

        # params for IMM-VAE
        self.imm_vae_dim = int(imm_vae_dim)
        self.IMM_data_sample = IMM_data_sample
        self.IMM_vae_net = IMM_vae

        self.binary_mask_data_sample = binary_mask_data_sample

        # params for interventions information
        self.num_labels = num_labels
        self.conditional = conditional
        self.name = "logs/neww/"+self.experiment_name
        self.keep_prob = keep_prob

        self.alpha_re = alpha_re
        self.alpha_kl = alpha_kl
        self.alpha_mt = alpha_mt
        self.alpha_ct = alpha_ct
        self.IGNITE_lr = IGNITE_lr


        self.outcomes = outcomes
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

    def embed_vis(self, embedding_var, type):
        saving_dir = self.name

    
        metadata_file = os.path.join(saving_dir, 'metadata_classes.tsv')
        with open(metadata_file, 'w') as f:
            f.write("patient_status\tpatient_status1\n")
            for i in range(embedding_var.shape[0]):
                c1 = self.interventions[i, 0]   # patient status
                f.write("%s\t%s\n" % (c1, c1))
        f.close()

        """Setup for Tensorboard embedding visualization"""
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = embedding_var.name
        embed.metadata_path = os.path.join('', 'metadata_classes.tsv')
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(self.summary_writer, config)
        saver_images = tf.train.Saver([embedding_var])
        saver_images.save(self.sess, os.path.join(saving_dir, 'embeddings.ckpt'))


        
    def build_tf_graph(self):
     
        if self.conditional:
            self.real_data_label_pl = tf.placeholder(
                dtype=float, shape=[None,self.time_steps, self.num_labels], name="real_data_label")
                
        self.observed_only_real_data_pl = tf.placeholder(
            dtype=float, shape=[None, self.time_steps, self.oo_vae_dim], name="observed_only_real_data")
        self.observed_only_real_data_binary_mask_pl = tf.placeholder(
            dtype=bool, shape=[None, self.time_steps, self.oo_vae_dim], name="observed_only_real_binary_mask")

        self.dynamic_batch = tf.placeholder_with_default(self.batch_size,shape=())
        
        if self.conditional:
            self.observed_only_decoded_output, self.observed_only_vae_sigma, self.observed_only_vae_mu, self.observed_only_vae_logsigma, self.c_enc_z = \
                self.observed_only_vae_net.build_vae(self.observed_only_real_data_pl,self.dynamic_batch, self.real_data_label_pl)
        else:
            self.observed_only_decoded_output, self.observed_only_vae_sigma, self.observed_only_vae_mu, self.observed_only_vae_logsigma, self.c_enc_z = \
                self.observed_only_vae_net.build_vae(self.observed_only_real_data_pl,self.dynamic_batch)

        

        self.IMM_real_data_pl = tf.placeholder(
            dtype=float, shape=[None, self.time_steps, self.imm_vae_dim], name="real_personalized")
    
        self.outcome_pl = tf.placeholder(
         dtype=bool, shape=[None], name="outcomes")
        if self.conditional:
            self.IMM_decoded_output, self.IMM_vae_sigma, self.IMM_vae_mu, self.IMM_vae_logsigma, self.d_enc_z = \
                self.IMM_vae_net.build_vae(self.IMM_real_data_pl, self.dynamic_batch, self.real_data_label_pl)
        else:
            self.IMM_decoded_output, self.IMM_vae_sigma, self.IMM_vae_mu, self.IMM_vae_logsigma, self.d_enc_z = \
                self.IMM_vae_net.build_vae(self.IMM_real_data_pl,self.dynamic_batch)
                
        self.binary_masked_observed_only_real_data_pl = tf.placeholder(
            dtype=float, shape=[None, self.time_steps, self.oo_vae_dim], name="observed_only_real_data_2")
        self.binary_masked_observed_only_decoded_output =  tf.placeholder(
            dtype=float, shape=[None, self.time_steps, self.oo_vae_dim], name="observed_only_real_data_3")
    

    def build_loss(self):

        #################
        # (1) VAE loss  #
        #################
     
        x_latent_1 = tf.stack(self.c_enc_z, axis=1)
        x_latent_2 = tf.stack(self.d_enc_z, axis=1)
    
        self.vae_matching_loss = self.alpha_mt*(tf.losses.mean_squared_error(x_latent_1, x_latent_2))
        
        
        self.vae_contra_loss = self.alpha_ct*(nt_xent_loss(tf.reshape(x_latent_1, [self.dynamic_batch, -1]),
                                  tf.reshape(x_latent_2, [self.dynamic_batch, -1]), self.dynamic_batch, hidden_norm=self.hidden_norm, temperature=self.temperature))
        
       
        #self.vae_contra_loss = self.alpha_ct*(ConLoss(tf.reshape(x_latent_1, [self.dynamic_batch, -1]),
         #                          tf.reshape(x_latent_2, [self.dynamic_batch, -1]), self.dynamic_batch, temperature=self.temperature))
        
        # HERE
        binary_masked_observed_only_decoded_output = tf.where(self.observed_only_real_data_binary_mask_pl, self.observed_only_decoded_output, tf.zeros_like(self.observed_only_decoded_output))  

        self.observed_only_re_loss = self.alpha_re*(tf.losses.mean_squared_error(self.binary_masked_observed_only_real_data_pl,binary_masked_observed_only_decoded_output))
        observed_only_kl_loss = [0] * self.time_steps 
        for t in range(self.time_steps):
            observed_only_kl_loss[t] = 0.5 * (tf.reduce_sum(self.observed_only_vae_sigma[t], 1) + tf.reduce_sum(
                tf.square(self.observed_only_vae_mu[t]), 1) - tf.reduce_sum(self.observed_only_vae_logsigma[t] + 1, 1))
        
        self.observed_only_kl_loss =self.alpha_kl*( tf.reduce_mean(tf.add_n(observed_only_kl_loss)))
    

        self.observed_only_vae_loss = self.observed_only_re_loss + \
                            self.observed_only_kl_loss + +self.vae_matching_loss + self.vae_contra_loss


    
        
        self.IMM_re_loss =  0.1*self.alpha_re*(tf.losses.mean_squared_error(self.IMM_real_data_pl, self.IMM_decoded_output))
        IMM_kl_loss = [0] * self.time_steps
        for t in range(self.time_steps):
            IMM_kl_loss[t] = 0.5 * (tf.reduce_sum(self.IMM_vae_sigma[t], 1) + tf.reduce_sum(
                tf.square(self.IMM_vae_mu[t]), 1) - tf.reduce_sum(self.IMM_vae_logsigma[t] + 1, 1))
        self.IMM_kl_loss = self.alpha_kl*(tf.reduce_mean(tf.add_n(IMM_kl_loss)))
    
 

        self.IMM_vae_loss =self.IMM_re_loss + \
                            self.IMM_kl_loss 
                            #+ self.vae_contra_loss

        #######################
        # Optimizer           #
        #######################
        t_vars = tf.trainable_variables()
        observed_only_vae_vars = [var for var in t_vars if 'observed_only_VAE' in var.name]
        s_vae_vars = [var for var in t_vars if 'Shared_VAE' in var.name]

        IMM_vae_vars = [var for var in t_vars if 'IMM_VAE' in var.name]
        


        self.oo_v_op_pre = tf.train.AdamOptimizer(learning_rate=self.IGNITE_lr)\
            .minimize(self.observed_only_vae_loss, var_list=observed_only_vae_vars +s_vae_vars)

        self.imm_v_op_pre = tf.train.AdamOptimizer(self.IGNITE_lr)\
            .minimize(self.IMM_vae_loss, var_list=IMM_vae_vars +s_vae_vars)


    def build_summary(self):
        self.observed_only_vae_summary = []
        self.observed_only_vae_summary.append(tf.summary.scalar("observed_only_loss/reconstruction_loss", self.observed_only_re_loss))
        self.observed_only_vae_summary.append(tf.summary.scalar("observed_only_loss/kl_divergence_loss", self.observed_only_kl_loss))
        self.observed_only_vae_summary.append(tf.summary.scalar("observed_only_loss/vae_loss", self.observed_only_vae_loss))
        self.observed_only_vae_summary = tf.summary.merge(self.observed_only_vae_summary)

        self.IMM_vae_summary = []
        self.IMM_vae_summary.append(tf.summary.scalar("IMM_VAE_loss/reconstruction_loss", self.IMM_re_loss))
        self.IMM_vae_summary.append(tf.summary.scalar("IMM_VAE_loss/kl_divergence_loss", self.IMM_kl_loss))
           
        self.IMM_vae_summary.append(tf.summary.scalar("IMM_VAE_loss/contrastive_loss", self.vae_contra_loss))
        self.IMM_vae_summary.append(tf.summary.scalar("IMM_VAE_loss/matching_loss", self.vae_matching_loss))
        #self.IMM_vae_summary.append(tf.summary.scalar("IMM_vae_summary/semantic_loss", self.vae_semantics_loss))
        self.IMM_vae_summary.append(tf.summary.scalar("IMM_VAE_loss/vae_loss", self.IMM_vae_loss))
       
        self.IMM_vae_summary = tf.summary.merge(self.IMM_vae_summary)
        
                                       
    def train(self):
        self.summary_writer = tf.summary.FileWriter(self.name, self.sess.graph)

       
        observed_only_x = self.observed_only_data_sample[: int(self.observed_only_data_sample.shape[0]), :, :]

        binary_mask_x = self.binary_mask_data_sample[: int(self.binary_mask_data_sample.shape[0]), :, :]
        
        outcomes_x = self.outcomes[: int(self.outcomes.shape[0])]

        
        IMM_x = self.IMM_data_sample[: int(self.IMM_data_sample.shape[0]), :, :]

        if self.conditional:
            label_data = self.interventions[: int(self.IMM_data_sample.shape[0]), :, :]
        data_size = observed_only_x.shape[0]
        num_batches = math.ceil(data_size / self.batch_size)
        print(data_size,num_batches)


        tf.global_variables_initializer().run()
        
        

        print('start training')
        global_id = 0

        for epoch in range(self.num_epochs):
            self.epochs = epoch

            if self.conditional:
                label_data_random = label_data
         
            print("Epoch %d" % epoch)

            observed_only_real_data_lst = []
            observed_only_rec_data_lst = []
            IMM_real_data_lst = []
            IMM_rec_data_lst = []
            d_binary_mask_lst = []
            
            for batch_index in range(num_batches):

                feed_dict = {}
               
                feed_dict[self.observed_only_real_data_pl] = observed_only_x[batch_index* self.batch_size: batch_index * self.batch_size + self.batch_size]
                feed_dict[self.IMM_real_data_pl] = IMM_x[batch_index* self.batch_size: batch_index * self.batch_size + self.batch_size]
                feed_dict[self.observed_only_real_data_binary_mask_pl] = binary_mask_x[batch_index* self.batch_size: batch_index * self.batch_size + self.batch_size]
                feed_dict[self.dynamic_batch]= feed_dict[self.observed_only_real_data_binary_mask_pl].shape[0]
                #print(batch_index, feed_dict[self.dynamic_batch])
                #print(feed_dict[self.observed_only_real_data_binary_mask_pl].shape, batch_index)

                feed_dict[self.binary_masked_observed_only_real_data_pl] =  np.where(feed_dict[self.observed_only_real_data_binary_mask_pl],  feed_dict[self.observed_only_real_data_pl], np.zeros_like( feed_dict[self.observed_only_real_data_pl]))  
                feed_dict[self.binary_masked_observed_only_real_data_pl] =  np.where(feed_dict[self.observed_only_real_data_binary_mask_pl],  feed_dict[self.observed_only_real_data_pl], np.zeros_like( feed_dict[self.observed_only_real_data_pl]))  

                #select_outcomes= outcomes_x_random[batch_index*self.batch_size: batch_index * self.batch_size + self.batch_size]
                
                
            

                if self.conditional:
                    feed_dict[self.real_data_label_pl] = label_data_random[batch_index* self.batch_size:  batch_index * self.batch_size + self.batch_size]

                summary_result_observed_only, _ = self.sess.run([self.observed_only_vae_summary, self.oo_v_op_pre], feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_result_observed_only, global_id)
         

                summary_result_IMM, _ = self.sess.run([self.IMM_vae_summary, self.imm_v_op_pre], feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_result_IMM, global_id)
             
                
                
                #wandb.log(summary_result)
                
                observed_only_real_data, observed_only_rec_data, d_binary_mask, masked_real, batch = self.sess.run([self.observed_only_real_data_pl, self.observed_only_decoded_output, self.observed_only_real_data_binary_mask_pl, self.binary_masked_observed_only_real_data_pl, self.dynamic_batch], feed_dict=feed_dict)
                observed_only_real_data_lst.append(observed_only_real_data)
                observed_only_rec_data_lst.append(observed_only_rec_data)

                IMM_real_data, IMM_rec_data,d_binary_mask= self.sess.run([self.IMM_real_data_pl, self.IMM_decoded_output, self.observed_only_real_data_binary_mask_pl], feed_dict=feed_dict)
                IMM_real_data_lst.append(IMM_real_data)
                IMM_rec_data_lst.append(IMM_rec_data)
                d_binary_mask_lst.append(d_binary_mask)


                assert not np.any(np.isnan(IMM_real_data))

                assert not np.any(np.isnan(IMM_rec_data))

                global_id += 1
                '''
                array = d_binary_mask   
                imputed_ours =(array * observed_only_real_data)+ ((1-array)*IMM_rec_data)
                auc, auprc=get_results_2(["results"],[imputed_ours],select_outcomes)
                aucs.append(auc)
                auprcs.append(auprc)
                wandb.log({"auc": auc,"aurpc": auprc, "step":global_id})
                '''
                #wandb.tensorflow.log(summary_result_observed_only,global_id)
                #wandb.tensorflow.log(summary_result_IMM,global_id)

                '''if (epoch%7 == 0):
                   
                    array[array == 0] = np.nan
                    masked_plot=(observed_only_real_data*array)
                    self.compare_plot(masked_plot, observed_only_rec_data,IMM_rec_data, epoch)
                '''
            IMM_rec=np.vstack(IMM_rec_data_lst)
            observed_only_rec_data_lst_rec=np.vstack(observed_only_rec_data_lst)

            observed_only_real=np.vstack(observed_only_real_data_lst)
            d_binary_masks = np.vstack(d_binary_mask_lst)
            imputed_ours =(d_binary_masks * observed_only_real)+ ((1-d_binary_masks)*IMM_rec)
            oo_ours =(d_binary_masks * observed_only_real)+ ((1-d_binary_masks)*observed_only_rec_data_lst_rec)

   
            auc, auprc,test_f1,test_balanced_accuracy, f, g=get_results_2(["results"],[imputed_ours],outcomes_x)
            #wandb.log({"aucs": auc,"aurpcs": auprc,  "epoch": epoch, "test_f1": test_f1,"test_balanced_accuracy": test_balanced_accuracy})

            auc, auprc,test_f1,test_balanced_accuracy,f, g=get_results_2(["results"],[oo_ours],outcomes_x)
            #wandb.log({"aucs_oo": auc,"aurpcs_oo": auprc,"test_f1_oo": test_f1,"test_balanced_accuracy_oo": test_balanced_accuracy})
        np.savez('data/'+self.experiment_name+'.npz', observed_only_real=np.vstack(observed_only_real_data_lst), observed_only_rec=np.vstack(observed_only_rec_data_lst),
                                     IMM_real=np.vstack(IMM_real_data_lst), IMM_rec=np.vstack(IMM_rec_data_lst))
        save_path = os.path.join(self.checkpoint_dir, "train_vae_{}".format(global_id))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save(global_id=global_id - 1, model_name='IGNITE', checkpoint_dir=save_path)
        print('finished model training')

