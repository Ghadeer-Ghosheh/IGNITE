import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()

import os

from Contrastivelosslayer import nt_xent_loss
from utils import ones_target, zeros_target
import numpy as np
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

class full_model(object):
    def __init__(self, sess,
                 # -- shared params:
                 batch_size, time_steps,
                 num_pre_epochs, 
                #  num_epochs,
                 checkpoint_dir,
                 # -- params for continuous-GAN
                 c_dim, 
                #  c_noise_dim,
                 c_z_size, c_data_sample,
                #  c_gan,
                 c_vae,
                 # -- params for discrete-GAN
                 d_dim, 
                #  d_noise_dim,
                 d_z_size, d_data_sample,
               #  d_gan,
                 d_vae, 
                #  PM_data_sample,
                 mask_data_sample,experiment_name,
                 # -- label information
                 conditional=False, num_labels=0,
                 statics_label=None):

        self.sess = sess
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.num_pre_epochs = num_pre_epochs
        self.checkpoint_dir = checkpoint_dir
        self.statics_label = statics_label
        self.experiment_name = experiment_name

        # params for continuous-GAN
        self.c_dim = c_dim
        # self.c_noise_dim = c_noise_dim
        self.c_z_size = c_z_size
        self.c_data_sample = c_data_sample
        self.c_rnn_vae_net = c_vae
        # self.cgan = c_gan

        # params for discrete-GAN
        self.d_dim = d_dim
        # self.d_noise_dim = d_noise_dim
        self.d_z_size = d_z_size
        self.d_data_sample = d_data_sample
        self.d_rnn_vae_net = d_vae
       # self.discriminator = d_gan

        # self.PM_data_sample = PM_data_sample
        self.mask_data_sample = mask_data_sample

        # params for label information
        self.num_labels = num_labels
        self.conditional = conditional
        self.name = "logs/last/"+self.experiment_name


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
                c1 = self.statics_label[i, 0]   # patient status
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
    
    def compare_plot(self,c_real_data_pl, c_decoded_output,d_rec_data, epoch):

        sns.set_style("whitegrid")
    
        fig, axes = plt.subplots(1, 19, figsize=(200, 10)) 

        # Set the ticks and ticklabels for all axes
        plt.setp(axes, xticks= np.arange(49, step = 1))
        num_plot = 1
    
        """continuous dim"""
        c_dim_list = []
        c_dim_list += list(range(c_real_data_pl.shape[2]))
        
        c_pid_index = random.sample(list(range(c_real_data_pl.shape[0])), num_plot)  # same index
      
        c_pid_index = [90]
        cols = ['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'TroponinT']
        # cols2= [ 'Mg', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight', 'pH]

       
        for i, b in zip(range(len(c_dim_list)), cols):
            df = pd.DataFrame(c_real_data_pl[c_pid_index, :, c_dim_list[i]])
            df2 = pd.DataFrame(c_decoded_output[c_pid_index, :, c_dim_list[i]])
            df3 = pd.DataFrame(d_rec_data[c_pid_index, :, c_dim_list[i]])
            # df4 = pd.DataFrame(d_real_data[c_pid_index, :, c_dim_list[i]])
            # df5 = pd.DataFrame(c_real_data[c_pid_index, :, c_dim_list[i]])
            
            axes[i].set_title(b)
            axes[i].plot(df.T, 'o-', color='black', label=i)
            axes[i].plot(df2.T,'o-',color='purple', label=i,alpha=0.5) 
            axes[i].plot(df3.T,'o-', color='green', label=i,alpha=0.5)
                # axes[i].plot(df4.T,'o-', color='blue', label=i, alpha=0.2)
                # axes[i].plot(df5.T, 'o-', color='blue', label=i, alpha=0.3)
            axes[i].legend(["Original",'VAE_observed_ELBO',"ours"])

        if not os.path.exists('plots/'+self.experiment_name):
            os.makedirs('plots/'+self.experiment_name)
        fig.savefig(os.path.join('plots/'+self.experiment_name+"/"+self.experiment_name+"_"+str(epoch)), format='png')
        plt.close(fig)
        
    def build_tf_graph(self):
        # Step 1: VAE pretraining 
        if self.conditional:
            self.real_data_label_pl = tf.placeholder(
                dtype=float, shape=[self.batch_size,self.time_steps, self.num_labels], name="real_data_label")
                
        self.c_real_data_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.c_dim], name="continuous_real_data")
        self.c_real_data_mask_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.c_dim], name="continuous_real_mask")


        if self.conditional:
            self.c_decoded_output, self.c_vae_sigma, self.c_vae_mu, self.c_vae_logsigma, self.c_enc_z = \
                self.c_rnn_vae_net.build_vae(self.c_real_data_pl, self.real_data_label_pl)
        else:
            self.c_decoded_output, self.c_vae_sigma, self.c_vae_mu, self.c_vae_logsigma, self.c_enc_z = \
                self.c_rnn_vae_net.build_vae(self.c_real_data_pl)


        self.d_real_data_pl = tf.placeholder(
            dtype=float, shape=[self.batch_size, self.time_steps, self.d_dim], name="real_personalized")
        if self.conditional:
            self.d_decoded_output, self.d_vae_sigma, self.d_vae_mu, self.d_vae_logsigma, self.d_enc_z = \
                self.d_rnn_vae_net.build_vae(self.d_real_data_pl, self.real_data_label_pl)
        else:
            self.d_decoded_output, self.d_vae_sigma, self.d_vae_mu, self.d_vae_logsigma, self.d_enc_z = \
                self.d_rnn_vae_net.build_vae(self.d_real_data_pl)

        # self.imputed, self.imputed_prob = self.discriminator.build_Discriminator(self.d_decoded_output)
    

    def build_loss(self):

        #################
        # (1) VAE loss  #
        #################
        alpha_re = 1
        alpha_kl = 0.5
        alpha_mt = 0.1
        alpha_ct =0.1

        x_latent_1 = tf.stack(self.c_enc_z, axis=1)
        x_latent_2 = tf.stack(self.d_enc_z, axis=1)
        self.vae_matching_loss = tf.losses.mean_squared_error(x_latent_1, x_latent_2)

        self.vae_contra_loss = nt_xent_loss(tf.reshape(x_latent_1, [x_latent_1.shape[0], -1]),
                                            tf.reshape(x_latent_2, [x_latent_2.shape[0], -1]), self.batch_size)

                    
        # HERE
        m_mask = tf.cast(self.c_real_data_mask_pl, tf.bool)
        masked_c_real_data_pl = tf.where(m_mask, self.c_real_data_pl, tf.zeros_like(self.c_real_data_pl))  
        masked_c_decoded_output = tf.where(m_mask, self.c_decoded_output, tf.zeros_like(self.c_decoded_output))  

        self.c_re_loss = tf.losses.mean_squared_error(masked_c_real_data_pl, masked_c_decoded_output)
    
        c_kl_loss = [0] * self.time_steps 
        for t in range(self.time_steps):
            c_kl_loss[t] = 0.5 * (tf.reduce_sum(self.c_vae_sigma[t], 1) + tf.reduce_sum(
                tf.square(self.c_vae_mu[t]), 1) - tf.reduce_sum(self.c_vae_logsigma[t] + 1, 1))
        
        self.c_kl_loss = tf.reduce_mean(tf.add_n(c_kl_loss))
        if self.conditional:
            self.c_vae_loss = alpha_re*self.c_re_loss + \
                            alpha_kl*self.c_kl_loss+ \
                            alpha_mt* self.vae_matching_loss + \
                            alpha_ct*self.vae_contra_loss  
                            # + \
                            # self.vae_semantics_loss
        else:
            self.c_vae_loss = alpha_re*self.c_re_loss + \
                            alpha_kl*self.c_kl_loss 
                            #+ \
                            #alpha_mt* self.vae_matching_loss + \
                            #alpha_ct*self.vae_contra_loss


        # start discrete losses

        # np.divide(self.d_decoded_output,self.d_real_data_personalized_mask_pl)
        
        self.d_re_loss = alpha_re* tf.losses.mean_squared_error(self.d_real_data_pl, self.d_decoded_output)
        d_kl_loss = [0] * self.time_steps
        for t in range(self.time_steps):
            d_kl_loss[t] = 0.5 * (tf.reduce_sum(self.d_vae_sigma[t], 1) + tf.reduce_sum(
                tf.square(self.d_vae_mu[t]), 1) - tf.reduce_sum(self.d_vae_logsigma[t] + 1, 1))
        self.d_kl_loss = alpha_kl * tf.reduce_mean(tf.add_n(d_kl_loss))

        if self.conditional:
            self.d_vae_loss = alpha_re*self.d_re_loss + \
                            alpha_kl*self.d_kl_loss + \
                             alpha_mt* self.vae_matching_loss + \
                            alpha_ct* self.vae_contra_loss 
                            #  + self.discriminator_loss
                            #  + \
                            #  self.vae_semantics_loss
        else:
            self.d_vae_loss = alpha_re*self.d_re_loss + \
                            alpha_kl*self.d_kl_loss + \
                               alpha_mt* self.vae_matching_loss + \
                               alpha_ct*self.vae_contra_loss 
                            #   + self.discriminator_loss

        # self.d_vae_valid_loss = tf.losses.mean_squared_error(self.d_vae_test_data_pl, self.d_vae_test_decoded)
  
  


        #######################
        # Optimizer           #
        #######################
        t_vars = tf.trainable_variables()
        c_vae_vars = [var for var in t_vars if 'Continuous_VAE' in var.name]
        d_vae_vars = [var for var in t_vars if 'Discrete_VAE' in var.name]
        s_vae_vars = [var for var in t_vars if 'Shared_VAE' in var.name]
       
        self.c_v_op_pre = tf.train.AdamOptimizer(learning_rate=0.0005)\
            .minimize(self.c_vae_loss, var_list=c_vae_vars+s_vae_vars)

        self.d_v_op_pre = tf.train.AdamOptimizer(learning_rate=0.0005)\
            .minimize(self.d_vae_loss, var_list=d_vae_vars+s_vae_vars)


    def build_summary(self):
        print("finalize")
        self.c_vae_summary = []
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/reconstruction_loss", self.c_re_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/kl_divergence_loss", self.c_kl_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/matching_loss", self.vae_matching_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/contrastive_loss", self.vae_contra_loss))
        # if self.conditional:
        #     self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/semantic_loss", self.vae_semantics_loss))
        self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/vae_loss", self.c_vae_loss))
        # self.c_vae_summary.append(tf.summary.scalar("C_VAE_loss/validation_loss", self.c_vae_valid_loss))
        self.c_vae_summary = tf.summary.merge(self.c_vae_summary)

        self.d_vae_summary = []
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/reconstruction_loss", self.d_re_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/kl_divergence_loss", self.d_kl_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/matching_loss", self.vae_matching_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/contrastive_loss", self.vae_contra_loss))
        # self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/discriminator_loss", self.discriminator_loss))
        # if self.conditional:
        #     self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/semantic_loss", self.vae_semantics_loss))
        self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/vae_loss", self.d_vae_loss))
        # self.d_vae_summary.append(tf.summary.scalar("D_VAE_loss/validation_loss", self.d_vae_valid_loss))
        self.d_vae_summary = tf.summary.merge(self.d_vae_summary)
 
    def train(self):
        self.summary_writer = tf.summary.FileWriter(self.name, self.sess.graph)

       
        continuous_x = self.c_data_sample[: int(self.c_data_sample.shape[0]), :, :]
        # continuous_x_test = self.c_data_sample[int(0.9 * self.c_data_sample.shape[0]) : , :, :]
        # continuous_x = strategy.experimental_distribute_dataset(continuous_x)
        # continuous_x_test = strategy.experimental_distribute_dataset(continuous_x_test)
        
        # PM_x = self.PM_data_sample[: int(self.PM_data_sample.shape[0]), :, :]
        mask_x = self.mask_data_sample[: int(self.mask_data_sample.shape[0]), :, :]
        # mask_x_test = self.mask_data_sample[int(0.9 * self.mask_data_sample.shape[0]) : , :, :]
        
        discrete_x = self.d_data_sample[: int(self.d_data_sample.shape[0]), :, :]
        # discrete_x_test = self.d_data_sample[int(0.9 * self.d_data_sample.shape[0]):, :, :]
        
        # discrete_x = strategy.experimental_distribute_dataset(discrete_x)
        # discrete_x_test = strategy.experimental_distribute_dataset(discrete_x_test)

        if self.conditional:
            # label_data = self.statics_label[: int(self.d_data_sample.shape[0]), :]
            label_data = self.statics_label[: int(self.d_data_sample.shape[0]), :, :]
        data_size = continuous_x.shape[0]
        num_batches = math.ceil(data_size // self.batch_size)
        print("data_size", data_size)

        tf.global_variables_initializer().run()

        # pretrain step
        print('start training')
        global_id = 0

        for pre in range(self.num_pre_epochs):

            # random_idx = np.random.permutation(data_size)
            continuous_x_random = continuous_x
            discrete_x_random = discrete_x
            mask_x_random = mask_x
            # PM_x_random = PM_x
            if self.conditional:
                label_data_random = label_data
            # print(label_data_random.get_shape())

            # random_idx_ = np.random.permutation(continuous_x_test.shape[0])
            # continuous_x_test_batch = continuous_x_test[:self.batch_size, :, :]
            # discrete_x_test_batch = discrete_x_test[:self.batch_size, :, :]
            # mask_x_test_test_batch = mask_x_test[:self.batch_size, :, :]


            print("VAE epoch %d" % pre)

            c_real_data_lst = []
            c_rec_data_lst = []
            d_real_data_lst = []
            d_rec_data_lst = []
            imputed_data_lst = []
            mask_lst = []

            for b in range(num_batches):

                feed_dict = {}
                feed_dict[self.c_real_data_pl] = continuous_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                # feed_dict[self.c_vae_test_data_pl] = continuous_x_test_batch
                feed_dict[self.d_real_data_pl] = discrete_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                # feed_dict[self.d_vae_test_data_pl] = discrete_x_test_batch
                feed_dict[self.c_real_data_mask_pl] = mask_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                # feed_dict[self.d_real_data_personalized_mask_pl] = PM_x_random[b * self.batch_size: (b + 1) * self.batch_size]

                # feed_dict[self.c_real_data_mask_test_pl] = mask_x_test_test_batch
                if self.conditional:
                    feed_dict[self.real_data_label_pl] = label_data_random[b * self.batch_size: (b + 1) * self.batch_size]

                summary_result, _ = self.sess.run([self.c_vae_summary, self.c_v_op_pre], feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_result, global_id)
                summary_result, _ = self.sess.run([self.d_vae_summary, self.d_v_op_pre], feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_result, global_id)
                
               


                c_real_data, c_rec_data, d_mask = self.sess.run([self.c_real_data_pl, self.c_decoded_output, self.c_real_data_mask_pl], feed_dict=feed_dict)
                c_real_data_lst.append(c_real_data)
                c_rec_data_lst.append(c_rec_data)

                d_real_data, d_rec_data,d_mask= self.sess.run([self.d_real_data_pl, self.d_decoded_output, self.c_real_data_mask_pl], feed_dict=feed_dict)
                d_real_data_lst.append(d_real_data)
                d_rec_data_lst.append(d_rec_data)
     
                # mask_lst.append(d_mask)
                # imputed_data_lst.append(d_mask * c_real_data + (1-d_mask) * d_rec_data)
                assert not np.any(np.isnan(d_real_data))

                assert not np.any(np.isnan(d_rec_data))

                global_id += 1
                '''
                if (pre%50 == 0):
                    array = d_mask
                    array[array == 0] = np.nan
                    masked_plot=(c_real_data*array)
                    masked_plot[:c_rec_data.shape[2]]
                    self.compare_plot(masked_plot, c_rec_data,d_rec_data, pre)
                '''
            print(len(d_rec_data_lst))      
        np.savez('data/'+self.experiment_name+'.npz', c_real=np.vstack(c_real_data_lst), c_rec=np.vstack(c_rec_data_lst),
                                     d_real=np.vstack(d_real_data_lst), d_rec=np.vstack(d_rec_data_lst))
        # saving the pre-trained model
        save_path = os.path.join(self.checkpoint_dir, "pretrain_vae_{}".format(global_id))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save(global_id=global_id - 1, model_name='m3gan', checkpoint_dir=save_path)
        print('finished model training')

