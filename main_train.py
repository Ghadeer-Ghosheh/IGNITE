#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from networks import observed_only_vae,IMM_vae

from utils import ones_target, zeros_target
import numpy as np
from IGNITE_model import IGNITE
import timeit
start = timeit.default_timer()
from bottleneck import push


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Load datasets


data_path = 'physionet/'
with open(os.path.join(data_path, 'FP_2012_mask.pkl'), 'rb') as f:
    miss = pickle.load(f)
with open(os.path.join(data_path, 'FP_2012_LOCF.pkl'), 'rb') as f:
    LVCF = pickle.load(f)
with open(os.path.join(data_path, 'FP_2012_Intervention.pkl'), 'rb') as f:
    interventions = pickle.load(f)

with open(os.path.join(data_path, 'FP_2012_origional.pkl'), 'rb') as f:
    origional = pickle.load(f)
zero=np.where(np.isnan(push(origional, axis = 1)), 0 , push(origional, axis = 1))



with open(os.path.join(data_path, 'person.pkl'), 'rb') as f:
    mask_personalized = pickle.load(f)


sample_size= len(LVCF)
intervention = np.asarray(interventions)[:sample_size]
IMM_vae_x = np.multiply(LVCF[:sample_size],mask_personalized[:sample_size])
observed_only_vae_x = zero[:sample_size]
miss_sample_x = miss[:sample_size]
intervention=np.where(np.isnan(intervention), 0, intervention)

batch_size = 400
time_steps = observed_only_vae_x.shape[1]
num_pre_epochs = 200

shared_latent_dim = 10
z_size = shared_latent_dim
ts_feat= observed_only_vae_x.shape[2]
conditional = True
num_labels = 1
observed_only_vae = observed_only_vae(batch_size=batch_size, time_steps=time_steps, dim=ts_feat, z_dim=z_size,
                  conditional=conditional, num_labels=num_labels)

IMM_vae = IMM_vae(batch_size=batch_size, time_steps=time_steps, dim=ts_feat, z_dim=z_size,
                  conditional=conditional, num_labels=num_labels)


checkpoint_dir = os.path.join("data/checkpoint/")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


exp_name = "IGNITE_dual_single_contrastive_400_treat_dot_IMM_35_001_001_0.8_discriminator"

tf.compat.v1.reset_default_graph()
run_config = tf.compat.v1.ConfigProto()
with tf.compat.v1.Session(config = run_config) as sess:
        model = IGNITE(sess=sess,
                      batch_size=batch_size,
                      time_steps=time_steps,
                      num_pre_epochs=num_pre_epochs,
                      checkpoint_dir=checkpoint_dir,
                      oo_vae_dim=ts_feat, 
                      z_size=z_size, observed_only_data_sample=observed_only_vae_x,
                      observed_only_vae=observed_only_vae, 
                      imm_vae_dim=ts_feat, 
                      IMM_data_sample=IMM_vae_x,
                      IMM_vae=IMM_vae,
                     binary_mask_data_sample=miss_sample_x,experiment_name=exp_name,
                      conditional=conditional, 
                       num_labels=num_labels,
                      interventions=intervention)
        model.build()
        model.train()

stop = timeit.default_timer()
print('Time: ', stop - start)



