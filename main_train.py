

import wandb


import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import os
import pickle
import pandas as pd
from networks import observed_only_vae,IMM_vae
from utils import create_conditions
from IGNITE_model import IGNITE
import timeit
start = timeit.default_timer()
from bottleneck import push
import argparse


  
def main (args):

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
    data_path = 'extracts/'
    with open(os.path.join(data_path, 'mask_combined.pkl'), 'rb') as f:
        miss = pickle.load(f)
    with open(os.path.join(data_path, 'LOCV_combined.pkl'), 'rb') as f:
        LVCF = pickle.load(f)
    with open(os.path.join(data_path, 'interv_combined.pkl'), 'rb') as f:
        interventions = pickle.load(f)
    
    with open(os.path.join(data_path, 'normalized_combined.pkl'), 'rb') as f:
        origional = pickle.load(f)
    zero=np.where(np.isnan(push(origional, axis = 1)), 0 , push(origional, axis = 1))
    
    
    with open(os.path.join(data_path, 'IMM_combined.pkl'), 'rb') as f:
        mask_personalized = pickle.load(f)
    
    with open(os.path.join(data_path, 'out_combined.pkl'), 'rb') as f:
        outcomes = pickle.load(f)
    
    with open(os.path.join(data_path, 'static_set_a.pkl'), 'rb') as f:
        static_a = pickle.load(f)
        
    with open(os.path.join(data_path, 'static_set_b.pkl'), 'rb') as f:
        static_b = pickle.load(f)    
    
    with open(os.path.join(data_path, 'static_set_c.pkl'), 'rb') as f:
        static_c = pickle.load(f)
        
    out_combined = pd.concat([static_a,static_b,static_c]) 
    

    intervention = np.asarray(interventions)
    intervention=np.where(np.isnan(intervention), 0, intervention)

    
    interventions_3d=create_conditions([intervention],out_combined)
    
    sample_size= len(LVCF)
    intervention = interventions_3d[:sample_size]
    IMM_vae_x = np.multiply(LVCF[:sample_size],mask_personalized[:sample_size])
    observed_only_vae_x = zero[:sample_size]
    miss_sample_x = miss[:sample_size]
    intervention=np.where(np.isnan(intervention), 0, intervention)
    
    
    time_steps = observed_only_vae_x.shape[1]

    ts_feat= observed_only_vae_x.shape[2]
    num_labels = interventions_3d.shape[2]
    observed_only_vae_inst = observed_only_vae(batch_size=args.batch_size, time_steps=time_steps, dim=ts_feat, z_dim=args.shared_latent_dim,
                      conditional=args.conditional, num_labels=num_labels)
    
    IMM_vae_inst = IMM_vae(batch_size=args.batch_size, time_steps=time_steps, dim=ts_feat, z_dim=args.shared_latent_dim,
                      conditional=args.conditional, num_labels=num_labels)
    
    
    checkpoint_dir = os.path.join("data/checkpoint/")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    
    exp_name = "base_IGNITE_no_OO)_plus_treat_1_1_1"
    #wandb.init(project="ignite", entity="..",sync_tensorboard=True,settings=dict(start_method='thread'), config = args)
    
    tf.reset_default_graph()
    
    run_config = tf.ConfigProto()
    with tf.Session(config = run_config) as sess:
            model = IGNITE(sess=sess,
                          batch_size=args.batch_size,
                          time_steps=time_steps,
                          num_epochs=args.num_epochs,
                          checkpoint_dir=checkpoint_dir,
                          oo_vae_dim=ts_feat, 
                          z_size=args.shared_latent_dim, observed_only_data_sample=observed_only_vae_x,
                          observed_only_vae=observed_only_vae_inst, 
                          imm_vae_dim=ts_feat, 
                          IMM_data_sample=IMM_vae_x,
                          IMM_vae=IMM_vae_inst, 
                          outcomes = outcomes,
                          alpha_re=args.alpha_re, alpha_kl=args.alpha_kl, alpha_mt=args.alpha_mt, 
                          alpha_ct=args.alpha_ct, alpha_discrim=args.alpha_discrim,
                         IGNITE_lr= args.IGNITE_lr,
                         binary_mask_data_sample=miss_sample_x,experiment_name=exp_name,
                          conditional=args.conditional, 
                           num_labels=num_labels,
                          interventions=intervention)
            model.build()
            model.train()
    
    stop = timeit.default_timer()
    print('Time: ', stop - start)



if __name__ == '__main__':  

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=400, help='The batch size for training the model.')
    parser.add_argument('--num_epochs', type=int, default=100, help='The number of epoches in training  IGNITE.')
    parser.add_argument('--shared_latent_dim', type=int, default=9, help='The dimension of latent space in training  IGNITE.')

    parser.add_argument('--IGNITE_lr', type=float, default=0.001)

    parser.add_argument('--alpha_re', type=float, default=1)
    parser.add_argument('--alpha_kl', type=float, default=1)
    parser.add_argument('--alpha_mt', type=float, default=0.005)
    parser.add_argument('--alpha_ct', type=float, default=0.0005)
    parser.add_argument('--alpha_discrim', type=float, default=0.25)

    parser.add_argument('--conditional', type=bool, default=True)

    args = parser.parse_args() 
  
    main(args)