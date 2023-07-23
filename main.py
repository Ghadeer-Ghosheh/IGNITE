# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 22:13:27 2023

@author: gghos
"""
import tensorflow.compat.v1 as tf

tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
import os
import random as rn
import pickle
from new_networks_DA import observed_only_vae,IMM_vae
from IGNITE_model import IGNITE
from prep_inputs import create_masks, introduce_miss_patient
import argparse
import numpy as np

os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "True"
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '0'
tf.compat.v1.disable_v2_behavior()
rn.seed(42)
np.random.seed(42)


def main (args, X, conditions, outcomes):
    
    # train_test_split
    sample_size= len(X)
    split_ratio=int(sample_size*0.8)
    X_training, X_test = X[:split_ratio,:,:], X[split_ratio:,:,:]
    outcomes_training, outcomes_testing = outcomes[:split_ratio],outcomes[split_ratio:]
    conditions=np.where(np.isnan(conditions), 0, conditions)

    conditions_training, conditions_test = conditions[:split_ratio,:,:], conditions[split_ratio:,:,:]

    # test MCAR reconstruction task

    if args.miss_test == True:
        mask=~np.isnan(X_test)*1
        X_test, new_mask,miss_indices=introduce_miss_patient(X_test,mask, args.miss_test_ratio,args.seed )


    
    # get the masks
    miss, mask_personalized, zero, noise_input, IMM_input,indicate_mask=create_masks(X_training, indicate_rate= args.indicate_rate)

    # define sizes
    time_steps = X_training.shape[1]
    ts_feat= X_training.shape[2]
    num_labels = conditions_training.shape[2]
    
    
    # initialize inputs
    observed_only_vae_x = zero
    IMM_vae_x= IMM_input
    intervention=conditions_training
    
    
    # pass inputs to model instances
    observed_only_vae_inst = observed_only_vae( time_steps=time_steps, dim=ts_feat, z_dim=args.shared_latent_dim, keep_prob=args.keep_prob,
                      conditional=args.conditional, enc_size=args.enc_size, dec_size= args.dec_size, l2scale= args.l2_scale, seed= args.seed)
    
    IMM_vae_inst = IMM_vae(time_steps=time_steps, dim=ts_feat, z_dim=args.shared_latent_dim, keep_prob=args.keep_prob,
                      conditional=args.conditional,enc_size=args.enc_size, dec_size= args.dec_size, l2scale= args.l2_scale, seed= args.seed)
  
    #wandb.init(....,sync_tensorboard=True,settings=dict(start_method='thread'), config = args)
    
    checkpoint_dir = "data/checkpoint/"
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    
    tf.reset_default_graph()
    model = IGNITE(  batch_size=args.batch_size,
                          time_steps=time_steps,
                          num_epochs=args.num_epochs,
                          oo_vae_dim=ts_feat, 
                          checkpoint_dir=checkpoint_dir,
                          z_size=args.shared_latent_dim, observed_only_data_sample=observed_only_vae_x,
                          observed_only_vae=observed_only_vae_inst, 
                          imm_vae_dim=ts_feat, 
                          IMM_mask= mask_personalized,
                          IMM_data_sample=IMM_vae_x,
                          indicating_mask_sample = indicate_mask, 
                          IMM_vae=IMM_vae_inst, 
                          enc_size = args.enc_size, dec_size = args.dec_size, 
                          outcomes = outcomes_training,
                          alpha_re=args.alpha_re, alpha_kl=args.alpha_kl,
                          alpha_contrastive=args.alpha_contrastive, alpha_matching=args.alpha_matching,
                          alpha_discrim=args.alpha_discrim, alpha_semantic=args.alpha_semantic,
                          alpha_MIT=args.alpha_MIT,
                         IGNITE_lr= args.IGNITE_lr,keep_prob = args.keep_prob,
                         binary_mask_data_sample=miss,
                          conditional=args.conditional, 
                           num_labels=num_labels,
                           experiment_name=args.experiment_name,
                          interventions=intervention)
    model.build()
    model.train() # train the model using the training data
    model.test(X_test,conditions_test, "test") # test the model on the test set to calculate the reconstruction loss for the MCAR experiments
    model.test(X,conditions, "full")






if __name__ == '__main__':  
    exp_name = "Final_test_IMM_input_missing_0.1_1024_onlytop2"

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=512, help='The batch size for training the model.')
    parser.add_argument('--num_epochs', type=int, default=30, help='The number of epoches in training  IGNITE.')
    parser.add_argument('--shared_latent_dim', type=int, default=10, help='The dimension of latent space in training  IGNITE.')
    parser.add_argument('--IGNITE_lr', type=float, default=0.0001, help='The learning rate in training IGNITE.')
    parser.add_argument('--l2_scale', type=float, default=0.0001, help='The regularization parameter in training IGNITE.')
    parser.add_argument('--keep_prob', type=float, default=0.7, help='The dropout rate in training IGNITE.')
    parser.add_argument('--enc_size', type=float, default=64,help="The size of the decoders used in IGNITE")
    parser.add_argument('--dec_size', type=float, default=64, help= "The size of the decoders used in IGNITE")
    parser.add_argument('--conditional', type=bool, default=True, help= "Use conditional variant for timeseries interventions")
    parser.add_argument('--experiment_name',  default=exp_name, help= "The name of the experiment")
    parser.add_argument('--indicate_rate', type=float, default=0.3,help="The ratio of missingness indication for training")

    parser.add_argument('--miss_test', type=bool, default= True)
    parser.add_argument('--seed', type=int, default= 42)
    parser.add_argument('--miss_test_ratio', type=float, default= 0.1)

    parser.add_argument('--alpha_re', type=float, default= 2)
    parser.add_argument('--alpha_kl', type=float, default=0.05)
    parser.add_argument('--alpha_discrim', type=float, default=1)
    parser.add_argument('--alpha_semantic', type=float, default=1)
    parser.add_argument('--alpha_matching', type=float, default=0.5)
    parser.add_argument('--alpha_contrastive', type=float, default=0.001)
    parser.add_argument('--alpha_MIT', type=float, default=1)

    args = parser.parse_args() 
    
    
    # Load datasets
    data_path = 'extracts/'
    with open(os.path.join(data_path, 'normalized_combined.pkl'), 'rb') as f:
        X = pickle.load(f)
    with open(os.path.join(data_path, 'condition.pkl'), 'rb') as f:
        conditions = pickle.load(f)
    with open(os.path.join(data_path, 'out_combined.pkl'), 'rb') as f:
        outcomes = pickle.load(f)
  
    main(args, X, conditions, outcomes)
    