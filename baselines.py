# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:06:50 2023

@author: gghos
"""
import os
import numpy as np
from prep_inputs import introduce_miss_patient,get_sets_sample_missingness,get_sets_feature_missingness

import pickle
from pypots.data import masked_fill
from pypots.imputation import BRITS, SAITS, Transformer

data_path = 'extracts/'
with open(os.path.join(data_path, 'normalized_combined.pkl'), 'rb') as f:
    X = pickle.load(f)
with open(os.path.join(data_path, 'condition.pkl'), 'rb') as f:
       conditions = pickle.load(f)
with open(os.path.join(data_path, 'out_combined.pkl'), 'rb') as f:
       outcomes = pickle.load(f)
    
X,conditions,outcomes=\
         get_sets_feature_missingness(X,conditions,outcomes, 0.25,0.75) # missingness ratio extact 
# train_test_split
sample_size= len(X)
split_ratio=int(sample_size*0.8)
X_training, X_test = X[:split_ratio,:,:], X[split_ratio:,:,:]

mask=~np.isnan(X_test)*1
#X_test, miss_indices, indicating_mask=introduce_miss_patient(X_test,mask, 0.2,42 )
#miss_new = ~np.isnan(X_test)*1

#X_test = masked_fill(X_test, 1 - miss_new, np.nan)
#print(X.shape)


Transform = Transformer(n_steps = 48, n_features= 35, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=10)
Transform.fit(X_training)  

'''
imputation = Transform.impute(X_test) 

with open('Transformer_0.2_MCAR.pkl', 'wb') as handle:
    pickle.dump(imputation, handle, protocol=pickle.HIGHEST_PROTOCOL)
 '''   
imputation = Transform.impute(X)  # for the downstream tasks
with open('Transformer_75_feature.pkl', 'wb') as handle:
    pickle.dump(imputation, handle, protocol=pickle.HIGHEST_PROTOCOL)


#####################################################

brits= BRITS(n_steps=48, n_features=35,epochs=10, rnn_hidden_size = 108)
brits.fit(X_training)  
'''
imputation = brits.impute(X_test) 
print(imputation.shape)
with open('BRITS_0.2_MCAR.pkl', 'wb') as handle:
    pickle.dump(imputation, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
imputation = brits.impute(X)  # for the downstream tasks
print(imputation.shape)
with open('BRITS_X_75_feature.pkl', 'wb') as handle:
    pickle.dump(imputation, handle, protocol=pickle.HIGHEST_PROTOCOL)

#####################################################
#####################################################

saits = SAITS(n_steps=48, n_features=35, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=10)
saits.fit(X_training) 
'''
imputation = saits.impute(X_test)
print(imputation.shape)
with open('SAITS_0.2_MCAR.pkl', 'wb') as handle:
    pickle.dump(imputation, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''

imputation = saits.impute(X)  # for the downstream tasks
print(imputation.shape)
with open('SAITS_X_75_feature.pkl', 'wb') as handle:
    pickle.dump(imputation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
