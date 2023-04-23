# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:08:37 2023

@author: gghos
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:06:50 2023

@author: gghos
"""
import os
import numpy as np
from downstream_eval import *

import pickle
from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import BRITS, SAITS, Transformer
from pypots.utils.metrics import cal_mae
data_path = 'extracts/'
with open(os.path.join(data_path, 'X_0.5_1234col_last.pkl'), 'rb') as f:

      X = pickle.load(f)
#with open(os.path.join(data_path, 'out_combined.pkl'), 'rb') as f:
#      outcomes = pickle.load(f)
missing_mask=~np.isnan(X)*1
#X,label=get_sets_feature_missingess2(X,outcomes, miss,0, 0.25)
  
#X_intact, X, missing_mask, indicating_mask = mcar(X, 0) # hold out 10% observed values as ground truth
X = masked_fill(X, 1 - missing_mask, np.nan)
print(X.shape)
# Model training. This is PyPOTS showtime. 
#saits = SAITS(n_steps=48, n_features=35, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=20)
saits= BRITS(n_steps=48, n_features=35,epochs=20, rnn_hidden_size = 108)
saits.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
imputation = saits.impute(X)  # impute the originally-missing values and artificially-missing values
print(imputation.shape)
with open('BRITS_0.5_1234_col_last.pkl', 'wb') as handle:
    pickle.dump(imputation, handle, protocol=pickle.HIGHEST_PROTOCOL)
#mae = cal_mae(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
