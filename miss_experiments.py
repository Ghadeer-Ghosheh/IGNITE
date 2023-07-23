# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:34:50 2023

@author: gghos
"""

import os
import numpy as np
import pickle
from prep_inputs import get_patient_level_MAE, get_patient_level_RMSE,get_patient_level_MSE, get_split_test, introduce_miss_patient, get_impuation

data_path = 'extracts/'
with open(os.path.join(data_path, 'normalized_combined.pkl'), 'rb') as f:
    X = pickle.load(f)
experiment_name = "imputedFinal_test_IMM_input_missing_0.1_1024_onlytop"
saved_path = 'data/'
IGNITE =  np.load(saved_path + experiment_name + '.npz')["imputed_data_oo"]
    
    
X_test= get_split_test(0.8, X)
mask=~np.isnan(X_test)*1

new_tensor, new_mask,miss_indices=introduce_miss_patient(X_test,mask, 0.1,42 )
LOCF, zero, mean, MICE= get_impuation(new_tensor)



print("RMSE")
print("----------------------")
get_patient_level_RMSE(LOCF, X_test, miss_indices, "LOCF") 
get_patient_level_RMSE(zero, X_test, miss_indices, "zero") 
get_patient_level_RMSE(mean, X_test, miss_indices, "mean")
get_patient_level_RMSE(MICE, X_test, miss_indices, "mice") 
#get_patient_level_RMSE(BRITS, normalized, indicating_mask, "BRITS")# 
#get_patient_level_RMSE(SAITS, normalized, indicating_mask, "SAITS")es)
get_patient_level_RMSE(IGNITE, X_test, miss_indices, "IGNITE") # cl


print("MAE")
print("----------------------")
get_patient_level_MAE(LOCF, X_test, miss_indices, "LOCF") 
get_patient_level_MAE(zero, X_test, miss_indices, "zero") # calculate mean absolute error on the ground truth (artificially-missing values)
get_patient_level_MAE(mean, X_test, miss_indices, "mean") # calculate mean absolute error on the ground truth (artificially-missing values)
get_patient_level_MAE(MICE, X_test, miss_indices, "mice") # calculate mean absolute error on the ground truth (artificially-missing values)
#get_patient_level_RMSE(BRITS, normalized, indicating_mask, "BRITS")# calculate mean absolute error on the ground truth (artificially-missing values)
#get_patient_level_RMSE(SAITS, normalized, indicating_mask, "SAITS") # calculate mean absolute error on the ground truth (artificially-missing values)
get_patient_level_MAE(IGNITE, X_test, miss_indices, "IGNITE") # cl



print("MSE")
print("----------------------")
get_patient_level_MSE(LOCF, X_test, miss_indices, "LOCF") 
get_patient_level_MSE(zero, X_test, miss_indices, "zero") # calculate mean absolute error on the ground truth (artificially-missing values)
get_patient_level_MSE(mean, X_test, miss_indices, "mean") # calculate mean absolute error on the ground truth (artificially-missing values)
get_patient_level_MSE(MICE, X_test, miss_indices, "mice") # calculate mean absolute error on the ground truth (artificially-missing values)
#get_patient_level_RMSE(BRITS, normalized, indicating_mask, "BRITS")# calculate mean absolute error on the ground truth (artificially-missing values)
#get_patient_level_RMSE(SAITS, normalized, indicating_mask, "SAITS") # calculate mean absolute error on the ground truth (artificially-missing values)
get_patient_level_MSE(IGNITE, X_test, miss_indices, "IGNITE") # cl


