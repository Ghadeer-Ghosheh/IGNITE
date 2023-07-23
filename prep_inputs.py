# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:45:54 2023

@author: gghos
"""

#from miss_utils import *
from bottleneck import push
import numpy as np
from bottleneck import push
from fancyimpute import IterativeImputer
import pycorruptor as corruptor

def create_individualized_missingness_mask2(mask):
  np.set_printoptions(suppress=False, precision= 9)
  samples_len =mask.shape[0]
  time_steps = mask.shape[1]
  features = mask.shape[2]
  
  personalized_mask_full = np.empty(shape=[samples_len,time_steps,features])
  personalized_mask_patient = []
  personalized_mask_sample = np.ones(shape=[time_steps,features])
  for patient_mask in mask:
        num_measurments_per_feature = patient_mask.sum(axis=0)
        # for each patient mask
        non_miss_featue= sum(num_measurments_per_feature !=0)

       
        tf=((num_measurments_per_feature)/time_steps)
     
        idf = np.log(features/non_miss_featue)
        if non_miss_featue == 0:
            idf= 1
        personalized_mask_patient.append(np.where(patient_mask == 0, tf*idf, patient_mask))
    # stack all feature-specific patient masks tnto a 3d tensor
  personalized_mask_full = np.stack(personalized_mask_patient, axis=0)
  return(personalized_mask_full)

def gen_input_noise(num_sample, T, noise_dim):
        return np.random.uniform(size=[num_sample, T, noise_dim])
    
def sum_nan_arrays(o,n):
       m =np.isnan(o)*1
       a = ((1-m)*o)
       b = ((m) *n)
       ma = np.isnan(a)
       mb = np.isnan(b)
       return np.where(ma&mb, np.nan, np.where(ma,0,a) + np.where(mb,0,b))
   
def create_masks(data, indicate_rate):
    miss=~np.isnan(data)*1
    new, indicate_mask,_=introduce_miss_patient(data,miss, indicate_rate,42 )
    miss=~np.isnan(new)*1
    mask_personalized = create_individualized_missingness_mask2(miss)
    zero=np.where(np.isnan(push(new, axis = 1)), 0 , push(new, axis = 1))
    noise= gen_input_noise(new.shape[0],new.shape[1],new.shape[2])
    noise_input= sum_nan_arrays(new,noise)
    IMM_input =push(new, axis=1)
    IMM_input= np.where(np.isnan(IMM_input), 0, IMM_input)
    IMM_input= np.multiply(noise_input,mask_personalized)
    return(miss, mask_personalized, zero, noise_input, IMM_input,indicate_mask)


def input_impute(data):
    miss=~np.isnan(data)*1
    mask_personalized = create_individualized_missingness_mask2(miss)
    zero=np.where(np.isnan(push(data, axis = 1)), 0 , push(data, axis = 1))
    noise= gen_input_noise(data.shape[0],data.shape[1],data.shape[2])
    noise_input= sum_nan_arrays(data,noise)
    IMM_input =push(data, axis=1)
    back_forward= np.where(np.isnan(IMM_input), 0, IMM_input)
    IMM_input= np.multiply(back_forward,mask_personalized)
    return(miss, mask_personalized, zero, noise_input, IMM_input)


def create_splits(split_ratio, X, conditions, outcomes):
    split=int(len(X)*split_ratio)
    X_training, X_test = X[:split,:,:], X[split:,:,:]
    outcomes_training, outcomes_testing = outcomes[:split],outcomes[split:]
    conditions=np.where(np.isnan(conditions), 0, conditions)
    conditions_training, conditions_test = conditions[:split_ratio,:,:], conditions[split_ratio:,:,:]
    return( X_training, X_test ,  conditions_training, conditions_test, outcomes_training, outcomes_testing  )


def get_split_test(split_ratio, X):
    split=int(len(X)*split_ratio)
    return(X[split:,:,:])


def get_impuation(X_test):
    # returns the imputations for the test data
    LO =push(X_test, axis=1)
    LOCF= np.where(np.isnan(LO), 0, LO) 
    zero= np.where(np.isnan(X_test), 0, X_test)
    means = []
    flatten= X_test.reshape(X_test.shape[0]*X_test.shape[1], X_test.shape[2])
    for i in range(flatten.shape[1]):
        means.append(np.nanmean(flatten[:,i]))
    mean_imputed =mean_fill(X_test, means)
        
    mice_impute = IterativeImputer()
    flatten= X_test.reshape(X_test.shape[0]*X_test.shape[1], X_test.shape[2])
    MICE_ = mice_impute.fit_transform(flatten)
    MICE=MICE_.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
    MICE =MICE.copy()
    return(LOCF, zero, mean_imputed, MICE)




def introduce_miss_patient(patient_data,patient_masks, miss_ratio, random_seed):
    np.random.seed(random_seed)
    df_list, df_stacked, masks_w_miss,masks_stacked,miss_indices_list, stack_miss_i  = [],[], [],[], [], []
    for mask_patient_, data_patient in zip(patient_masks, patient_data):
        # reshape masks and patients
        mask_patient = mask_patient_.reshape(-1).copy()
        data_patient_reshaped= data_patient.reshape(-1).copy()

        obs_indices = np.where(mask_patient)[0].tolist()
        miss_indices = np.random.choice(obs_indices, (int)(len(obs_indices) * miss_ratio), replace=False) # select % of new patient_missingness
        if len(miss_indices) != 0:
            data_patient_reshaped[miss_indices] = np.nan
        data_patient_reshaped= data_patient_reshaped.reshape(48, 35)
        for i in miss_indices:
            mask_patient[i] = 0
        mask_patient = mask_patient.reshape(mask_patient_.shape[0],mask_patient_.shape[1])
        masks_w_miss.append(mask_patient)
        df_list.append(data_patient_reshaped)
        miss_indices_list.append(miss_indices)
        
    df_stacked= np.stack(df_list, axis = 0)
    masks_stacked = np.stack(masks_w_miss, axis = 0)
    return df_stacked,masks_stacked, miss_indices_list



def renormalize(data, means, stds):
    renorms=[]
    for patient in data:
        r= patient*stds+ means
        renorms.append(r)
    renorm_full = np.stack(renorms, axis=0)
    return(renorm_full)
def normalize(data, mins, maxs):

    renorms=[]

    for patient in data:

        r= (patient-mins) / (np.array(maxs) - np.array(mins) + 1e-7)

        renorms.append(r)

    renorm_full = np.stack(renorms, axis=0)

    return(renorm_full)
def prepare_fills(df, means):
    # input for the IMM network (forward/backward fill then population mean if never observed)
    filled = np.flip(push(df, axis=1), axis=1)
    df2 =np.flip(push(filled, axis=1), axis = 1)
    for i in range(df.shape[2]):
        df2= np.where(np.isnan(df2), means[i], df2)
    return(df2)

def mean_fill(df, means):
    df_list = []
                # popultaion mean for each feature imputation
    for i in range(df.shape[2]):
        df_ = df[:,:,i]
        df_= np.where(np.isnan(df_), means[i], df_)
        df_list.append(df_)
    stacked= np.stack(df_list, axis = 2)
    return(stacked)


def cal_missing_rate(X):
    return corruptor.cal_missing_rate(X)



def get_patient_level_MSE(imputation,observed, miss_indices_, name):
    array_mse = []
    for i in range(len(miss_indices_)):
        if len(miss_indices_[i])> 0:
            predictedMatrix = imputation.reshape(-1)[miss_indices_[i]]   
            target = observed[i].reshape(-1)[miss_indices_[i]]                 
            mse=((target- predictedMatrix) ** 2).mean(axis=None)
        else:
            mse = 0.0
        array_mse.append(mse)
    print(name, np.mean(array_mse), np.std(array_mse))

  
def get_patient_level_RMSE(imputation,observed, miss_indices_, name):
    array_rmse = []
    for i in range(len(miss_indices_)):
        if len(miss_indices_[i])> 0:
            predictedMatrix = imputation.reshape(-1)[miss_indices_[i]]   
            target = observed[i].reshape(-1)[miss_indices_[i]]
                     
            mse=((target- predictedMatrix) ** 2).mean(axis=None)
            rms=np.sqrt(mse)
        else:
            rms= 0.0
        array_rmse.append(rms)
    print(name, np.mean(array_rmse), np.std(array_rmse))

def get_patient_level_MAE(imputation,observed, miss_indices_, name):
    array_mea = []
    for i in range(len(miss_indices_)):
        if len(miss_indices_[i])> 0:
             predictedMatrix = imputation.reshape(-1)[miss_indices_[i]]   
             target = observed[i].reshape(-1)[miss_indices_[i]]
                
             mea = np.mean(abs(target - predictedMatrix), axis=None)
        else:
            mea= 0.0
        array_mea.append(mea)
    print(name, np.mean(array_mea),np.std(array_mea))
       