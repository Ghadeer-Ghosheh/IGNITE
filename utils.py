import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
def ones_target(size, min, max):
    return tf.random_uniform([size, 1], min, max, dtype=tf.float32)

def zeros_target(size, min, max):
    return tf.random_uniform([size, 1], min, max, dtype=tf.float32)

def target_flip(size, min, max, ratio):
    ori = np.random.uniform(low=min, high=max, size=[size, 1])
    index = np.random.choice(size, int(size * ratio), replace=False)
    ori[index] = 1 - ori[index]
    return tf.convert_to_tensor(ori, dtype=tf.float32)
def create_new_column(row):
      if  row['Age'] > 50:
          return 1
      else:
          return 0
def create_new_column_g(row):
      if  row['Gender'] == 1:
          return 1
      else:
          return 0
  
  
def create_x_matrix(x):
      return x.iloc[:, 2:].values
def create_individualized_missingness_mask(mask):
  np.set_printoptions(suppress=False, precision= 9)
  samples_len =mask.shape[0]
  time_steps = mask.shape[1]
  features = mask.shape[2]
  
  personalized_mask_full = np.empty(shape=[samples_len,time_steps,features])
  personalized_mask_patient = []
  for patient_mask in mask:
        num_measurments_per_feature = patient_mask.sum(axis=0)
        # for each patient mask
        tf=((num_measurments_per_feature)/time_steps)
        personalized_mask_patient.append(np.where(patient_mask == 0, tf, patient_mask))
    # stack all feature-specific patient masks tnto a 3d tensor
  personalized_mask_full = np.stack(personalized_mask_patient, axis=0)
  return(personalized_mask_full)
  
def create_conditions(treat, static_data):
      dfs = []
      names = []
      names_2 = []
      static_data['age'] = static_data.apply(lambda w: create_new_column(w), axis=1)
      static_data['gender'] = static_data.apply(lambda w: create_new_column_g(w), axis=1)
  
      for i in range(len(static_data)):
          df = pd.DataFrame()
          for x in range(len(treat)):
              name = "intervention"+ str(x)
              df[name]=pd.Series(treat[x][i].reshape(48))
              names.append(name)
          df["Gender"] = static_data["gender"].iloc[i]
          df["Age"] = static_data["age"].iloc[i]
          df["ID"] = i
          for i in names:
              name_new = i+ "enc"
              name_new= pd.get_dummies(df[i],prefix=i,drop_first=True)
              names_2.append(name_new)
          df_oh = pd.concat(names_2, axis = 1)
          df_final = pd.concat([df["ID"],df["Gender"],df["Age"],df_oh], axis =1 )
          dfs.append(df_final)
          names = []
          names_2 = []
      combined=pd.concat(dfs).reset_index().set_index(["ID","index"]).fillna(0)
      combined=combined.astype('bool')*1
      result=combined.apply(lambda r:str(''.join(str(r[col]) for col in combined.columns)),axis=1)
      result=pd.get_dummies(result)
      interventions_3d = np.array(list(result.reset_index().groupby("ID").apply(create_x_matrix)))
      return(interventions_3d)
  
def new_static(static_data, timesteps):
    static_data['age'] = static_data.apply(lambda w: create_new_column(w), axis=1)
    static_data['gender'] = static_data.apply(lambda w: create_new_column_g(w), axis=1)

    combined = static_data[["gender", "age"]]
    result=combined.apply(lambda r:str(''.join(str(r[col]) for col in combined.columns)),axis=1)
    result=pd.get_dummies(result)
    repeated_encoding = np.stack([result]*timesteps, axis=1)
    return (repeated_encoding)