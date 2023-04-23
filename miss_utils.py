import numpy as np
from bottleneck import push
import pycorruptor as corruptor


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
       

def introduce_miss_cols(tensor,mask_, miss_ratio, random_seed, select_cols):
    np.random.seed(random_seed)
    mask_array = np.zeros((tensor.shape[0],tensor.shape[1],tensor.shape[2]), dtype=int)
    mask_array[:,:,select_cols] = 1
    masks = mask_.reshape(-1).copy()
    origionals = tensor.reshape(-1).copy()
    
    x= np.where( mask_array.reshape(-1) == 1)[0].tolist()
    
    obs_indices = np.where(masks)[0].tolist()
    select_miss= list(set(x) & set(obs_indices))
    miss_indices = np.random.choice(
    select_miss, (int)(len(select_miss) * miss_ratio), replace=False)
    
    L= origionals.reshape(-1).copy()
    L[miss_indices] = np.nan
    L= L.reshape(12000,48, 35)
    
    
    flat= np.zeros((12000,48,35)).reshape(-1)
    flat[miss_indices] = 1
    mis_indicate=flat.reshape(12000,48,35)

    return L,miss_indices,mis_indicate

def introduce_miss_cols_last(tensor,mask_, miss_ratio, random_seed, select_cols):
    np.random.seed(random_seed)
    origionals = tensor.reshape(-1).copy()
    masks = mask_.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = []
    for i in select_cols:
      mask_array = np.zeros((tensor.shape[0],tensor.shape[1],tensor.shape[2]), dtype=int)
      mask_array[:,:,i] = 1
 
      x= np.where(mask_array.reshape(-1) == 1)[0].tolist()
      select_miss= list(set(x) & set(obs_indices))
      miss_indices_ = np.random.choice(select_miss, (int)(len(select_miss) * miss_ratio), replace=False)
      miss_indices = list(miss_indices) + list(miss_indices_)
    
    L= origionals.reshape(-1).copy()
    L[miss_indices] = np.nan
    L= L.reshape(12000,48, 35)
    flat= np.zeros((12000,48,35)).reshape(-1)
    flat[miss_indices] = 1
    mis_indicate=flat.reshape(12000,48,35)

    return L,miss_indices,mis_indicate

def array_rsme(tensor,mask_, miss_ratio, random_seed):
    np.random.seed(random_seed)

    masks = mask_.reshape(-1).copy()
    origionals = tensor.reshape(-1).copy()

    obs_indices = np.where(masks)[0].tolist()

    miss_indices = np.random.choice(
    obs_indices, (int)(len(obs_indices) * miss_ratio), replace=False)
    L= origionals.reshape(-1).copy()
    L[miss_indices] = np.nan
    L= L.reshape(12000,48, 35)
    return L,origionals,miss_indices
       
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

def create_individualized_missingness_mask(mask):
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
        tf=((num_measurments_per_feature)/time_steps)
        personalized_mask_patient.append(np.where(patient_mask == 0, tf, patient_mask))
    # stack all feature-specific patient masks tnto a 3d tensor
  personalized_mask_full = np.stack(personalized_mask_patient, axis=0)
  return(personalized_mask_full)
def masked_fill(X, mask, val):
    return corruptor.masked_fill(X, mask, val)

def cal_mse(inputs, target, mask=None):
    """calculate Mean Square Error"""
    assert type(inputs) == type(target), (
        f"types of inputs and target must match, but got"
        f"type(inputs)={type(inputs)}, type(target)={type(target)}"
    )
    lib = np
    if mask is not None:
        return lib.sum(lib.square(inputs - target) * mask) / (lib.sum(mask) + 1e-9)
    else:
        return lib.mean(lib.square(inputs - target))


def cal_rmse(inputs, target, mask=None):
    """calculate Root Mean Square Error"""
    assert type(inputs) == type(target), (
        f"types of inputs and target must match, but got"
        f"type(inputs)={type(inputs)}, type(target)={type(target)}"
    )
    return np.sqrt(cal_mse(inputs, target, mask))


def cal_mre(inputs, target, mask=None):
    """calculate Mean Relative Error"""
    assert type(inputs) == type(target), (
        f"types of inputs and target must match, but got"
        f"type(inputs)={type(inputs)}, type(target)={type(target)}"
    )
    if mask is not None:
        return np.sum(np.abs(inputs - target) * mask) / (
            np.sum(np.abs(target * mask)) + 1e-9
        )
    else:
        return np.mean(np.abs(inputs - target)) / (np.sum(np.abs(target)) + 1e-9)
def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))
def mcar(X, rate, nan=0):
    return corruptor.mcar(X, rate, nan)
def cal_mae(inputs, target, mask=None):
    """calculate Mean Absolute Error"""
    assert type(inputs) == type(target), (
        f"types of inputs and target must match, but got"
        f"type(inputs)={type(inputs)}, type(target)={type(target)}"
    )
    lib = np 
    if mask is not None:
        return lib.sum(lib.abs(inputs - target) * mask) / (lib.sum(mask) + 1e-9)
    else:
        return lib.mean(lib.abs(inputs - target))

def cal_mae(inputs, target, mask=None):
    """calculate Mean Absolute Error"""
    assert type(inputs) == type(target), (
        f"types of inputs and target must match, but got"
        f"type(inputs)={type(inputs)}, type(target)={type(target)}"
    )
    lib =np
    if mask is not None:
        return lib.sum(lib.abs(inputs - target) * mask) / (lib.sum(mask) + 1e-9)
    else:
        return lib.mean(lib.abs(inputs - target))

def get_patient_level_error(miss_indices_, length, observed, basline):
    array_RMSE = []
    array_MAE = []
    array_weight =[]
    array_mse = []
    copy_= basline.reshape(-1).copy()
    for selected,observe in zip(miss_indices_,observed):
#         print(len(selected), print(observe.shape))
        if len(selected)> 0:
            rms= np.mean_squared_error(observe.reshape(-1)[selected], copy_[selected], squared= False)
            mse= np.mean_squared_error(observe.reshape(-1)[selected], copy_[selected], squared= True)

            mea = mae(observe.reshape(-1)[selected], copy_[selected])
        else:
            rms = 0.0
            mea = 0.0
            mse = 0.0
        array_RMSE.append(rms)
        array_MAE.append(mea)
        array_mse.append(mse)
    return(array_mse, array_RMSE,array_MAE)

'''
def get_patient_level_error_new(basline,observed, miss_indices_):
    array_RMSE = []
    array_MAE = []
    array_weight =[]
    array_mse = []
    copy_= basline.reshape(-1).copy()
    for i in range(len(miss_indices_)):
        if len(miss_indices_[i])> 0:
            predictedMatrix = observed[i].reshape(-1)[miss_indices_[i]]
            target = copy_[miss_indices_[i]]                    
            mse=((predictedMatrix - target) ** 2).mean(axis=None)
            rms=np.sqrt(mse)
            mea = np.mean(abs(target - predictedMatrix), axis=None)
        else:
            rms = 0.0
            mea = 0.0
            mse = 0.0
        array_RMSE.append(rms)
        array_MAE.append(mea)
        array_mse.append(mse)
    return(array_mse, array_RMSE,array_MAE)
'''

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
    
def get_patient_level_MSE_array(imputation,observed, miss_indices_, name):
    array_mse = []
    observed = np.where(np.isnan(observed), 0, observed)

    for i in range(miss_indices_.shape[0]):
              
             mse= np.sum(np.square(imputation[i]- observed[i]) * miss_indices_[i]) / (np.sum(miss_indices_[i]) + 1e-9)
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


def get_patient_level_RMSE_array(imputation,observed, miss_indices_, name):
   array_rmse = []
   observed = np.where(np.isnan(observed), 0, observed)

   for i in range(miss_indices_.shape[0]):
            #predictedMatrix = imputation[i]*miss_indices_[i]
            #target = observed[i]*miss_indices_[i]
            #mse=np.sum(np.square(imputation - target) * miss_indices_) / (np.sum(miss_indices_) + 1e-9)
            mse= np.sum(np.square(imputation[i]- observed[i]) * miss_indices_[i]) / (np.sum(miss_indices_[i]) + 1e-9)

            rms=np.sqrt(mse)
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
    
def get_patient_level_MAE_array(imputation,observed, miss_indices_, name):
    array_mea = []
    observed = np.where(np.isnan(observed), 0, observed)

    for i in range(miss_indices_.shape[0]):
        mea= np.sum(np.abs(imputation[i] - observed[i] )*miss_indices_[i]) / (np.sum(miss_indices_[i]) + 1e-9)
        array_mea.append(mea)
    print(name, np.mean(array_mea),np.std(array_mea))

def get_population_MAE(imputation,observed, miss_indices_, name):
        observed = np.where(np.isnan(observed), 0, observed)
        mea= np.sum(np.abs(imputation - observed)*miss_indices_) / (np.sum(miss_indices_) + 1e-9)

        print(name, np.sum(mea))
def get_population_MSE(imputation,observed, miss_indices_, name):
    
    observed = np.where(np.isnan(observed), 0, observed)
    mse= np.sum(np.square(imputation- observed) * miss_indices_) / (np.sum(miss_indices_) + 1e-9)
    print(name, np.sum(mse))
    
    
def get_population_RMSE(imputation,observed, miss_indices_, name):
    observed = np.where(np.isnan(observed), 0, observed)
    mse= np.sum(np.square(imputation- observed) * miss_indices_) / (np.sum(miss_indices_) + 1e-9)
    rms=np.sqrt(mse)

    print(name, np.sum(rms))
    
def get_population_RMSE_weighted(imputation,observed, indicating_mask, name):
    array_RMSE = []
    array_weight =[]
    observed = np.where(np.isnan(observed), 0, observed)

    for count in range(imputation.shape[0]):
         selected = indicating_mask[count]
         w= (selected.sum())/(indicating_mask.sum())
         array_weight.append(w)
         if len(selected)> 0:
           mse=np.sum(np.square(imputation[count] - observed[count]) * selected) / (np.sum(selected) + 1e-9)
           rms=np.sqrt(mse)
         else:
             rms = 0.0
         array_RMSE.append(rms*w)
    print(name, np.sum(array_RMSE))
    
def get_population_MSE_weighted(imputation,observed, indicating_mask, name):
    array_MSE = []
    array_weight =[]
    observed = np.where(np.isnan(observed), 0, observed)


    for count in range(imputation.shape[0]):
        selected = indicating_mask[count]
        w= (selected.sum())/(indicating_mask.sum())
        array_weight.append(w)
        if len(selected)> 0:
          mse=np.sum(np.square(imputation[count] - observed[count]) * selected) / (np.sum(selected) + 1e-9)
        else:
             mse = 0.0
        array_MSE.append(mse*w)
    print(name, np.sum(array_MSE))
def get_population_MAE_weighted(imputation,observed, indicating_mask, name):
    array_MAE = []
    array_weight =[]
    observed = np.where(np.isnan(observed), 0, observed)

    for count in range(imputation.shape[0]):
        selected = indicating_mask[count]
        w= (selected.sum())/(indicating_mask.sum())
        array_weight.append(w)
        if (selected.sum())> 0:
            mae = np.sum(np.abs(imputation[count] - observed[count])*selected) / (np.sum(selected) + 1e-9)

        else:
            mae = 0.0
        array_MAE.append(mae*w)
    print(name, np.nansum(array_MAE))

def get_patient_level(miss_indices_, length, observed, basline):
    array_RMSE = []
    array_MAE = []
    array_weight =[]
    array_mse = []
    copy_= basline.reshape(-1).copy()
    basline.shape[1]
    for count in range(length):
        selected = miss_indices_[miss_indices_ > (48*35)* (count)]
        selected = selected[selected <= (48*35)* (count+1)]
        w= len(selected)/len(miss_indices_)
        array_weight.append(w)
        if len(selected)> 0:
            rms= mean_squared_error(observed[selected], copy_[selected], squared= False)
            mse= mean_squared_error(observed[selected], copy_[selected], squared= True)

            mea = mae(observed[selected], copy_[selected])
        else:
            rms = 0.0
            mea = 0.0
            mse = 0.0
        array_RMSE.append(rms*w)
        array_MAE.append(mea*w)
        array_mse.append(mse*w)
    return(array_mse, array_RMSE,array_MAE,array_weight)
    #return(array_mea)