from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score,balanced_accuracy_score,recall_score,precision_score

from sklearnex import patch_sklearn 
patch_sklearn()
from sklearn.svm import SVC

def get_results_2(baseline_names, list_baselines, outcome):
  for name,baseline in zip(baseline_names,list_baselines):
    imputed = baseline.reshape(-1,baseline.shape[1] * baseline.shape[2])
    label = outcome[:imputed.shape[0]] 
    X_train, X_test, y_train, y_test = train_test_split(imputed, label, test_size=0.2, random_state=42)
    auc = []
    auprc = []
    test_f1s =[]
    test_balanced_accuracys = []
    for i in range(5):
        model = SVC(probability= True).fit(X_train,y_train)
        pred = model.predict_proba(X_test)
        y_pred_ = (pred > 0.5) 

        test_f1 = f1_score(y_test.reshape(-1,), y_pred_[:, 1].reshape(-1, ))
        test_balanced_accuracy = balanced_accuracy_score(y_test.reshape(-1,), y_pred_[:, 1].reshape(-1, ))
        test_recall = recall_score(y_test.reshape(-1,), y_pred_[:, 1].reshape(-1, ))
        test_precision_score = precision_score(y_test.reshape(-1,), y_pred_[:, 1].reshape(-1, ))
        
        test_f1s.append(test_f1)
        test_balanced_accuracys.append(test_balanced_accuracy)
        auc.append(roc_auc_score(y_test.reshape(-1,), pred[:, 1].reshape(-1, )))
        auprc.append(average_precision_score(y_test.reshape(-1,), pred[:, 1].reshape(-1, )))
    return(round(np.mean(auc),3), round(np.mean(auprc),3), round(np.mean(test_f1),3), round(np.mean(test_balanced_accuracy),3),\
          round(np.mean(test_recall),3), round(np.mean(test_precision_score),3))

    
def get_pecent_missing_samples(mask_, percent, percent_2, feat_measured_min, feat_measured_max):
    max_= mask_.shape[1]*mask_.shape[2]
    count=0
    index = []
    high= []
    min_feat_measured = feat_measured_min*mask_.shape[2]
    max_feat_measured = feat_measured_max*mask_.shape[2]
    for i in mask_:
        missing= ((max_-i.sum())/max_)
        num_measurments_per_feature = i.sum(axis=0)
        if ((missing>= percent) & (missing< percent_2)) :
            index.append(count)
        observed_columns= np.count_nonzero(num_measurments_per_feature)
        missing_cols = mask_.shape[2]- observed_columns
        if ((missing_cols>= min_feat_measured) & (missing_cols< max_feat_measured)) :
             high.append(count)
        count= count+1
    return(index,high)

    

def get_sample_missingness_population(binary_mask):
    # get population-level stats for the sample level missingness for latex table

    sample_missingness =[]
    feat_length = binary_mask.shape[2]
    flatten= binary_mask.reshape(binary_mask.shape[0]*binary_mask.shape[1],feat_length)
    for i in range(feat_length):
        sum_missing=flatten.shape[0]-np.sum(flatten[:,i])
        sample_missingness.append(sum_missing/flatten.shape[0]*100)
    return(sample_missingness)
def get_feature_missingness_population(binary_mask):
    # get population-level stats for the feature level missingness  for latex table
    feature_missingness_patient = []
    feat_length = binary_mask.shape[2]
    percentages = []
    for patient_mask in binary_mask:
            feature_missingness_patient.append(patient_mask.sum(axis=0))
    feature_missingness_full = np.stack(feature_missingness_patient, axis=0)
    for feat in range(feat_length):
        percentages.append(((np.count_nonzero(feature_missingness_full[:,feat]==0))/ binary_mask.shape[0]*100))
    return(percentages)



def get_sample_missingness_patient(binary_mask):
    # get sample missingness in a single patient mask
    max_= binary_mask.shape[1]*binary_mask.shape[0]
    missing= ((max_-binary_mask.sum())/max_)
    return(missing)


def get_feature_missingness_patient(binary_mask):
    # get feature missingness in a single patient mask
    observed_columns= np.count_nonzero(binary_mask.sum(axis = 0))
    return(1-(observed_columns/binary_mask.shape[1]))

def miss_get_quantiles(miss_, quantile_lower = 0.25, quantile_upper = 0.75):
        # get quantile split thresholds for the sample and feature-wise missingness experiments
        list_sample = []
        list_features = []
        for i in miss_:
            list_sample.append(get_sample_missingness_patient(i))
            list_features.append(get_feature_missingness_patient(i))
        top_feat= np.quantile(list_features, quantile_upper)
        lower_feat = np.quantile(list_features, quantile_lower)
        top_sample= np.quantile(list_sample, quantile_upper)
        lower_sample = np.quantile(list_sample, quantile_lower)
        return(lower_sample,top_sample,lower_feat,top_feat)



def get_sets_sample_missingess(data,outcome, mask_, min_SM_percent, max_SM_percent):
       #Get data splits for each of the sample-wise missingness experiments        
       ids_sample, id_feat= get_pecent_missing_samples(mask_, min_SM_percent, max_SM_percent, 0, 1)
       imputed = data
       imputed = imputed[ids_sample,]
       label = outcome[ids_sample]
       data = np.nan_to_num(imputed)
       #print("Number of Included Samples",len(data))
       #print("Number of Positive Samples",label.sum())

       return(data,label)
       
def get_sets_feature_missingess(data_2, data,mask_2,outcome, interven, mask_, min_feat_miss_percent, max_feat_miss_percent):

      #Get data splits for each of the feature-wise missingness experiments        

       ids_sample, id_feat= get_pecent_missing_samples(mask_, 0, 1, min_feat_miss_percent, max_feat_miss_percent)

       interven = interven[id_feat,]
       data = data[id_feat,]
       data_2 = data_2[id_feat,]
       print(outcome.shape)
       label = outcome[id_feat]
       mask_2 = mask_2[id_feat,]
       # print("Number of Positive Samples",label.sum())

       return(data_2,data,mask_2,interven, label)
   
def get_sets_sample_missingness(data_2, data,mask_2,outcome, interven, mask_, min_feat_miss_percent, max_feat_miss_percent):

        #Get data splits for each of the feature-wise missingness experiments        

         ids_sample, id_feat= get_pecent_missing_samples(mask_, min_feat_miss_percent, max_feat_miss_percent, 0, 1)
         id_feat = ids_sample
         interven = interven[id_feat,]
         data = data[id_feat,]
         data_2 = data_2[id_feat,]
         print(outcome.shape)
         label = outcome[id_feat]
         mask_2 = mask_2[id_feat,]
         # print("Number of Positive Samples",label.sum())

         return(data_2,data,mask_2,interven, label)
          
def get_sets_feature_missingess2( data,outcome, mask_, min_feat_miss_percent, max_feat_miss_percent):
      #Get data splits for each of the feature-wise missingness experiments        

       ids_sample, id_feat= get_pecent_missing_samples(mask_, 0, 1, min_feat_miss_percent, max_feat_miss_percent)

       data = data[id_feat,]
       print(outcome.shape)
       label = outcome[id_feat]
       # print("Number of Positive Samples",label.sum())

       return(data, label)
   
       
def get_sets_feature_missingess3( data,mask_2,outcome, mask_, min_feat_miss_percent, max_feat_miss_percent):
      #Get data splits for each of the feature-wise missingness experiments        

       ids_sample, id_feat= get_pecent_missing_samples(mask_, 0, 1, min_feat_miss_percent, max_feat_miss_percent)

       data = data[id_feat,]
       print(outcome.shape)
       label = outcome[id_feat]
       mask_2 = mask_2[id_feat,]

       # print("Number of Positive Samples",label.sum())

       return(data, mask_2, label)
def get_sets_samples_2( data,outcome, mask_, min_feat_miss_percent, max_feat_miss_percent):
      #Get data splits for each of the feature-wise missingness experiments        

       ids_sample, id_feat= get_pecent_missing_samples(mask_, min_feat_miss_percent, max_feat_miss_percent, 0, 1)

       data = data[ids_sample,]
       print(outcome.shape)
       label = outcome[ids_sample]
       # print("Number of Positive Samples",label.sum())

       return(data, label)