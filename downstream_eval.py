from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from sklearn.svm import SVC

def get_pecent_missing_samples(mask_, percent, percent_2, feat_measured_min, feat_measured_max):
    max_= mask_.shape[1]*mask_.shape[2]
    print(mask_.shape)
    count=0
    index = []
    high= []
    min_feat_measured = feat_measured_min*35
    max_feat_measured = feat_measured_max*35
    
    #print(min_feat_measured)
    for i in mask_:
        missing= ((max_-i.sum())/max_)
        num_measurments_per_feature = i.sum(axis=0)
#         print(num_measurments_per_feature)
        if ((missing>= percent) & (missing< percent_2)) :
            index.append(count)
        observed_columns= np.count_nonzero(num_measurments_per_feature)
        missing_cols = mask_.shape[2]- observed_columns
        if ((missing_cols>= min_feat_measured) & (missing_cols< max_feat_measured)) :
             high.append(count)
        count= count+1
    return(index,high)

def get_results(baseline_names, list_baselines, outcome):
  for name,baseline in zip(baseline_names,list_baselines):
    imputed = baseline.reshape(-1,baseline.shape[1] * baseline.shape[2])
    label = outcome[:imputed.shape[0]]
    data = np.nan_to_num(imputed)
    print("Number of Included Samples",len(imputed))
    print("Number of Positive Samples",label.sum())
    X_train, X_test, y_train, y_test = train_test_split(imputed, label, test_size=0.2, random_state=42)
    auc = []
    auprc = []
    print(name)
    for i in range(1):
        model = SVC(probability= True).fit(X_train,y_train)
        pred = model.predict_proba(X_test)
        auc.append(roc_auc_score(y_test.reshape(-1,), pred[:, 1].reshape(-1, )))
        auprc.append(average_precision_score(y_test.reshape(-1,), pred[:, 1].reshape(-1, )))
    print("Average AUC", round(np.mean(auc),3), "Average AUPRC",round(np.mean(auprc),3))
    
    
def get_result_sample_missingess(baseline_names, list_baselines,outcome, mask_, min_SM_percent, max_SM_percent):
      for name,baseline in zip(baseline_names,list_baselines):
        ids_sample, id_feat= get_pecent_missing_samples(mask_, min_SM_percent, max_SM_percent, 0, 1)
        imputed = baseline.reshape(-1,baseline.shape[1] * baseline.shape[2])
        imputed = imputed[ids_sample,]
        label = outcome[ids_sample]
        data = np.nan_to_num(imputed)
        print("Number of Included Samples",len(imputed))
        print("Number of Positive Samples",label.sum())

        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
        auc = []
        auprc = []
        print(name)
        for i in range(2):
            model = SVC(probability= True).fit(X_train,y_train)
            pred = model.predict_proba(X_test)
            auc.append(roc_auc_score(y_test.reshape(-1,), pred[:, 1].reshape(-1, )))
            auprc.append(average_precision_score(y_test.reshape(-1,), pred[:, 1].reshape(-1, )))
        print("Average AUC", round(np.mean(auc),3), "Average AUPRC", round(np.mean(auprc),3))        
        
       
        
def get_result_feature_missingess(baseline_names, list_baselines,outcome, mask_, min_feat_miss_percent, max_feat_miss_percent):
    for name,baseline in zip(baseline_names,list_baselines):
        ids_sample, id_feat= get_pecent_missing_samples(mask_, 0, 1, min_feat_miss_percent, max_feat_miss_percent)
        imputed = baseline.reshape(-1,baseline.shape[1] * baseline.shape[2])
        imputed = imputed[id_feat,]
        label = outcome[id_feat]
        data = np.nan_to_num(imputed)
        print("Number of Included Samples",len(data))
        print("Number of Positive Samples",label.sum())

        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
        auc = []
        auprc = []
        print(name)
        for i in range(2):
            model = SVC(probability= True).fit(X_train,y_train)
            pred = model.predict_proba(X_test)
            auc.append(roc_auc_score(y_test.reshape(-1,), pred[:, 1].reshape(-1, )))
            auprc.append(average_precision_score(y_test.reshape(-1,), pred[:, 1].reshape(-1, )))
        print("Average AUC", round(np.mean(auc),3), "Average AUPRC",round(np.mean(auprc),3))    