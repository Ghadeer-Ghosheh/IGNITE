# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 13:56:05 2023

@author: gghos
"""

import numpy as np
import pickle
import os 
import functools
import tensorflow as tf
import random as rn
from sklearn.model_selection import train_test_split
from bottleneck import push
from fancyimpute import IterativeImputer
from prep_inputs import get_impuation, get_sets_sample_missingness, get_sets_feature_missingness
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score,f1_score,balanced_accuracy_score,recall_score,precision_score
tf.random.set_seed(42)
np.random.seed(42)
rn.seed(42)

os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import argparse

def main (args):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
           try:
             for gpu in gpus:
               tf.config.experimental.set_memory_growth(gpu, True)
           except RuntimeError as e:
             print(e)
        
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        experiment_name = "imputedfullFINAL_main_256_lr_new_05_30_condition_IMM"
        saved_path = 'data/'
       # observed_only_real = np.load(saved_path + experiment_name + '.npz')["observed_only_real"]
        #observed_only_rec = np.load(saved_path + experiment_name + '.npz')["observed_only_rec"]
       # IMM_real = np.load(saved_path + experiment_name + '.npz')["IMM_real"]
        IMM_rec = np.load(saved_path + experiment_name + '.npz')["imputed_data_oo"]
        data_path = 'extracts/'
        with open(os.path.join(data_path, 'normalized_combined.pkl'), 'rb') as f:
            original = pickle.load(f)
        with open(os.path.join(data_path, 'out_combined.pkl'), 'rb') as f:
            outcomes = pickle.load(f)
        with open(os.path.join('BRITS_0.1_X_full.pkl'), 'rb') as f:
            BRITS = pickle.load(f)

        with open(os.path.join('SAITS_X_full.pkl'), 'rb') as f:
         SAITS = pickle.load(f)

        original,conditions,outcomes= get_sets_sample_missingness(original,conditions,outcomes, 0.75,1.1)

        miss=~np.isnan(original)*1
        zero=np.where(np.isnan(original), 0, original)
        array= np.array(miss, dtype=float)   
        imputed_ours =(array * (zero)+ ((1-array) *IMM_rec))
        
        
        LOCF, zero, mean_imputed, MICE=get_impuation(original)
        
        
        ### params in lstm network -----------------
        BATCH_SIZE = args.BATCH_SIZE
        EPOCHS = args.EPOCHS
        KEEP_PROB = args.KEEP_PROB
        REGULARIZATION = args.REGULARIZATION
        NUM_HIDDEN = [args.dim, args.dim, args.dim] 
        LEARNING_RATE = args.LEARNING_RATE
        STRATIFY_FLAG = 1
        WEIGHT_FLAG = [1, 1]
        NUM_CLASSES = 2
        
       
        
        
        # control randomness params
        RANDOM = 42
        label_mort = outcomes

        # assign x and y array
        x_array =  imputed_ours
        print(x_array.shape)

        # assign labels
        enc = OneHotEncoder()
        enc.fit(label_mort.reshape([-1, 1]))
        y_array_classes = enc.transform(label_mort.reshape([-1, 1])).toarray()  
        
        # splitting ratio
        test_split_ratio = 0.3
        
        
        wandb.init(project="ignite_lstm", entity="....",sync_tensorboard=True,settings=dict(start_method='thread'), config = args)
        print("active")
        
        # training/test/validation set split
        split_ratio=int(x_array.shape[0]*0.8)
        x_train, x_test = x_array[:split_ratio,:,:], x_array[split_ratio:,:,:]
        y_train, y_test = y_array_classes[:split_ratio],y_array_classes[split_ratio:]
        
        print("Training size: ", x_train.shape, y_train.shape)
        print("Test size: ", x_test.shape, y_test.shape)
        print(y_train[1].sum(), y_test[1].sum())
        
        # LSTM -------------------------------
        def lazy_property(function):
            attribute = '_' + function.__name__
        
            @property
            @functools.wraps(function)
            def wrapper(self):
                if not hasattr(self, attribute):
                    setattr(self, attribute, function(self))
                return getattr(self, attribute)
        
            return wrapper
        
        
        class VariableSequenceLabelling:
        
            def __init__(self, data, target, dropout_prob, reg, num_hidden):
                self.data = data
                self.target = target
                self.dropout_prob = dropout_prob
                self.reg = reg
                self._num_hidden = num_hidden
                self._num_layers = len(num_hidden)
                self.num_classes = NUM_CLASSES
                self.class_weights = WEIGHT_FLAG
                self.prediction
                self.error
                self.optimize
        
            @lazy_property
            def make_rnn_cell(self, base_cell=tf.compat.v1.nn.rnn_cell.BasicLSTMCell, state_is_tuple=True):
        
                input_dropout = self.dropout_prob
                output_dropout = self.dropout_prob
        
                cells = []
                for num_units in self._num_hidden:
                    cell = base_cell(num_units, state_is_tuple=state_is_tuple)
                    cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout,seed=RANDOM) ##seed=RANDOM
                    cells.append(cell)
        
                cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
                return cell
        
            # predictor for slices
            @lazy_property
            def prediction(self):
        
                cell = self.make_rnn_cell
        
                # Recurrent network.
                output, final_state = tf.compat.v1.nn.dynamic_rnn(cell, self.data, dtype=tf.float32)
        
                with tf.compat.v1.variable_scope("model") as scope:
                    tf.compat.v1.get_variable_scope().reuse_variables()
        
                    # final weights
                    num_classes = self.num_classes
                    weight, bias = self._weight_and_bias(self._num_hidden[-1], num_classes)
        
                    # flatten + sigmoid
                    logits = tf.matmul(final_state[-1][-1], weight) + bias
                    prediction = tf.nn.softmax(logits)
        
                    return logits, prediction
            
            @lazy_property
            def cross_ent(self):
                logits = self.prediction[0]
                onehot_labels = self.target
                sample_weights = tf.reduce_sum(input_tensor=tf.multiply(onehot_labels, (self.class_weights)), axis=-1)
                ce_loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=sample_weights) 
                l2 = self.reg * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables())
                ce_loss += l2
                return ce_loss
        
            @lazy_property
            def optimize(self):
                learning_rate = LEARNING_RATE
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
                return optimizer.minimize(self.cross_ent)
        
            @lazy_property
            def error(self):
                prediction = tf.argmax(input=self.prediction[1], axis=1)
                real = tf.cast(self.target, tf.int32)
                prediction = tf.cast(prediction, tf.int32)
                mistakes = tf.not_equal(real, prediction)
                mistakes = tf.cast(mistakes, tf.float32)
                mistakes = tf.reduce_sum(input_tensor=mistakes, axis=0)
                total = BATCH_SIZE
                mistakes = tf.divide(mistakes, tf.cast(total, dtype=tf.float32))
                return mistakes
        
            @staticmethod
            def _weight_and_bias(in_size, out_size):
                weight = tf.random.truncated_normal([in_size, out_size], stddev=0.01, seed = RANDOM)
                bias = tf.constant(0.1, shape=[out_size])
                return tf.Variable(weight), tf.Variable(bias)
        
            @lazy_property
            def summaries(self):
                tf.compat.v1.summary.scalar('loss', tf.reduce_mean(input_tensor=self.cross_ent))
                tf.compat.v1.summary.scalar('error', self.error)
                merged = tf.compat.v1.summary.merge_all()
                return merged
        
        ## the training of lstm
        tf.compat.v1.reset_default_graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        step = 0
        
        with tf.compat.v1.Session(config=config) as sess:
            _, length, num_features = x_train.shape
            num_data_cols = num_features
        
            # placeholders
            data = tf.compat.v1.placeholder(tf.float32, [None, length, num_data_cols])
            target = tf.compat.v1.placeholder(tf.float32, [None, NUM_CLASSES])
            dropout_prob = tf.compat.v1.placeholder(tf.float32)
            reg = tf.compat.v1.placeholder(tf.float32)
        
            # initialization
            model = VariableSequenceLabelling(data, target, dropout_prob, reg, num_hidden=NUM_HIDDEN)
            sess.run(tf.compat.v1.global_variables_initializer())
        
            batch_size = BATCH_SIZE
            dp = KEEP_PROB
            rp = REGULARIZATION
            train_samples = x_train.shape[0]
            indices = list(range(train_samples))
        
            # for storing results
            test_aucs_macro = []
            train_aucs_macro = []
            test_auprcs_macro =[]
        
            epoch = -1
            
        
            while (epoch < EPOCHS):
                epoch += 1
                np.random.seed(RANDOM)
                #np.random.shuffle(indices)
        
                num_batches = train_samples // batch_size
                for batch_index in range(num_batches):
                    sample_indices = indices[batch_index * batch_size:batch_index * batch_size + batch_size]
                    batch_data = x_train[sample_indices, :, :num_data_cols]
                    batch_target = y_train[sample_indices]
                    _, loss = sess.run([model.optimize, model.cross_ent], {data: batch_data, target: batch_target, dropout_prob: dp, reg: rp})
                    wandb.log({"loss": loss, "step_new" :step})

                # prediction - train
                cur_train_preds = sess.run(model.prediction, {data: x_train, target: y_train, dropout_prob: 1, reg: rp})
                train_preds = cur_train_preds[1]
                train_preds_=np.argmax(train_preds, axis=1)
                train_preds_ = (train_preds_ > 0.5) 
                train_auc_macro = roc_auc_score(y_train, train_preds)
               
                train_f1 = f1_score(np.argmax(y_train,axis = 1), train_preds_)        
                train_balanced_accuracy = balanced_accuracy_score(np.argmax(y_train,axis = 1), train_preds_)
                train_aucs_macro.append(train_auc_macro)
                # train_acc = accuracy_score(y_train[:, 0], train_preds[:, 0])
                #wandb.log({"train aucs": train_auc_macro,"train f1": train_f1, "train balanaced acc":train_balanced_accuracy,"epoch":epoch})

                  # prediction - test
                cur_test_preds, pred= sess.run(model.prediction, {data: x_test, target: y_test, dropout_prob: 1, reg: rp})
                test_preds = pred[:,1]
        
                test_auc = roc_auc_score(y_test[:,1], test_preds)
                test_auprc = average_precision_score(y_test[:,1], test_preds)

                test_aucs_micro.append(test_auc)    
                test_auprcs_micro.append(test_auprc)    
                print("Test AUPRC on epoch {} is {}".format(epoch, test_auprc))
                print("Test AUC on epoch {} is {}".format(epoch, test_auc))
                
                
                
                y_pred_ = (test_preds > 0.5) 
                test_f1 = f1_score(y_test[:,1], y_pred_)
                test_recall = recall_score(y_test[:,1], y_pred_)
                test_precision_score = precision_score(y_test[:,1], y_pred_)
                test_balanced_accuracy = balanced_accuracy_score(y_test[:,1], y_pred_)
                specificty =recall_score(y_test[:,1], y_pred_,pos_label=0)
                
                #wandb.log({"epoch":epoch, "test aucs": test_auc_macro,"test auprc":test_auprc_macro,"test f1": test_f1, "test balanaced acc":test_balanced_accuracy, "test Recall": test_recall, "Test Precision": test_precision_score, "specificty": specificty})        
        

if __name__ == '__main__':  
    exp_name = "FINAL_main_256_lr_new_05_30_condition_IMM"

    parser = argparse.ArgumentParser()

    parser.add_argument('--BATCH_SIZE', type=int, default=1024, help='The batch size for training the model.')
    parser.add_argument('--EPOCHS', type=int, default=600, help='The number of epoches in training  IGNITE.')
    parser.add_argument('--REGULARIZATION', type=float, default=0.0001)
    parser.add_argument('--LEARNING_RATE', type=float, default=0.0005)

    parser.add_argument('--KEEP_PROB', type=float, default=0.8)
    parser.add_argument("--dim", type= int,default = 128)
    parser.add_argument('--experiment_name',  default=exp_name, help= "The name of the experiment")

    args = parser.parse_args() 
  
    main(args)
