import numpy as np
import pickle
import os 
import functools
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import os
import functools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

experiment_name = "hirid"

### load the data
saved_path = 'data/48hrs/'
observed_only_real = np.load(saved_path + experiment_name + '.npz')["observed_only_real"]
observed_only_rec = np.load(saved_path + experiment_name + '.npz')["observed_only_rec"]
IMM_real = np.load(saved_path + experiment_name + '.npz')["IMM_real"]
IMM_rec = np.load(saved_path + experiment_name + '.npz')["IMM_rec"]

### get the label of mortality
data_path = 'hirid_dataset/48hrs'
with open(os.path.join(data_path, 'patient_tb_selected.pkl'), 'rb') as f:
    patient_statics = pickle.load(f)

### match labels
sample_size = observed_only_real.shape[0]
patient_statics_ = patient_statics[:sample_size]
patient_statics_.discharge_status = patient_statics_["discharge_status"].replace({"alive":0, "dead":1})
print(patient_statics_.columns)
label_mort = np.asarray(patient_statics_.discharge_status)
 
### params in lstm network -----------------
BATCH_SIZE = 64
EPOCHS = 200
KEEP_PROB = 0.8
REGULARIZATION = 0.0001
NUM_HIDDEN = [64, 64, 64] 
LEARNING_RATE = 0.0001
STRATIFY_FLAG = 1
WEIGHT_FLAG = [1, 1]
NUM_CLASSES = 2

# control randomness params
RANDOM = None

# assign x and y array
x_array = IMM_rec

# assign labels
enc = OneHotEncoder()
enc.fit(label_mort.reshape([-1, 1]))
y_array_classes = enc.transform(label_mort.reshape([-1, 1])).toarray()  # label_binarize(label_mort, classes=range(NUM_CLASSES))  #label_mort.reshape([-1, 1]) 

# splitting ratio
test_split_ratio = 0.2

# training/test/validation set split
if STRATIFY_FLAG == 0:
    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array_classes, test_size=test_split_ratio)
else:
    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array_classes, test_size=test_split_ratio, random_state=RANDOM, stratify=y_array_classes)

print("Training size: ", x_train.shape, y_train.shape)
print("Test size: ", x_test.shape, y_test.shape)

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
    def make_rnn_cell(self, base_cell=tf.nn.rnn_cell.BasicLSTMCell, state_is_tuple=True):

        input_dropout = self.dropout_prob
        output_dropout = self.dropout_prob

        cells = []
        for num_units in self._num_hidden:
            cell = base_cell(num_units, state_is_tuple=state_is_tuple)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout) ##seed=RANDOM
            cells.append(cell)

        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
        return cell

    # predictor for slices
    @lazy_property
    def prediction(self):

        cell = self.make_rnn_cell

        # Recurrent network.
        output, final_state = tf.nn.dynamic_rnn(cell, self.data, dtype=tf.float32)

        with tf.variable_scope("model") as scope:
            tf.get_variable_scope().reuse_variables()

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
        sample_weights = tf.reduce_sum(tf.multiply(onehot_labels, (self.class_weights)), -1)
        ce_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=sample_weights) 
        l2 = self.reg * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        ce_loss += l2
        return ce_loss

    @lazy_property
    def optimize(self):
        learning_rate = LEARNING_RATE
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.cross_ent)

    @lazy_property
    def error(self):
        prediction = tf.argmax(self.prediction[1], 1)
        real = tf.cast(self.target, tf.int32)
        prediction = tf.cast(prediction, tf.int32)
        mistakes = tf.not_equal(real, prediction)
        mistakes = tf.cast(mistakes, tf.float32)
        mistakes = tf.reduce_sum(mistakes, reduction_indices=0)
        total = BATCH_SIZE
        mistakes = tf.divide(mistakes, tf.to_float(total))
        return mistakes

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @lazy_property
    def summaries(self):
        tf.summary.scalar('loss', tf.reduce_mean(self.cross_ent))
        tf.summary.scalar('error', self.error)
        merged = tf.summary.merge_all()
        return merged

## the training of lstm
tf.reset_default_graph()
# set_random_seed(RANDOM)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    _, length, num_features = x_train.shape
    num_data_cols = num_features

    # placeholders
    data = tf.placeholder(tf.float32, [None, length, num_data_cols])
    target = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    dropout_prob = tf.placeholder(tf.float32)
    reg = tf.placeholder(tf.float32)

    # initialization
    model = VariableSequenceLabelling(data, target, dropout_prob, reg, num_hidden=NUM_HIDDEN)
    sess.run(tf.global_variables_initializer())

    batch_size = BATCH_SIZE
    dp = KEEP_PROB
    rp = REGULARIZATION
    train_samples = x_train.shape[0]
    indices = list(range(train_samples))
    num_classes = NUM_CLASSES

    # for storing results
    test_data = x_test
    test_aucs_macro = []
    train_aucs_macro = []

    epoch = -1

    while (epoch < EPOCHS):
        epoch += 1
        np.random.seed(RANDOM)
        np.random.shuffle(indices)

        num_batches = train_samples // batch_size
        for batch_index in range(num_batches):
            sample_indices = indices[batch_index * batch_size:batch_index * batch_size + batch_size]
            batch_data = x_train[sample_indices, :, :num_data_cols]
            batch_target = y_train[sample_indices]
            _, loss = sess.run([model.optimize, model.cross_ent], {data: batch_data, target: batch_target, dropout_prob: dp, reg: rp})

        # prediction - train
        cur_train_preds = sess.run(model.prediction, {data: x_train, target: y_train, dropout_prob: 1, reg: rp})
        train_preds = cur_train_preds[1]
        train_auc_macro = roc_auc_score(y_train, train_preds)
        train_aucs_macro.append(train_auc_macro)
        # train_acc = accuracy_score(y_train[:, 0], train_preds[:, 0])

        # prediction - test
        cur_test_preds = sess.run(model.prediction, {data: x_test, target: y_test, dropout_prob: 1, reg: rp})
        test_preds = cur_test_preds[1]
        test_auc_macro = roc_auc_score(y_test, test_preds)
        test_aucs_macro.append(test_auc_macro)    
             
        print("Test  AUC on epoch {} is {}".format(epoch, test_auc_macro))
        print("Train AUC on epoch {} is {}".format(epoch, train_auc_macro))
        # print("Train Acc on epoch {} is {}".format(epoch, train_acc))

    print("Best AUC: {}".format(max(test_aucs_macro)))

