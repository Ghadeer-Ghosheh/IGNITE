import tensorflow as tf2
tf2.random.set_seed(42)
from tfdeterminism import patch
import os 
import numpy as np
tf2.config.run_functions_eagerly(True)

os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow.compat.v1 as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_v2_behavior()

def nt_xent_loss(out, out_aug, batch_size, hidden_norm=True, temperature=0.1):
    """
    https://github.com/google-research/simclr/blob/master/objective.py
    """
    if hidden_norm:
        out = tf.nn.l2_normalize(out,-1)
        out_aug = tf.nn.l2_normalize(out_aug, -1)
    INF = 1e9 # np.inf
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)
    masksINF = masks * INF

    logits_aa = tf.matmul(out, out, transpose_b=True) / temperature
    logits_bb = tf.matmul(out_aug, out_aug, transpose_b=True) / temperature

    logits_aa = logits_aa - masksINF
    logits_bb = logits_bb - masksINF

    logits_ab = tf.matmul(out, out_aug, transpose_b=True) / temperature
    logits_ba = tf.matmul(out_aug, out, transpose_b=True) / temperature

    loss_a = tf.losses.softmax_cross_entropy(labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.losses.softmax_cross_entropy(labels, tf.concat([logits_ba, logits_bb], 1))
    loss = loss_a + loss_b
    
    return loss

   
  
