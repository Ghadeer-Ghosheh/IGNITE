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

   
  
    

@tf2.function

def ConLoss(zis, zjs, batch_size, temperature=0.1):
    
    
        representations = tf2.concat([zis, zjs], axis = 0)
        cosine_loss = tf2.keras.losses.CosineSimilarity(axis= -1)
        similarity_matrix = cosine_loss( tf2.expand_dims(representations,1),  tf2.expand_dims(representations, 0))
        r_pos= tf2.linalg.diag_part(similarity_matrix,batch_size)
        positives = r_pos.reshape(batch_size, 1)
        
        diag = np.eye(2 *batch_size)
        l1 = np.eye((2 * batch_size), 2 *batch_size, k=batch_size)
        l2 = np.eye((2 *batch_size), 2 *batch_size, k=batch_size)
        initial_mask = diag + l1 + l2
        # mask for patient-patient or drug-drug match
        initial_mask[:batch_size, :batch_size] = 1.
        initial_mask[batch_size:,batch_size:] = 1.
        initial_mask[batch_size:, :batch_size] = 1.
        
        mask = tf2.convert_to_tensor((diag + l1 + l2), dtype = bool)
        mask = (1 - mask)
        
        negatives = similarity_matrix[mask].reshape(batch_size, -1)
        
        
        logits = tf2.cat((positives, negatives), axis=1)
        logits /= temperature

        labels = tf2.zeros(batch_size)
        loss= tf2.losses.softmax_cross_entropy(logits, labels)
        
        loss_= loss / (batch_size), r_pos
        return loss_
