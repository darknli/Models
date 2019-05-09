import tensorflow as tf
from keras import backend as K


def earth_movers_distance(y_true, y_pred):
    """
    Method solving regression tasks using way of classification
    """
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)


def keras_weight_cross_entroy(y_true, y_pred):
    pass


def tf_weight_cross_entroy(y_true, y_pred, c=1.2):
    """
    The loss function weighted Coefficient c (default 1.2) losses same samples which belong to the No.0 of classes and predicted wrongly.
    In other word, the loss of sample weighted c (default 1.2) when both 2 conditions are met:
        1) The true label of sample is the first class(but the index of first class is 0 actually)
        2) The probabilily that sample be predicted its true class less than 0.5
    """
    label_true = tf.argmax(y_true, axis=-1)
    target_positive = tf.zeros_like(label_true)
    mul_y = tf.reduce_sum(y_true*y_pred, axis=-1)
    log_mul = -tf.math.log(mul_y)
    ch_p = tf.equal(target_positive, label_true)
    ch_res = tf.less_equal(mul_y, 0.5)
    loss = tf.where(tf.logical_and(ch_p, ch_res), c * log_mul, log_mul)
    return tf.reduce_mean(loss)