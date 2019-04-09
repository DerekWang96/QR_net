import tensorflow as tf


def NonLinearFusion(r_Q, r_A, rnn_dim, batch_size):
    with tf.variable_scope("Fusion", reuse=tf.AUTO_REUSE):
        wQ = tf.get_variable("wQ", [2 * rnn_dim, 2 * rnn_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
        wA = tf.get_variable("wA", [2 * rnn_dim, 2 * rnn_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
        uQA = tf.nn.tanh(
            tf.matmul(tf.tile(tf.expand_dims(wQ, 0), [batch_size, 1, 1]), r_Q)
            + tf.matmul(tf.tile(tf.expand_dims(wA, 0), [batch_size, 1, 1]), r_A))
    return uQA


def NonLinear_concat_Fusion(r_1, r_2, rnn_dim, batch_size):
    with tf.variable_scope("Concat_Fusion", reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable("w1", [2 * rnn_dim, 4 * rnn_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
        w2 = tf.get_variable("w2", [2 * rnn_dim, 4 * rnn_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
        uQR = tf.nn.tanh(
            tf.matmul(tf.tile(tf.expand_dims(w1, 0), [batch_size, 1, 1]), r_1)
            + tf.matmul(tf.tile(tf.expand_dims(w2, 0), [batch_size, 1, 1]), r_2))
    return uQR