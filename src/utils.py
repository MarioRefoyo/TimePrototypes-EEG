import tensorflow as tf


@tf.function
def pairwise_dist(x, y):
    """
    Pairwise distance between two vectors
    # Argument:
                X: ndtensor
                Y: mdtensor
    # Returnes:
                D: (m x n) matrix
    """
    # squared norm of each vector
    xx = tf.reduce_sum(tf.square(x), 1)
    yy = tf.reduce_sum(tf.square(y), 1)
    # XX is a row vector and YY is a column vector
    xx = tf.reshape(xx, [-1, 1])
    yy = tf.reshape(yy, [1, -1])
    return xx + yy - 2 * tf.matmul(x, tf.transpose(y))


@tf.function
def normalize(x_i):
    """
    Normalize the vector
    """
    min_i = tf.math.reduce_min(x_i)
    max_i = tf.math.reduce_max(x_i)
    x_i_normalized = (x_i - min_i) / (max_i - min_i)
    return x_i_normalized
