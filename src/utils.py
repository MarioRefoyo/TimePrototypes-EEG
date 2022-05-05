import tensorflow as tf


@tf.function
def pairwise_dist(X, Y):
    '''
    Pairwise distance between two vectors
    # Argument:
                X: ndtensor
                Y: mdtensor
    # Returnes:
                D: (m x n) matrix
    '''
    # squared norm of each vector
    XX = tf.reduce_sum(tf.square(X), 1)
    YY = tf.reduce_sum(tf.square(Y), 1)
    # XX is a row vector and YY is a column vector
    XX = tf.reshape(XX, [-1, 1])
    YY = tf.reshape(YY, [1, -1])
    return XX + YY - 2 * tf.matmul(X, tf.transpose(Y))


@tf.function
def normalize(x_i):
    """
    Normalize the vector
    """
    min_i = tf.math.reduce_min(x_i)
    max_i = tf.math.reduce_max(x_i)
    x_i_normalized = (x_i - min_i) / (max_i - min_i)
    return x_i_normalized
