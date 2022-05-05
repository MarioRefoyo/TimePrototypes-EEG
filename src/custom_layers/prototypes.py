import tensorflow as tf
from utils import pairwise_dist


class PrototypeLayer(tf.keras.layers.Layer):
    def __init__(self, n_prototypes, name='protolayer'):
        """
        Implimentataion of the Prototype layer
        # Returen:
            output: class distribution
            prototype_distances
            feature_vector_distances
            prototypes: a tensor (n_prototypes, n_features)
            a: prototype distribution, return the 'a' layer output as describe in the paper
        """
        super(PrototypeLayer, self).__init__(name=name)

        """self.prototype_feature_vectors = tf.Variable(tf.random.uniform(shape=[n_prototypes, input_shape],
                                                                       dtype=tf.float32),
                                                     name='prototype_feature_vectors')"""
        self.n_prototypes = n_prototypes

    def build(self, input_shape):
        self.prototype_feature_vectors = self.add_weight(shape=(self.n_prototypes, input_shape[-1]),
                                                         initializer="random_normal",
                                                         trainable=True)

    def call(self, inputs):
        prototype_distances = pairwise_dist(inputs, self.prototype_feature_vectors)
        return prototype_distances
