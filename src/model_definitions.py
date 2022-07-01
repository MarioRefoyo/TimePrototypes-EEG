import numpy as np
import tensorflow as tf
from src.custom_layers.prototypes import PrototypeLayer


def create_init_weights(n_prototypes_class, n_classes):
    w = np.ones((n_prototypes_class*n_classes, n_classes)) * -0.5
    for i in range(n_classes):
        w[i*n_prototypes_class:(i+1)*n_prototypes_class, i] = [1]*n_prototypes_class
    return w


class TimeProtoNet(tf.keras.Model):

    # ToDo: Add params
    def __init__(self, input_shape, n_prototypes_class, n_classes):
        super(TimeProtoNet, self).__init__()

        # Get number of channels
        n_channels = input_shape[-1]

        # Create encoder
        visible = tf.keras.layers.Input(shape=input_shape)
        conv1 = tf.keras.layers.Conv1D(512, kernel_size=8, activation='relu', padding="same")(visible)
        pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
        conv2 = tf.keras.layers.Conv1D(256, kernel_size=4, activation='relu', padding="same")(pool1)
        pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)
        conv3 = tf.keras.layers.Conv1D(100, kernel_size=4, activation='relu', padding="same")(pool2)
        pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv3)
        latent = tf.keras.layers.Flatten()(pool3)
        self.encoder = tf.keras.models.Model(inputs=visible, outputs=latent)

        # Create decoder
        decoder_input = tf.keras.layers.Input(shape=(6400,))
        reshaped = tf.keras.layers.Reshape(target_shape=(64, 100))(decoder_input)
        upconv1 = tf.keras.layers.Conv1DTranspose(100, kernel_size=4, activation='relu', strides=2, padding="same")(reshaped)
        upconv2 = tf.keras.layers.Conv1DTranspose(256, kernel_size=4, activation='relu', strides=2, padding="same")(upconv1)
        upconv3 = tf.keras.layers.Conv1DTranspose(512, kernel_size=8, activation='relu', strides=2, padding="same")(upconv2)
        decoded = tf.keras.layers.Conv1D(n_channels, kernel_size=3, strides=1, padding='same', name='reconstruction')(upconv3)  # No activation
        self.decoder = tf.keras.models.Model(inputs=decoder_input, outputs=decoded)

        # Create prototypical layer and posterior dense
        pnet_input = tf.keras.layers.Input(shape=(6400,))
        p_layer = PrototypeLayer(n_prototypes_class=n_prototypes_class, n_classes=n_classes,
                                 name='proto_layer')(pnet_input)
        logits = tf.keras.layers.Dense(n_classes, kernel_initializer=tf.constant_initializer(
            create_init_weights(n_prototypes_class=n_prototypes_class, n_classes=n_classes)),
            name='classification_logits')(p_layer)
        output = tf.keras.layers.Activation(tf.keras.activations.softmax)(logits)
        self.pnet = tf.keras.models.Model(inputs=pnet_input, outputs=output)

        # Create complete model for training
        complete_pnet_input = tf.keras.layers.Input(shape=input_shape)
        encoded_sample = self.encoder(complete_pnet_input)
        decoded_sample = self.decoder(encoded_sample)
        classification_output = self.pnet(encoded_sample)
        self.complete_pnet = tf.keras.models.Model(inputs=complete_pnet_input, outputs=[classification_output, decoded_sample])

    def encode(self, x):
        latent_code = self.encoder(x)
        return latent_code

    def decode(self, z):
        reconstructed_input = self.decoder(z)
        return reconstructed_input

    def classify(self, x):
        latent_code = self.encode(x)
        classification = self.pnet(latent_code)
        return classification

    def reconstruct(self, x):
        latent_code = self.encode(x)
        reconstruction = self.decode(latent_code)
        return reconstruction

    def get_prototypes(self):
        return self.pnet.get_layer('proto_layer').weights


class TimeProtoFlatNet(tf.keras.Model):

    # ToDo: Add params
    def __init__(self, input_shape):
        super(TimeProtoFlatNet, self).__init__()

        # Get number of channels
        n_samples = input_shape[0]
        n_channels = input_shape[1]

        # Create encoder
        visible = tf.keras.layers.Input(shape=input_shape)
        flat = tf.keras.layers.Reshape(target_shape=(n_samples*n_channels, 1))(visible)
        conv1 = tf.keras.layers.Conv1D(512, kernel_size=8, activation='relu', padding="same")(flat)
        pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
        conv2 = tf.keras.layers.Conv1D(256, kernel_size=4, activation='relu', padding="same")(pool1)
        pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)
        conv3 = tf.keras.layers.Conv1D(100, kernel_size=4, activation='relu', padding="same")(pool2)
        pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv3)
        conv4 = tf.keras.layers.Conv1D(1, kernel_size=4, activation='relu', padding="same")(pool3)
        pool4 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv4)
        latent = tf.keras.layers.Flatten()(pool4)
        self.encoder = tf.keras.models.Model(inputs=visible, outputs=latent)

        # Create decoder
        decoder_input = tf.keras.layers.Input(shape=(1088,))
        reshaped = tf.keras.layers.Reshape(target_shape=(544, 2))(decoder_input)
        upconv1 = tf.keras.layers.Conv1DTranspose(2, kernel_size=4, activation='relu', strides=2, padding="same")(reshaped)
        upconv2 = tf.keras.layers.Conv1DTranspose(10, kernel_size=4, activation='relu', strides=2, padding="same")(upconv1)
        upconv3 = tf.keras.layers.Conv1DTranspose(32, kernel_size=4, activation='relu', strides=2, padding="same")(upconv2)
        upconv4 = tf.keras.layers.Conv1DTranspose(32, kernel_size=8, activation='relu', strides=2, padding="same")(upconv3)
        decoded = tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='same', name='reconstruction')(upconv4)  # No activation
        decoded = tf.keras.layers.Reshape(target_shape=(n_samples, n_channels))(decoded)
        self.decoder = tf.keras.models.Model(inputs=decoder_input, outputs=decoded)

        # Create prototypical layer and posterior dense
        pnet_input = tf.keras.layers.Input(shape=(1088,))
        p_layer = PrototypeLayer(n_prototypes_class=10, n_classes=2, name='proto_layer')(pnet_input)
        logits = tf.keras.layers.Dense(1, name='classification_logits')(p_layer)
        output = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(logits)
        self.pnet = tf.keras.models.Model(inputs=pnet_input, outputs=output)

        # Create complete model for training
        complete_pnet_input = tf.keras.layers.Input(shape=input_shape)
        encoded_sample = self.encoder(complete_pnet_input)
        decoded_sample = self.decoder(encoded_sample)
        classification_output = self.pnet(encoded_sample)
        self.complete_pnet = tf.keras.models.Model(inputs=complete_pnet_input, outputs=[classification_output, decoded_sample])

    def encode(self, x):
        latent_code = self.encoder(x)
        return latent_code

    def decode(self, z):
        reconstructed_input = self.decoder(z)
        return reconstructed_input

    def classify(self, x):
        latent_code = self.encode(x)
        classification = self.pnet(latent_code)
        return classification

    def reconstruct(self, x):
        latent_code = self.encode(x)
        reconstruction = self.decode(latent_code)
        return reconstruction

    def get_prototypes(self):
        return self.pnet.get_layer('proto_layer').weights
