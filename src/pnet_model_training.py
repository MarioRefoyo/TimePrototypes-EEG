import time
import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_load_chbmit import read_chbmit_annotations
from model_training import separate_normal_seizures_registers, create_training_dataset
from custom_layers.prototypes import PrototypeLayer

MSE = tf.keras.losses.MeanSquaredError()
BCE = tf.keras.losses.BinaryCrossentropy()


class TimeProtoNet(tf.keras.Model):
    # ToDo: Add params
    def __init__(self, input_shape):
        super(TimeProtoNet, self).__init__()

        # Get number of channels
        n_channels = input_shape[-1]

        # Create encoder
        visible = tf.keras.layers.Input(shape=input_shape)
        conv1 = tf.keras.layers.Conv1D(32, kernel_size=8, activation='relu', padding="same")(visible)
        pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
        conv2 = tf.keras.layers.Conv1D(32, kernel_size=4, activation='relu', padding="same")(pool1)
        pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)
        conv3 = tf.keras.layers.Conv1D(10, kernel_size=4, activation='relu', padding="same")(pool2)
        pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv3)
        latent = tf.keras.layers.Flatten()(pool3)
        self.encoder = tf.keras.models.Model(inputs=visible, outputs=latent)

        # Create decoder
        decoder_input = tf.keras.layers.Input(shape=(640,))
        reshaped = tf.keras.layers.Reshape(target_shape=(64, 10))(decoder_input)
        upconv1 = tf.keras.layers.Conv1DTranspose(10, kernel_size=4, activation='relu', strides=2, padding="same")(reshaped)
        upconv2 = tf.keras.layers.Conv1DTranspose(32, kernel_size=4, activation='relu', strides=2, padding="same")(upconv1)
        upconv3 = tf.keras.layers.Conv1DTranspose(32, kernel_size=8, activation='relu', strides=2, padding="same")(upconv2)
        decoded = tf.keras.layers.Conv1D(n_channels, kernel_size=3, strides=1, padding='same', name='reconstruction')(upconv3)  # No activation
        self.decoder = tf.keras.models.Model(inputs=decoder_input, outputs=decoded)

        # Create prototypical layer and posterior dense
        pnet_input = tf.keras.layers.Input(shape=(640,))
        p_layer = PrototypeLayer(n_prototypes=20, name='proto_layer')(pnet_input)
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
        return self.pnet.get_layer('proto_layer').get_weights()[0]


@tf.function
def compute_pnet_loss(model, x, y):

    # Generate predictions
    z = model.encode(x)
    x_reconstructed = model.decode(z)
    y_pred = model.classify(x)

    # Calculate the classification loss
    class_loss = BCE(y, y_pred)
    # Calculate the reconstruction loss
    recons_loss = MSE(x, x_reconstructed)

    # Calculate total loss
    total = class_loss + 1000*recons_loss

    return total


@tf.function
def train_step(model, x, y, optimizer):

    with tf.GradientTape() as tape:
        # Compute loss
        loss = compute_pnet_loss(model, x, y)
    gradients_of_pnet = tape.gradient(loss, pnet_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_pnet, pnet_model.trainable_variables))
    return loss


def train(pnet_model, train_dataset, val_dataset, optimizer, epochs):
    for epoch in range(epochs):
        start = time.time()
        for x, y in train_dataset:
            train_loss = train_step(pnet_model, x, y, optimizer)
        end_time = time.time()

        # Calculate loss in the validation set
        val_loss = '-'
        """for sample_val_batch in val_dataset:
            X_val, outputs_val = sample_val_batch
            y_val, _ = outputs_val
            # Predict
            y_val_pred, X_val_reconstructed = pnet_model.predict(X_val)
            val_loss = compute_pnet_loss(y_val, y_val_pred, X_val, X_val_reconstructed)"""

        print(f'Epoch: {epoch+1}, Train loss: {train_loss}, Test loss: {val_loss}, time elapse for current epoch: '
              f'{end_time - start}')


if __name__ == '__main__':

    print(tf.config.list_physical_devices('GPU'))

    # Select patient
    patient = 'chb18'
    # Select window size
    window_size = 512

    # Read patient annotations
    annotation_data_path = f'../data/chb-mit/{patient}/{patient}-summary.txt'
    registers_info, channel_index = read_chbmit_annotations(annotation_data_path)

    # Separate files with seizures and with normal activity in different lists
    seizure_files, normal_files = separate_normal_seizures_registers(registers_info)

    # Read patient dataset with all registers
    with open(f'../data/processed/{patient}_{window_size}_reshaped_data.pickle', "rb") as input_file:
        complete_dataset = pickle.load(input_file)

    # Filter dataset to get the training data
    # Get normal files to include in the training dataset
    # ToDo: revisar aplicacion del random
    number_of_normal_files_to_use = 4
    random_seed = 42
    random.seed(random_seed)
    normal_files_to_use = random.choices(normal_files, k=number_of_normal_files_to_use)

    # Get the complete list of files to train
    files_used_to_train = normal_files_to_use + seizure_files

    # Create the final dataset
    X, y = create_training_dataset(complete_dataset, normal_files_to_use, seizure_files, only_seizures=True)

    # MinMax scaler
    max_register_value = np.max(np.abs(X))
    X = X / max_register_value

    # Select only a certain number of normal windows as a multiplier of seizure windows
    normal_mult = 1
    normal_indexes = list(np.where(y == 0)[0])
    seizure_indexes = list(np.where(y == 1)[0])
    selected_normal_indexes = random.choices(normal_indexes, k=normal_mult*len(seizure_indexes))
    X = X[selected_normal_indexes+seizure_indexes, :, :]
    y = y[selected_normal_indexes+seizure_indexes, :]

    # Balance train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Create with tensorflow format
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(64)

    # Create model
    input_shape = (X.shape[1], X.shape[2])
    #pnet_model = make_pnet_model(input_shape)
    pnet_model = TimeProtoNet(input_shape)
    print(pnet_model.complete_pnet.summary())

    # Train model
    optimizer = tf.keras.optimizers.Adam()
    train(pnet_model, train_dataset, test_dataset, optimizer, epochs=50)

    # Predict test windows
    y_pred = pnet_model.classify(X_test).numpy()
    y_pred = (y_pred > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))

    # Plot reconstructions
    X_reconstructed = pnet_model.reconstruct(X_test)
    for i in range(5):
        fig, axs = plt.subplots(X_test.shape[2])
        for channel in range(X_test.shape[2]):
            axs[channel].plot(X_test[i, :, channel] * max_register_value, color='blue', label='original')
            axs[channel].plot(X_reconstructed[i, :, channel] * max_register_value, color='red', label='reconstructed')
            axs[channel].set_ylim([-1.05*50, 1.05*50])
        plt.show()

    # Inspect decoded prototypes
    latent_prototypes = pnet_model.get_prototypes()
    # Decode prototypes
    prototypes = pnet_model.decode(latent_prototypes).numpy() * max_register_value

    # Plot prototypes
    for i in range(prototypes.shape[0]):
        fig, axs = plt.subplots(prototypes.shape[2])
        for channel in range(prototypes.shape[2]):
            axs[channel].plot(prototypes[i, :, channel])
            axs[channel].set_ylim([-1.05*150, 1.05*150])
        plt.show()

    # Inspect outputs for the pure prototypes
    y_proto = pnet_model.classify(prototypes).numpy()
    #y_proto = (y_proto > 0.5).astype("int32")
    print("Finished")






