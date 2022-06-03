import pickle
import random
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.data_load_chbmit import read_chbmit_annotations
from src.data_preparation import separate_normal_seizures_registers, create_training_dataset


def make_vanilla_cnn_model(input_shape):
    visible = tf.keras.layers.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv1D(32, kernel_size=8, activation='relu')(visible)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
    conv2 = tf.keras.layers.Conv1D(64, kernel_size=4, activation='relu')(pool1)
    pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)
    conv3 = tf.keras.layers.Conv1D(128, kernel_size=4, activation='relu')(pool2)
    pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv3)
    flat = tf.keras.layers.Flatten()(pool3)
    hidden1 = tf.keras.layers.Dense(32, activation='relu')(flat)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden1)
    model = tf.keras.models.Model(inputs=visible, outputs=output)
    return model


if __name__ == '__main__':
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
    model = make_vanilla_cnn_model(input_shape)
    model.summary()

    # Train model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=50)

    # Predict test windows
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))





