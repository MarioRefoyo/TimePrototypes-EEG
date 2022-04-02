import numpy as np
import pandas as pd
import pickle


def truncate_register_samples(data, w_size):
    n_samples, n_channels = data.shape
    n_windows = n_samples // w_size
    trunc_data = data[:n_windows*w_size, :]
    return trunc_data, n_windows


def resize_channel_data(raw_channel_data, w_size):
    # Create empty vector for data samples
    n_samples, n_channels = raw_channel_data.shape
    n_windows = n_samples // w_size
    channel_data_reshaped = np.zeros((n_windows, w_size, n_channels))
    # Iterate over channels
    for i in range(n_channels):
        data_channel_reshaped = raw_channel_data[:, i].reshape(-1, w_size)
        channel_data_reshaped[:, :, i] = data_channel_reshaped

    return channel_data_reshaped


def generate_seizure_labels(sample_labels, w_size):
    # Reshape and sum the rows
    seizure_labels = sample_labels.reshape(-1, w_size).sum(axis=1)
    # Put target as one if at least half of the window contain seizure samples
    seizure_labels[seizure_labels > (w_size//2)] = 1
    seizure_labels[seizure_labels <= (w_size//2)] = 0

    return seizure_labels


def create_dataset(raw_patient_dataset, w_size):
    registers_data = {}
    for register in raw_patient_dataset['file'].unique():
        print(f'Working on {register}')

        # Select file
        file_data = raw_patient_dataset[raw_patient_dataset['file'] == register].copy()
        target_data = file_data['seizure'].astype(int).values.reshape(-1, 1)
        raw_channel_data = file_data.drop(['file', 'seizure'], axis=1).values

        # Resize with the specified window_size
        trunc_register, n_windows = truncate_register_samples(raw_channel_data, w_size)
        trunc_labels, _ = truncate_register_samples(target_data, w_size)

        # Reshape the raw eeg time series
        channel_data_reshaped = resize_channel_data(trunc_register, w_size)

        # Generate seizure windows_labels
        seizure_labels = generate_seizure_labels(trunc_labels, w_size)

        # Append to dictionary
        registers_data[register] = (channel_data_reshaped, seizure_labels)

    return registers_data


if __name__ == '__main__':

    # ---------- CREATE DATASET --------- #
    # Select patient
    patient = 'chb01'
    # Select window size
    window_size = 512

    # Load concatenated data for the patient
    raw_patient_dataset = pd.read_pickle('../data/raw_concat/chb01_data_concatenated.pickle')

    # Create dataset with labels
    registers_dataset = create_dataset(raw_patient_dataset, window_size)

    # Save as pickle
    with open(f'../data/processed/{patient}_{window_size}_reshaped_data.pickle', 'wb') as handle:
        pickle.dump(registers_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)



