import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.data_load_chbmit import read_chbmit_annotations
from src.data_preparation import separate_normal_seizures_registers, create_training_dataset
from src.model_definitions import TimeProtoNet, TimeProtoFlatNet
from src.model_training import train, train_prototype_network
from src.visualization import plot_tsne


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
    # Generate final balanced dataset
    X = X[selected_normal_indexes+seizure_indexes, :, :]
    y = y[selected_normal_indexes+seizure_indexes, :]
    y = to_categorical(y)

    # Balance train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Create with tensorflow format
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(64)

    # Create model
    input_shape = (X.shape[1], X.shape[2])
    pnet_model = TimeProtoNet(input_shape, 10, 2)
    # pnet_model = TimeProtoFlatNet(input_shape)
    print(pnet_model.complete_pnet.summary())

    # Train model
    optimizer = tf.keras.optimizers.Adam()
    train_history_df, prototype_checkpoints = train_prototype_network(pnet_model, train_dataset, test_dataset,
                                                                      optimizer, epochs_1=40, epochs_2=10,
                                                                      checkpoints_epoch=10)
    train_history_df[['train_class_bce', 'val_class_bce']].plot()
    train_history_df[['train_recons_mse', 'val_recons_mse']].plot()
    train_history_df[['train_reg1_loss', 'val_reg1_loss']].plot()
    train_history_df[['train_reg2_loss', 'val_reg2_loss']].plot()
    train_history_df[['train_total_loss', 'val_total_loss']].plot()

    # Predict test windows
    y_pred = pnet_model.classify(X_test).numpy()
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

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

    """# Plot prototypes
    for i in range(prototypes.shape[0]):
        fig, axs = plt.subplots(prototypes.shape[2])
        for channel in range(prototypes.shape[2]):
            axs[channel].plot(prototypes[i, :, channel])
            axs[channel].set_ylim([-1.05*150, 1.05*150])
        plt.show()"""

    # Inspect outputs for the pure prototypes
    y_proto = pnet_model.classify(prototypes).numpy()
    print(y_proto)

    # 2D tsne graph with classes and prototypes
    train_embeddings = pnet_model.encode(X_train)
    plot_tsne(train_embeddings.numpy(), classes=np.argmax(y_train, axis=1), prototypes_embeddings_list=prototype_checkpoints)
    print('Finished')






