import numpy as np


def separate_normal_seizures_registers(registers_info_dict):
    # Initialize the lists
    seizure_registers = []
    normal_registers = []

    for register_name, register in registers_info_dict.items():
        if register.n_seizures > 0:
            seizure_registers.append(register_name)
        else:
            normal_registers.append(register_name)

    return seizure_registers, normal_registers


def create_training_dataset(complete_data_dict, normal_registers, seizure_registers, only_seizures=True):
    # Get dimension sizes
    _, w_size, n_channels = list(complete_data_dict.values())[0][0].shape

    # Init empty vectors of data
    x_complete = np.zeros((0, w_size, n_channels))
    y_complete = np.zeros((0, 1))

    for register in normal_registers:
        x_register, y_register = complete_data_dict[register]
        x_complete = np.vstack((x_complete, x_register))
        y_complete = np.vstack((y_complete, y_register))

    for register in seizure_registers:
        x_register, y_register = complete_data_dict[register]
        if only_seizures:
            seizure_index = np.where(y_register == 1)[0]
            x_register = x_register[seizure_index, :, :]
            y_register = y_register[seizure_index, :]
        x_complete = np.vstack((x_complete, x_register))
        y_complete = np.vstack((y_complete, y_register))

    return x_complete, y_complete.astype(int)