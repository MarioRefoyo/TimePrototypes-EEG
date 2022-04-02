import numpy as np
import pandas as pd
import pyedflib


# Define a class to work with EEG registers.
class Register:
    def __init__(self, name, fs, n_seizures):
        self.name = name
        self.fs = fs
        self.n_seizures = n_seizures
        self.seizures = []
        self.channels = []
        self.ictaltime = 0

    def addSeizure(self, start, end):
        self.ictaltime += end - start
        seizure = [start, end]
        self.seizures.append(seizure)


def read_chbmit_annotations(annotation_file):
    """
    Parse the annotations file of a patient to extract information such as registers including seizures, time of start and end of the seizures,
    and channel montages

    :param annotation: Read an annotation file of a particular patient
    :return:
        - registers (dict): Dictionary of registers objects
        - channel_index (dict): Dictionary containing the common channels for all the registers in a patient
    """
    with open(annotation_file) as f:
        # Declare registers dict containing the register for each key name
        registers_info = {}
        # Declare registers dict containing the number of channels
        channels_dict = {}
        # Initialize number of montages for a patient to 1
        n_montages = 1

        # Parse the annotation file
        for line in f:
            # If line matches to header of Data Sampling Rate declaration
            if "Data Sampling Rate" in line:
                line = line.split()
                # Extract fs
                fs = int(line[3])

            # If line matches to header of channel number - name declaration
            if "Channel " in line:
                line = line.split()
                # Extract channel name
                channel = line[2]
                if channel in channels_dict:
                    # Sum 1 to the counter of each channel apparition
                    channels_dict.update({channel: channels_dict[channel] + 1})
                else:
                    channels_dict[channel] = 1

            # If line matches to change in channel montage
            if "Channels changed" in line:
                n_montages += 1

            # If line matches file name with declaration
            elif "File Name" in line:
                name = line.split()[2]

                # Loop to parse all the possible seizures
                while True:
                    new_line = f.readline()

                    # If line matches to number of seizures declaration
                    if "Number of Seizures" in new_line:
                        n_seizures = int(new_line.split()[5])

                        # Create register object
                        register_info = Register(name, fs, n_seizures)

                        if n_seizures > 0:
                            # Loop to read all the seizures for a register
                            for i in range(n_seizures):
                                # Parse seizure information (start and end)
                                line1 = f.readline().split()
                                line2 = f.readline().split()
                                # Chapuza porque hay veces que aparece la palabra Time en la línea
                                if line1[3] == "Time:":
                                    start = int(line1[4])
                                    end = int(line2[4])
                                else:
                                    start = int(line1[3])
                                    end = int(line2[3])
                                # Add seizure information to the register info
                                register_info.addSeizure(start, end)

                        # Save the register in the registers dictionary
                        registers_info[name] = register_info
                        break

    # Get the list of common channels of every register for a patient
    common_channels = []
    [common_channels.append(key) for key in channels_dict.keys() if channels_dict[key] == n_montages]
    # Create the channel index dictionary as the combination of channel order and name. Only constructed with common
    # channels in all dataset
    channel_index = dict(zip(list(np.arange(len(common_channels))), common_channels))

    return registers_info, channel_index


def read_chbmit_register(filename, channels_to_read=None):
    """
    Read an edf file and return the data, sampling frequency and time axis of the file

    :param filename: name and path of the file to read
    :param channels_to_read: list of the channel names to read
    :return:
        - data (numpy array): eeg signals
        - fs (int): sampling frequency
        - time (numpy array): time axis
    """
    # Read an edf file
    f = pyedflib.EdfReader(filename)

    # if no channels are passed to the function uses the channels in the edf file
    if channels_to_read is None:
        channels_to_read = f.getSignalLabels()

    # Read channels available in the register
    channel_names = f.getSignalLabels()
    # Get sampling frequency
    fs = f.getSampleFrequencies()

    # Init array with zeros
    data = np.zeros((f.getNSamples()[0], len(channels_to_read)))
    # Read channels by order
    for i, channel in enumerate(channels_to_read):
        data[:, i] = f.readSignal(channel_names.index(channel))

    # Define time axis
    time = np.linspace(0, data.shape[1] / fs[0], data.shape[1])
    # Close file
    f._close()

    return data, fs[0], time


def generate_patient_data(patient_data_path, registers_info, channel_index):
    """
    Returns a single DataFrame with all the data of the registers passed to the function as a parameter. The returned df
    contains the seizure labels as well

    :param patient_data_path: path of the patient files
    :param registers_info: dictionary containing the registers information
    :param channel_index: channels to read and their index
    :return:
        - patient_data_df (DataFrame): dataframe with all the registers concatenated
    """
    # Init dataframe with all the patient data
    patient_data_df = pd.DataFrame()

    # Loop reading every register
    for register_name, register_info in registers_info.items():
        # Read register
        register_path = patient_data_path + register_name
        signals, original_fs, time_index = read_chbmit_register(register_path, channel_index.values())

        # Create register dataframe
        register_df = pd.DataFrame(signals, columns=channel_index.values(), dtype='float32')

        # Add column seizure column as boolean
        seizure = generate_seizure_label(register_info.seizures, register_info.fs, len(register_df))
        register_df['seizure'] = seizure
        register_df['seizure'] = register_df['seizure'].astype(bool)

        # Add column with the name of the register file
        register_df['file'] = [register_info.name]*len(register_df)

        # Concat to the dataframe containing all data
        patient_data_df = pd.concat([patient_data_df, register_df])

        print(f'Register {register_name} read and concatenated')

    return patient_data_df


def generate_seizure_label(seizures, fs, length):
    """
    Generate an array with 0's and 1's where there are normal and epileptic samples

    :param seizures: list of seiures start and end times
    :param fs: sampling frequency of the signal
    :param length: length of the vector
    :return:
        - seizure (numpy array): array with the labels
    """
    seizure = np.zeros(length)
    for n in range(len(seizures)):
        start = seizures[n][0] * fs
        end = seizures[n][1] * fs
        seizure[start:end] = np.ones(end - start)

    return seizure


if __name__ == "__main__":
    # Declare data path and the desired patient to read the data
    data_path = '../data/chb-mit/'
    patient = 'chb01'
    patient_data_path = f'{data_path}{patient}/'
    # Calculate the annotation file path for such patient
    annotation_fname = f'{patient}-summary.txt'
    annotation_data_path = patient_data_path + annotation_fname

    # Obtain register info and channel order for patient
    registers_info, channel_index = read_chbmit_annotations(annotation_data_path)

    # Es posible que el dataframe sea muy grande para que quepa en memoria. Si esto pasa, se puede eliminar algunos
    # registros de 'registers_info'

    # Get the dataframe with all the info of a patient
    patient_data_df = generate_patient_data(patient_data_path, registers_info, channel_index)

    # Store dataframe as pickle
    patient_data_df.to_pickle(f'../data/raw_concat/{patient}_data_concatenated.pickle')

    # Extract the data of the
    print(patient_data_df.info())
