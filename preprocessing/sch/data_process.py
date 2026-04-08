# Graph-based analysis of brain connectivity in  schizophrenia

from glob import glob # Read files from folder
import os # Operating System interfaces

import mne # Analyze EEG data
import pickle
import numpy as np # Calculations
import pandas as pd # Dataframes


def save_samples_to_pkl(eeg_array, label, output_dir="eeg_samples"):
    for i in range(eeg_array.shape[0]):
        sample_data = {
            'X': eeg_array[i],  # (channel, length)
            'Y': label
        }

        filename = os.path.join(output_dir, f"sample_{i:04d}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(sample_data, f)


# Function to read data from file paths
def read_data(file_path):

    data = mne.io.read_raw_edf(file_path, preload=True)
    data.set_eeg_reference()
    data.filter(l_freq=0.3, h_freq=45)
    data, _ = data[:, :]
    return data


def create_overlapping_segments(data, window_samples, overlap_ratio=0.5):
    step = int(window_samples * (1 - overlap_ratio))

    segments = []
    n_samples = data.shape[1]

    for start in range(0, n_samples - window_samples + 1, step):
        end = start + window_samples
        segment = data[:, start:end]
        segments.append(segment)

    return np.array(segments)


if __name__ == '__main__':
    # Access all edf files from directory
    cur_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(os.path.dirname(cur_path))

    sampling_rate = 250
    raw_data_path = os.path.join(root_path, 'data/sch/raw_data')

    all_file_path = glob(raw_data_path + '/*.edf')
    healthy_file_path = [i for i in all_file_path if 'h' in i.split('/')[-1]]
    patient_file_path = [i for i in all_file_path if 's' in i.split('/')[-1]]

    window_sec=10

    for num in range(len(healthy_file_path)):
        data = read_data(healthy_file_path[num])
        segment_length = window_sec * sampling_rate
        segments = create_overlapping_segments(data, window_samples=segment_length, overlap_ratio=0.5)
        processed_data_path = os.path.join(root_path, 'data/sch/processed_data')

        processed_data_path = processed_data_path +'/'+'H' + str(num)
        os.makedirs(processed_data_path, exist_ok=True)
        label = 0
        save_samples_to_pkl(segments, label, output_dir=processed_data_path)

    for num in range(len(patient_file_path)):
        data = read_data(patient_file_path[num])

        segment_length = window_sec * sampling_rate
        segments = create_overlapping_segments(data, window_samples=segment_length, overlap_ratio=0.5)
        processed_data_path = os.path.join(root_path, 'data/sch/processed_data')

        processed_data_path = processed_data_path +'/'+'P' + str(num)
        os.makedirs(processed_data_path, exist_ok=True)
        label = 1
        save_samples_to_pkl(segments, label, output_dir=processed_data_path)