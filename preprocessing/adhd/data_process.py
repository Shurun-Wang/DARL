# https://ieee-dataport.org/open-access/eeg-data-adhd-control-children?check_logged_in=1

from glob import glob
import os

import mne
import pickle
import numpy as np
from scipy.io import loadmat



def create_overlapping_segments(data, window_samples, overlap_ratio=0.5):
    step = int(window_samples * (1 - overlap_ratio))

    segments = []
    n_samples = data.shape[1]

    for start in range(0, n_samples - window_samples + 1, step):
        end = start + window_samples
        segment = data[:, start:end]
        segments.append(segment)

    return np.array(segments)


def save_samples_to_pkl(eeg_array, label, output_dir="eeg_samples"):
    for i in range(eeg_array.shape[0]):
        sample_data = {
            'X': eeg_array[i],  #  (channel, length)
            'Y': int(label)
        }

        filename = os.path.join(output_dir, f"sample_{i:04d}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(sample_data, f)


# Function to read data from file paths
def read_data(file_path, STANDARD_CHANNELS):
    # Read in raw edf data from path
    data = loadmat(file_path)
    sub_str = file_path.split('/')[-1].split('.')[0]
    data = data[sub_str].T
    data = data * 1e-6
    sfreq = 128
    ch_types = ["eeg"] * data.shape[0]
    info = mne.create_info(ch_names=STANDARD_CHANNELS, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020", on_missing="ignore")
    raw.set_eeg_reference()
    raw.filter(l_freq=0.3, h_freq=45)
    data, _ = raw[:, :]
    return data


if __name__ == '__main__':

    cur_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(os.path.dirname(cur_path))

    sampling_rate = 128
    window_sec = 10
    STANDARD_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']

    control_data_path = os.path.join(root_path, 'data/adhd/raw_data/control')
    all_control_path = glob(control_data_path + '/*.mat')


    for num in range(len(all_control_path)):
        data = read_data(all_control_path[num], STANDARD_CHANNELS)

        segment_length = window_sec * sampling_rate
        segments = create_overlapping_segments(data, window_samples=segment_length, overlap_ratio=0.5)
        processed_data_path = os.path.join(root_path, 'data/adhd/processed_data')

        processed_data_path = processed_data_path +'/'+'H' + str(num)
        os.makedirs(processed_data_path, exist_ok=True)
        label = 0
        save_samples_to_pkl(segments, label, output_dir=processed_data_path)

    patient_data_path = os.path.join(root_path, 'data/adhd/raw_data/patient')
    all_patient_path = glob(patient_data_path + '/*.mat')
    for num in range(len(all_patient_path)):
        data = read_data(all_patient_path[num], STANDARD_CHANNELS)

        segment_length = window_sec * sampling_rate
        segments = create_overlapping_segments(data, window_samples=segment_length, overlap_ratio=0.5)
        processed_data_path = os.path.join(root_path, 'data/adhd/processed_data')

        processed_data_path = processed_data_path +'/'+'P' + str(num)
        os.makedirs(processed_data_path, exist_ok=True)
        label = 1
        save_samples_to_pkl(segments, label, output_dir=processed_data_path)

