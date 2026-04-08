# EEG dataset of individuals with intellectual and developmental disorder and healthy controls under rest and music stimuli
from glob import glob # Read files from folder
import os
import mne
import pickle
import numpy as np



def save_samples_to_pkl(eeg_array, label, output_dir="eeg_samples"):
    for i in range(eeg_array.shape[0]):
        sample_data = {
            'X': eeg_array[i],  # 形状为 (channel, length)
            'Y': int(label)  # 标签值
        }
        # 保存为单独的pkl文件
        filename = os.path.join(output_dir, f"sample_{i:04d}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(sample_data, f)


# Function to read data from file paths
def read_control_data(file_path, num):
    # Read in raw edf data from path
    raw = mne.io.read_raw_eeglab(file_path+'/CGS0'+str(num+1)+'_Rest.set', preload=True)
    raw.set_eeg_reference()
    raw.filter(l_freq=0.3, h_freq=45)
    data, _ = raw[:]
    return data


def read_patient_data(file_path, num):
    # Read in raw edf data from path
    raw = mne.io.read_raw_eeglab(file_path+'/NDS00'+str(num+1)+'_Rest.set', preload=True)
    raw.set_eeg_reference()
    raw.filter(l_freq=0.3, h_freq=45)
    data, _ = raw[:]
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
    cur_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(os.path.dirname(cur_path))

    sampling_rate = 128

    window_sec = 10

    control_data_path = os.path.join(root_path, 'data/idd/raw_data/control')
    for num in range(7):
        data = read_control_data(control_data_path, num)
        segment_length = window_sec * sampling_rate
        segments = create_overlapping_segments(data, window_samples=segment_length, overlap_ratio=0.5)
        processed_data_path = os.path.join(root_path, 'data/idd/processed_data')

        processed_data_path = processed_data_path +'/'+'H' + str(num)
        os.makedirs(processed_data_path, exist_ok=True)
        label = 0
        save_samples_to_pkl(segments, label, output_dir=processed_data_path)

    patient_data_path = os.path.join(root_path, 'data/idd/raw_data/patient')
    for num in range(7):
        data = read_patient_data(patient_data_path, num)

        segment_length = window_sec * sampling_rate
        segments = create_overlapping_segments(data, window_samples=segment_length, overlap_ratio=0.5)
        processed_data_path = os.path.join(root_path, 'data/idd/processed_data')

        processed_data_path = processed_data_path +'/'+'P' + str(num)
        os.makedirs(processed_data_path, exist_ok=True)
        label = 1
        save_samples_to_pkl(segments, label, output_dir=processed_data_path)

