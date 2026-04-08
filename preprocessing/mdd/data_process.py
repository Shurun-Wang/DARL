# https://figshare.com/articles/dataset/EEG_Data_New/4244171/2

import os
import mne
import numpy as np
import pickle


def iter_files(rootDir):
    files_dict = {'healthy': [], 'mdd': []}
    for file in os.listdir(rootDir):
        if 'EC' in file or 'EO' in file:  # 只保留 EC 和 EO 文件
            # 按空格分割文件名
            parts = file.split()
            if len(parts) >= 3:  # 假设文件名格式是类似 'H S1 EC' 或 '123_H S14 EO'
                # 判断是否为健康组或者 MDD 组
                if 'H' in parts[0]:  # 健康组
                    files_dict['healthy'].append(file)
                elif 'MDD' in parts[0]:  # MDD 组
                    files_dict['mdd'].append(file)
    return files_dict


def signal_seg(raw):
    eeg_array = raw.to_data_frame().values
    eeg_array = eeg_array[:, 1:]  # drop time column
    points, chs = eeg_array.shape
    print(f"EEG data shape for healthy S{subject_idx}: {eeg_array.shape}")

    win_len = 10 * 256
    hop_len = 5 * 256

    num_windows = (points - win_len) // hop_len + 1
    if num_windows <= 0:
        return np.empty((0, chs, win_len))

    segments = np.empty((num_windows, chs, win_len))
    for i in range(num_windows):
        start = i * hop_len
        end = start + win_len
        segments[i] = eeg_array[start:end, :].T

    return segments



def save_samples_to_pkl(eeg_array, label, output_dir="eeg_samples"):
    for i in range(eeg_array.shape[0]):
        sample_data = {
            'X': eeg_array[i],  # (channel, length)
            'Y': label
        }
        # 保存为单独的pkl文件
        filename = os.path.join(output_dir, f"sample_{i:04d}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(sample_data, f)


if __name__ == '__main__':
    cur_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(os.path.dirname(cur_path))

    selected_channels = ['EEG Fp1-LE', 'EEG Fp2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
                         'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE',
                         'EEG T5-LE', 'EEG T6-LE', 'EEG Fz-LE', 'EEG Cz-LE', 'EEG Pz-LE']

    rootDir = root_path + '/data/mdd/raw_data'
    files_dict = iter_files(rootDir)

    for subject_idx in range(1, 31):
        preprocess_dir = root_path +'/data/mdd/processed_data/H'+str(subject_idx-1)
        if not os.path.exists(preprocess_dir):
            os.makedirs(preprocess_dir)

        ec_file = [file for file in files_dict['healthy'] if f"S{subject_idx}"==file.split()[1] and "EC" in file]
        eo_file = [file for file in files_dict['healthy'] if f"S{subject_idx}"==file.split()[1] and "EO" in file]

        if ec_file:
            raw_ec = mne.io.read_raw_edf(os.path.join(rootDir, ec_file[0]), preload=True)
            raw_ec.pick_channels(selected_channels, ordered=True)
            raw_ec.set_eeg_reference()
            raw_ec.filter(l_freq=0.3, h_freq=45)
            eeg_array_ec = signal_seg(raw_ec)

        if eo_file:
            raw_eo = mne.io.read_raw_edf(os.path.join(rootDir, eo_file[0]), preload=True)
            raw_eo.pick_channels(selected_channels, ordered=True)
            raw_eo.set_eeg_reference()
            raw_eo.filter(l_freq=0.3, h_freq=45)
            eeg_array_eo = signal_seg(raw_eo)

        if raw_ec and raw_eo:
            eeg_array = np.concatenate([eeg_array_ec, eeg_array_eo], axis=0)
        elif raw_ec:
            eeg_array = eeg_array_ec
        elif raw_eo:
            eeg_array = eeg_array_eo
        else:
            continue

        label = 0
        save_samples_to_pkl(eeg_array, label, output_dir=preprocess_dir)

    for subject_idx in range(1, 35):

        preprocess_dir = root_path + '/data/mdd/processed_data/P'+str(subject_idx-1)
        if not os.path.exists(preprocess_dir):
            os.makedirs(preprocess_dir)

        ec_file = [file for file in files_dict['mdd'] if f"S{subject_idx}"==file.split()[1] and "EC" in file]
        eo_file = [file for file in files_dict['mdd'] if f"S{subject_idx}"==file.split()[1] and "EO" in file]

        if ec_file:
            raw_ec = mne.io.read_raw_edf(os.path.join(rootDir, ec_file[0]), preload=True)
            raw_ec.pick_channels(selected_channels, ordered=True)
            raw_ec.set_eeg_reference()
            raw_ec.filter(l_freq=0.3, h_freq=45)
            eeg_array_ec = signal_seg(raw_ec)


        if eo_file:
            raw_eo = mne.io.read_raw_edf(os.path.join(rootDir, eo_file[0]), preload=True)
            raw_eo.pick_channels(selected_channels, ordered=True)
            raw_eo.set_eeg_reference()
            raw_eo.filter(l_freq=0.3, h_freq=45)
            eeg_array_eo = signal_seg(raw_eo)

        if raw_ec and raw_eo:
            eeg_array = np.concatenate([eeg_array_ec, eeg_array_eo], axis=0)  #
        elif raw_ec:
            eeg_array = eeg_array_ec
        elif raw_eo:
            eeg_array = eeg_array_eo
        else:
            continue

        label = 1
        save_samples_to_pkl(eeg_array, label, output_dir=preprocess_dir)
