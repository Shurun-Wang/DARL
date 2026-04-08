import json
import os
import pickle
import numpy as np
from natsort import natsorted
from collections import defaultdict
import random
import sys


cur_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(cur_path))

processed_data_path = os.path.join(root_path, 'data/idd/processed_data')
sampling_rate = 128
ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1','O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

num_channels = len(ch_names)
random.seed(42)


def load_subject_metadata(start, subject_folder):
    subject_data = []
    folder_name = os.path.basename(subject_folder)
    num = 1
    subject_num = int(folder_name[num:])
    subject_name = start + f"{subject_num}"

    for file in natsorted(f for f in os.listdir(subject_folder) if f.endswith('.pkl')):
        file_path = os.path.join(subject_folder, file)
        try:
            with open(file_path, 'rb') as f:
                eeg_data = pickle.load(f)
            subject_data.append({
                "subject_id": subject_num,
                "subject_name": subject_name,
                "file": file_path,
                "label": eeg_data['Y']
            })
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    return subject_data


def compute_normalization_params(data_list):
    """Calculate normalization parameters."""
    total_mean = np.zeros(num_channels)
    total_std = np.zeros(num_channels)
    max_val, min_val = -np.inf, np.inf
    count = 0

    for data in data_list:
        with open(data["file"], 'rb') as f:
            eeg = pickle.load(f)['X']

        max_val = max(max_val, eeg.max())
        min_val = min(min_val, eeg.min())
        total_mean += eeg.mean(axis=1)
        total_std += eeg.std(axis=1)
        count += 1

    return (total_mean / count).tolist(), (total_std / count).tolist(), max_val, min_val


def save_dataset(data_list, save_path, norm_params=None):
    if norm_params is None:
        print("Computing normalization parameters...")
        mean, std, max_val, min_val = compute_normalization_params(data_list)
    else:
        mean, std, max_val, min_val = norm_params

    dataset = {
        "subject_data": data_list,
        "dataset_info": {
            "sampling_rate": sampling_rate,
            "ch_names": ch_names,
            "min": min_val,
            "max": max_val,
            "mean": mean,
            "std": std
        }
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved to {save_path}")


def create_fully_independent_balanced_folds(h_folders, p_folders, n_folds=10):
    h_ids = [int(os.path.basename(f)[1:]) for f in h_folders]
    p_ids = [int(os.path.basename(f)[1:]) for f in p_folders]

    random.shuffle(h_ids)
    random.shuffle(p_ids)

    n_h = len(h_ids)
    n_p = len(p_ids)

    h_test_size = n_h // n_folds
    h_val_size = h_test_size

    p_test_size = n_p // n_folds
    p_val_size = p_test_size

    folds = []

    for fold in range(n_folds):
        h_test_start = fold * h_test_size
        h_test_end = (fold + 1) * h_test_size if fold < n_folds - 1 else n_h
        h_test_ids = h_ids[h_test_start:h_test_end]

        p_test_start = fold * p_test_size
        p_test_end = (fold + 1) * p_test_size if fold < n_folds - 1 else n_p
        p_test_ids = p_ids[p_test_start:p_test_end]

        h_remaining = h_ids[:h_test_start] + h_ids[h_test_end:]
        p_remaining = p_ids[:p_test_start] + p_ids[p_test_end:]

        random.shuffle(h_remaining)
        random.shuffle(p_remaining)

        h_val_ids = h_remaining[:h_val_size]
        h_train_ids = h_remaining[h_val_size:]

        p_val_ids = p_remaining[:p_val_size]
        p_train_ids = p_remaining[p_val_size:]

        folds.append({
            'h_train_ids': h_train_ids,
            'h_val_ids': h_val_ids,
            'h_test_ids': h_test_ids,
            'p_train_ids': p_train_ids,
            'p_val_ids': p_val_ids,
            'p_test_ids': p_test_ids
        })

    return folds


def main_balanced():
    h_subject_folders = natsorted(
        os.path.join(processed_data_path, f) for f in os.listdir(processed_data_path)
        if f.startswith("H") and os.path.isdir(os.path.join(processed_data_path, f))
    )

    p_subject_folders = natsorted(
        os.path.join(processed_data_path, f) for f in os.listdir(processed_data_path)
        if f.startswith("P") and os.path.isdir(os.path.join(processed_data_path, f))
    )

    folds = create_fully_independent_balanced_folds(h_subject_folders, p_subject_folders, n_folds=5)
    for fold_idx, fold in enumerate(folds):

        data_split_path = os.path.join(cur_path, f'json_{fold_idx + 1}')
        os.makedirs(data_split_path, exist_ok=True)

        save_train_path = os.path.join(data_split_path, 'train.json')
        save_val_path = os.path.join(data_split_path, 'val.json')
        save_test_path = os.path.join(data_split_path, 'test.json')

        all_train = []

        for subject_id in fold['h_train_ids']:
            subject_folder = os.path.join(processed_data_path, f"H{subject_id}")
            if os.path.exists(subject_folder):
                all_train.extend(load_subject_metadata('H', subject_folder))

        for subject_id in fold['p_train_ids']:
            subject_folder = os.path.join(processed_data_path, f"P{subject_id}")
            if os.path.exists(subject_folder):
                all_train.extend(load_subject_metadata('P', subject_folder))

        norm_params = compute_normalization_params(all_train)

        all_val = []

        for subject_id in fold['h_val_ids']:
            subject_folder = os.path.join(processed_data_path, f"H{subject_id}")
            if os.path.exists(subject_folder):
                all_val.extend(load_subject_metadata('H', subject_folder))

        for subject_id in fold['p_val_ids']:
            subject_folder = os.path.join(processed_data_path, f"P{subject_id}")
            if os.path.exists(subject_folder):
                all_val.extend(load_subject_metadata('P', subject_folder))

        all_test = []
        for subject_id in fold['h_test_ids']:
            subject_folder = os.path.join(processed_data_path, f"H{subject_id}")
            if os.path.exists(subject_folder):
                all_test.extend(load_subject_metadata('H', subject_folder))

        for subject_id in fold['p_test_ids']:
            subject_folder = os.path.join(processed_data_path, f"P{subject_id}")
            if os.path.exists(subject_folder):
                all_test.extend(load_subject_metadata('P', subject_folder))

        save_dataset(all_train, save_train_path, norm_params)
        save_dataset(all_val, save_val_path, norm_params)
        save_dataset(all_test, save_test_path, norm_params)

if __name__ == "__main__":
    main_balanced()