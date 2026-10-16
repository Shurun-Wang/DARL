import json
import os
import pickle
import numpy as np
from natsort import natsorted
from collections import defaultdict
import random
import sys

from pathlib import Path

# Support both direct script execution and python -m.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocessing.split_utils import create_subject_level_folds
from preprocessing.subject_groups import MDD_DUPLICATE_GROUPS, subject_group


cur_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(cur_path))

processed_data_path = os.path.join(root_path, 'data/mdd/processed_data_nonoverlap')
data_split_path = cur_path
os.makedirs(data_split_path, exist_ok=True)

sampling_rate = 256
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
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
                "group_id": subject_group("mdd", subject_name),
                "file": file_path,
                "label": eeg_data['Y']
            })
        except Exception as e:
            raise ValueError(f"Error loading {file_path}") from e
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
            "std": std,
            "split_seed": 42,
            "validation_fraction": 0.2,
            "split_level": "subject_group",
            "validation_fraction_unit": "groups",
            "duplicate_subject_groups": MDD_DUPLICATE_GROUPS,
            "window_seconds": 10,
            "overlap_ratio": 0.0
        }
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved to {save_path}")


def create_fully_independent_balanced_folds(h_folders, p_folders, n_folds=5):
    # Stratify unique groups, then expand back to the original subject IDs.
    # Thus ~20% of development groups form validation; subject counts may vary.
    members = defaultdict(list)
    for folder in natsorted(list(h_folders) + list(p_folders)):
        members[subject_group("mdd", Path(folder).name)].append(str(folder))
    representatives = {group: paths[0] for group, paths in members.items()}
    folds = create_subject_level_folds(
        [path for group, path in representatives.items() if group.startswith("H")],
        [path for group, path in representatives.items() if group.startswith("P")],
        n_folds=n_folds,
        validation_fraction=0.2,
        seed=42,
    )
    for fold in folds:
        group_sets = []
        for split, legacy in (("train_subjects", "train"),
                              ("validation_subjects", "val"),
                              ("test_subjects", "test")):
            groups = {subject_group("mdd", Path(path).name) for path in fold[split]}
            group_sets.append(groups)
            fold[split] = natsorted(path for group in groups for path in members[group])
            for prefix in ("H", "P"):
                fold[f'{prefix.lower()}_{legacy}_ids'] = [
                    int(Path(path).name[1:]) for path in fold[split]
                    if Path(path).name.startswith(prefix)]
        if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i)):
            raise ValueError('Duplicate group crosses split boundaries')
    return folds


def main_balanced():
    h_subject_folders = natsorted(
        os.path.join(processed_data_path, f) for f in os.listdir(processed_data_path)
        if f.startswith("H") and os.path.isdir(os.path.join(processed_data_path, f))
        and any(Path(processed_data_path, f).glob('*.pkl'))
    )

    p_subject_folders = natsorted(
        os.path.join(processed_data_path, f) for f in os.listdir(processed_data_path)
        if f.startswith("P") and os.path.isdir(os.path.join(processed_data_path, f))
        and any(Path(processed_data_path, f).glob('*.pkl'))
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
