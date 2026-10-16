"""Deterministic subject-level cross-validation split utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def create_subject_level_folds(
    healthy_folders: Iterable[str],
    patient_folders: Iterable[str],
    *,
    n_folds: int = 5,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> list[dict[str, list[str]]]:
    """Create cross-subject test folds and a subject-level validation split.

    Each outer test fold is produced by stratified K-fold cross-validation. The
    validation set is then sampled from the remaining training subjects only.
    The same single seed controls both operations.
    """
    subjects = [str(Path(path)) for path in healthy_folders]
    labels = [0] * len(subjects)
    patient_subjects = [str(Path(path)) for path in patient_folders]
    subjects.extend(patient_subjects)
    labels.extend([1] * len(patient_subjects))

    subjects_array = np.asarray(subjects, dtype=object)
    labels_array = np.asarray(labels, dtype=int)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds: list[dict[str, list[str]]] = []
    for fold_index, (development_indices, test_indices) in enumerate(
        splitter.split(subjects_array, labels_array), start=1
    ):
        development_subjects = subjects_array[development_indices]
        development_labels = labels_array[development_indices]
        class_count = len(np.unique(development_labels))
        validation_size = max(
            class_count,
            int(round(len(development_subjects) * validation_fraction)),
        )
        train_subjects, validation_subjects = train_test_split(
            development_subjects,
            test_size=validation_size,
            random_state=seed,
            shuffle=True,
            stratify=development_labels,
        )

        fold = {
            "train_subjects": sorted(train_subjects.tolist()),
            "validation_subjects": sorted(validation_subjects.tolist()),
            "test_subjects": sorted(subjects_array[test_indices].tolist()),
        }
        for split_name, legacy_name in (
            ("train_subjects", "train"),
            ("validation_subjects", "val"),
            ("test_subjects", "test"),
        ):
            fold[f"h_{legacy_name}_ids"] = _numeric_ids(fold[split_name], "H")
            fold[f"p_{legacy_name}_ids"] = _numeric_ids(fold[split_name], "P")
        _validate_fold(fold, fold_index)
        folds.append(fold)

    _validate_outer_test_coverage(folds, set(subjects))
    return folds


def _numeric_ids(subject_folders: list[str], prefix: str) -> list[int]:
    return [
        int(Path(folder).name[1:])
        for folder in subject_folders
        if Path(folder).name.startswith(prefix)
    ]


def _validate_fold(fold: dict[str, list[str]], fold_index: int) -> None:
    train = set(fold["train_subjects"])
    validation = set(fold["validation_subjects"])
    test = set(fold["test_subjects"])
    if train & validation or train & test or validation & test:
        raise RuntimeError(f"Subject leakage detected in fold {fold_index}")


def _validate_outer_test_coverage(
    folds: list[dict[str, list[str]]], all_subjects: set[str]
) -> None:
    test_subjects = [subject for fold in folds for subject in fold["test_subjects"]]
    if len(test_subjects) != len(set(test_subjects)):
        raise RuntimeError("A subject appears in more than one outer test fold")
    if set(test_subjects) != all_subjects:
        raise RuntimeError("Outer test folds do not cover every subject exactly once")
