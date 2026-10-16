"""Shared validation metrics used by every HPO algorithm."""

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def evaluate_balanced_accuracy(model, device, data_loader):
    return _evaluate(model, device, data_loader, compute_auc=False)[0]


def evaluate_balanced_accuracy_and_auc(model, device, data_loader):
    return _evaluate(model, device, data_loader, compute_auc=True)


def _evaluate(model, device, data_loader, compute_auc):
    model.eval()
    probability_batches, target_batches, subjects = [], [], []
    has_subjects = None
    with torch.inference_mode():
        for batch in data_loader:
            samples, targets = batch[:2]
            batch_has_subjects = len(batch) == 3
            if has_subjects is not None and batch_has_subjects != has_subjects:
                raise ValueError('Mixed subject metadata within validation loader')
            has_subjects = batch_has_subjects
            samples = samples.to(device, non_blocking=True)
            probability_batches.append(torch.softmax(model(samples), dim=1))
            target_batches.append(targets.cpu())
            if has_subjects:
                subjects.extend(str(sid) for sid in batch[2])

    if not probability_batches:
        raise ValueError('Empty validation loader')
    # Transfer only N x class_count probabilities, once per evaluation.
    probabilities = torch.cat(probability_batches).cpu().numpy()
    targets = torch.cat(target_batches).numpy()
    if has_subjects:
        names, first, inverse, counts = np.unique(
            subjects, return_index=True, return_inverse=True, return_counts=True)
        subject_targets = targets[first]
        if np.any(targets != subject_targets[inverse]):
            raise RuntimeError('Inconsistent labels for the same subject')
        sums = np.zeros((len(names), probabilities.shape[1]), dtype=probabilities.dtype)
        # Ordered additions retain each subject's original window order.
        np.add.at(sums, inverse, probabilities)
        probabilities = sums / counts[:, None].astype(sums.dtype)
        targets = subject_targets
    score = balanced_accuracy_score(targets, probabilities.argmax(axis=1))
    auc = None
    if compute_auc:
        try:
            auc = float(roc_auc_score(targets, probabilities[:, 1]))
        except ValueError:
            auc = 0.5
    return float(score), auc
