import json
import yaml
import pickle
from scipy import signal
import random
import torch
import numpy as np
from captum.attr import IntegratedGradients
from thop import profile
from torch.utils.data import Dataset
from optimizer.search_space import search_space_DeprNet,search_space_STGEFormer,search_space_MBSzEEGNet,search_space_OhCNN, \
    search_space_SzHNN


def compute_fold_attributions(model, device, data_loader, *, return_metadata=False,
                              n_steps=64, max_steps=256, batch_size=8,
                              internal_batch_size=32, atol=.001, rtol=.05):
    from pathlib import Path
    from torch.utils.data import SequentialSampler
    if min(n_steps, max_steps, batch_size, internal_batch_size) < 1 or max_steps < n_steps:
        raise ValueError('Invalid IG integration counts')
    if atol < 0 or rtol < 0:
        raise ValueError('IG tolerances must be nonnegative')
    rows = None
    if return_metadata:
        if not isinstance(data_loader.sampler, SequentialSampler) or data_loader.drop_last:
            raise ValueError('NPZ row identifiers require a sequential test loader without drop_last')
        rows = data_loader.dataset.data
    was_training = model.training
    model.eval()
    # Same scalar target for patients and controls; avoids class-dependent sign.
    def margin(x):
        logits = model(x)
        return logits[:, 1] - logits[:, 0]
    ig = IntegratedGradients(margin)
    arrays = {k: [] for k in ('absolute', 'signed', 'labels', 'predictions', 'probabilities',
                             'completeness_delta', 'completeness_tolerance', 'ig_steps', 'completeness_pass')}
    cursor = 0
    try:
        for batch in data_loader:
            samples, targets = batch[:2]
            targets_cpu = targets.detach().cpu().numpy().astype(np.int64)
            if not set(targets_cpu.tolist()) <= {0, 1}:
                raise ValueError('Expected binary targets')
            if return_metadata:
                if len(batch) != 3:
                    raise ValueError('NPZ export needs subject IDs from the test loader')
                for offset, (sid, label) in enumerate(zip(batch[2], targets_cpu)):
                    source = rows[cursor + offset]
                    if str(sid) != str(source['subject_name']) or int(source['label']) != int(label):
                        raise ValueError(f'Test row identity/label mismatch at {cursor + offset}')
            for start in range(0, len(samples), batch_size):
                x = samples[start:start + batch_size].to(device).float()
                if x.ndim != 3 or not torch.isfinite(x).all():
                    raise ValueError('Expected finite EEG with shape B x C x T')
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1).cpu().numpy()
                    change = margin(x) - margin(torch.zeros_like(x))
                    tolerance = atol + rtol * change.abs()
                attr, delta = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=n_steps,
                    internal_batch_size=max(internal_batch_size, len(x)), return_convergence_delta=True)
                steps = torch.full((len(x),), n_steps, dtype=torch.long, device=x.device)
                failed = delta.abs() > tolerance
                current_steps = n_steps
                while failed.any() and current_steps < max_steps:
                    current_steps = min(current_steps * 2, max_steps)
                    indices = failed.nonzero().flatten()
                    refined, residual = ig.attribute(x[indices], baselines=torch.zeros_like(x[indices]),
                        n_steps=current_steps, internal_batch_size=max(internal_batch_size, len(indices)),
                        return_convergence_delta=True)
                    attr[indices], delta[indices], steps[indices] = refined, residual, current_steps
                    failed = delta.abs() > tolerance
                if not torch.isfinite(attr).all() or not torch.isfinite(delta).all():
                    raise ValueError('Nonfinite IG values; refusing to export a misleading NPZ')
                arrays['absolute'].append(attr.abs().mean(dim=-1).detach().cpu().numpy())
                arrays['signed'].append(attr.mean(dim=-1).detach().cpu().numpy())
                arrays['labels'].append(targets_cpu[start:start + len(x)])
                arrays['probabilities'].append(probs)
                arrays['predictions'].append(probs.argmax(axis=1))
                arrays['completeness_delta'].append(delta.detach().cpu().numpy())
                arrays['completeness_tolerance'].append(tolerance.cpu().numpy())
                arrays['ig_steps'].append(steps.cpu().numpy())
                arrays['completeness_pass'].append((~failed).cpu().numpy())
            cursor += len(samples)
            print(f'IG windows: {cursor}' + (f'/{len(rows)}' if rows is not None else ''), flush=True)
        if not cursor:
            raise ValueError('Empty test loader')
        result = {key: np.concatenate(value) for key, value in arrays.items()}
        hc, pt = result['labels'] == 0, result['labels'] == 1
        result.update(hc=result['absolute'][hc], pt=result['absolute'][pt],
                      hc_signed=result['signed'][hc], pt_signed=result['signed'][pt],
                      correct=result['labels'] == result['predictions'])
        if not return_metadata:
            return result['hc'], result['pt']
        if cursor != len(rows):
            raise ValueError('Export did not cover all test windows')
        result.update(subject_ids=np.asarray([str(r['subject_name']) for r in rows]),
                      subject_numbers=np.asarray([str(r.get('subject_id', '')) for r in rows]),
                      group_ids=np.asarray([str(r.get('group_id', r['subject_name'])) for r in rows]),
                      files=np.asarray([str(r['file']) for r in rows]),
                      window_ids=np.asarray([str(r['subject_name']) + '/' + Path(r['file']).name for r in rows]),
                      test_row_indices=np.arange(cursor, dtype=np.int64),
                      channel_names=np.asarray(data_loader.dataset.get_ch_names()),
                      target=np.asarray('logit(patient=1)-logit(control=0)'),
                      baseline=np.asarray('zero in normalized input'),
                      aggregation=np.asarray('per-window mean over time; unnormalized; all test windows'))
        for prefix, mask in [('hc', hc), ('pt', pt)]:
            for name in ['subject_ids', 'subject_numbers', 'group_ids', 'files', 'window_ids',
                         'test_row_indices', 'predictions', 'correct']:
                result[f'{prefix}_{name}'] = result[name][mask]
        failures = int((~result['completeness_pass']).sum())
        print(f'IG exported {cursor} windows; completeness failures retained: {failures}', flush=True)
        return result
    finally:
        model.train(was_training)


def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


# ------------------------------ Utils ------------------------------
def seed_everything(seed=6718):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_forward_check(model, device, loader):
    """Single-batch forward pass to early catch shape mismatch."""
    model.eval()
    try:
        samples, _ = next(iter(loader))
        samples = samples.to(device)
        with torch.no_grad():
            _ = model(samples)
        return True
    except Exception:
        return False


def get_dict(model_name):
    if model_name == "DeprNet":
        return search_space_DeprNet
    if model_name == "STGEFormer":
        return search_space_STGEFormer
    if model_name == "MBSzEEGNet":
        return search_space_MBSzEEGNet
    if model_name == "OhCNN":
        return search_space_OhCNN
    if model_name == "SzHNN":
        return search_space_SzHNN



class CustomDataLoader(Dataset):
    def __init__(self, root, new_sr, dim=0, return_subject_id=False):
        self.root = root
        with open(root, 'r', encoding='utf-8') as stream:
            self.files = json.load(stream)
        self.old_sr = self.files['dataset_info']['sampling_rate']
        self.channel_name = self.files['dataset_info']['ch_names']
        self.mean_value = self.files['dataset_info']['mean']
        self.std_value = self.files['dataset_info']['std']
        self.data = self.files['subject_data']
        self.dim = dim
        self.new_sr = new_sr
        self.return_subject_id = return_subject_id
        self._mean = np.asarray(self.mean_value)[:, None]
        self._std = np.asarray(self.std_value)[:, None]
        self._cached_x = None
        self._cached_y = None

    def preload(self):
        """Cache this split's transformed windows once, using its own normalization."""
        if self._cached_x is not None or not self.data:
            return
        x, y = self._load_window(0)
        cache = torch.empty((len(self.data), *x.shape), dtype=torch.float32)
        labels = torch.empty(len(self.data), dtype=torch.long)
        cache[0], labels[0] = x, int(y)
        for index in range(1, len(self.data)):
            x, y = self._load_window(index)
            cache[index], labels[index] = x, int(y)
        self._cached_x, self._cached_y = cache, labels

    @property
    def cache_bytes(self):
        if self._cached_x is None:
            return 0
        return (self._cached_x.numel() * self._cached_x.element_size()
                + self._cached_y.numel() * self._cached_y.element_size())

    def __len__(self):
        return len(self.data)

    def get_ch_names(self):
        return self.channel_name

    def normalize(self, X):
        X = (X - self._mean) / (self._std + 1e-8)
        return X

    def resample_data(self, data):
        if self.old_sr == self.new_sr:
            return data
        else:
            number_of_samples = int(data.shape[-1] * self.new_sr / self.old_sr)
            return signal.resample(data, number_of_samples, axis=-1)

    def _load_window(self, index):
        trial = self.data[index]
        file_path = trial['file']
        with open(file_path, "rb") as stream:
            sample = pickle.load(stream)
        X = sample["X"]
        if X.ndim < 2:
            X = np.expand_dims(X, axis=0)
        X = self.resample_data(X)
        X = self.normalize(X)
        X = torch.FloatTensor(X)
        Y = sample["Y"]
        return X, Y

    def __getitem__(self, index):
        if self._cached_x is None:
            X, Y = self._load_window(index)
        else:
            X, Y = self._cached_x[index], self._cached_y[index]
        if self.return_subject_id:
            return X, Y, self.data[index]["subject_name"]
        return X, Y


def get_model_complexity_info(model, input_shape=(1, 3, 224, 224), device="cpu"):
    params = sum(p.numel() for p in model.parameters()) / 1e6   # M
    dummy_input = torch.randn(input_shape).to(device)
    flops_cnt, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops = flops_cnt / 1e9   # G
    return params, flops
