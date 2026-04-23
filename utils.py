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


def compute_fold_attributions(model, device, data_loader):
    model.eval()
    cudnn_was_enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    hc_attributions = []
    patient_attributions = []

    try:
        ig = IntegratedGradients(model)
        for samples, targets in data_loader:
            samples, targets = samples.to(device), targets.to(device)
            samples.requires_grad_()

            outputs = model(samples)
            _, predicted = torch.max(outputs, 1)
            correct_idx = (predicted == targets)

            for i in range(len(targets)):
                if correct_idx[i]:
                    single_sample = samples[i:i + 1]
                    target_class = targets[i].item()
                    attr = ig.attribute(single_sample, target=target_class, n_steps=15)
                    channel_importance = torch.mean(torch.abs(attr.squeeze(0)), dim=1).detach().cpu().numpy()

                    if target_class == 0:
                        hc_attributions.append(channel_importance)
                    else:
                        patient_attributions.append(channel_importance)
        return hc_attributions, patient_attributions

    finally:
        torch.backends.cudnn.enabled = cudnn_was_enabled


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
    def __init__(self, root, new_sr, dim=0):
        self.root = root
        self.files = json.load(open(root, "r"))
        self.old_sr = self.files['dataset_info']['sampling_rate']
        self.channel_name = self.files['dataset_info']['ch_names']
        self.mean_value = self.files['dataset_info']['mean']
        self.std_value = self.files['dataset_info']['std']
        self.data = self.files['subject_data']
        self.dim = dim
        self.new_sr = new_sr

    def __len__(self):
        return len(self.data)

    def get_ch_names(self):
        return self.channel_name

    def normalize(self, X):
        mean_value, std_value = np.array(self.mean_value), np.array(self.std_value)
        mu, sigma = np.expand_dims(mean_value, axis=1), np.expand_dims(std_value, axis=1)
        X = (X - mu) / (sigma + 1e-8)
        return X

    def resample_data(self, data):
        if self.old_sr == self.new_sr:
            return data
        else:
            number_of_samples = int(data.shape[-1] * self.new_sr / self.old_sr)
            return signal.resample(data, number_of_samples, axis=-1)

    def __getitem__(self, index):
        trial = self.data[index]
        file_path = trial['file']
        sample = pickle.load(open(file_path, "rb"))
        X = sample["X"]
        if X.ndim < 2:
            X = np.expand_dims(X, axis=0)
        X = self.resample_data(X)
        X = self.normalize(X)
        X = torch.FloatTensor(X)
        Y = sample["Y"]
        return X.float(), Y


def get_model_complexity_info(model, input_shape=(1, 3, 224, 224), device="cpu"):
    params = sum(p.numel() for p in model.parameters()) / 1e6   # M
    dummy_input = torch.randn(input_shape).to(device)
    flops_cnt, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops = flops_cnt / 1e9   # G
    return params, flops

