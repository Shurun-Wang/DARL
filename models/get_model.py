
import torch.nn as nn
import torch.optim as optim
from models.DeprNet import DeprNet
from models.MBSzEEGNet import MBSzEEGNet
from models.OhCNN import OhCNN
from models.SzHNN import SzHNN
from models.STGEFormer import STGEFormer


def get_model(model_name, ch_num, cfg):
    if model_name == 'DeprNet':
        model = DeprNet(ch_num, cfg)
    elif model_name == 'MBSzEEGNet':
        model = MBSzEEGNet(ch_num, cfg)
    elif model_name == 'OhCNN':
        model = OhCNN(ch_num, cfg)
    elif model_name == 'SzHNN':
        model = SzHNN(ch_num, cfg)
    elif model_name == 'STGEFormer':
        model = STGEFormer(ch_num, cfg)
    else:
        raise NotImplementedError
    return model


def get_opt(par, cfg):
    if cfg["opt"] == 'adamw':
        optimizer = optim.AdamW(par, cfg["lr"])
    elif cfg["opt"] == 'sgd':
        optimizer = optim.SGD(par, cfg["lr"])
    elif cfg["opt"] == 'rms':
        optimizer = optim.RMSprop(par, cfg["lr"])
    elif cfg["opt"] == 'adam':
        optimizer = optim.Adam(par, cfg["lr"])
    else:
        raise NotImplementedError
    return optimizer

