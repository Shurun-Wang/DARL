# TIM2022 Szhnn: a novel and scalable deep convolution hybrid neural network framework for schizophrenia detection using multichannel eeg

import torch.nn as nn
import torch.nn.functional as F


class SzHNN(nn.Module):
    def __init__(self, channel, cfg):
        super(SzHNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channel, out_channels=cfg['conv1_filter'], kernel_size=cfg['conv1_kernel_size'], stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=cfg['pool1_kernel_size'], stride=cfg['pool1_stride'])
        self.conv2 = nn.Conv1d(in_channels=cfg['conv1_filter'], out_channels=cfg['conv2_filter'], kernel_size=cfg['conv2_kernel_size'], stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=cfg['pool2_kernel_size'], stride=cfg['pool2_stride'])

        self.lstm = nn.LSTM(input_size=cfg['conv2_filter'], hidden_size=cfg['lstm_hidden'], num_layers=cfg['lstm_num_layer'], batch_first=True)

        self.dense = nn.Linear(cfg['lstm_hidden'], cfg['dense'])
        self.dropout = nn.Dropout(cfg['drop_rate'])
        self.output = nn.Linear(cfg['dense'], 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Reshaping for LSTM layer
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dense(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output(x)

        return x