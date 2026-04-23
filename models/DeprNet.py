# TIM2021 DeprNet: A Deep Convolution Neural Network  Framework for Detecting Depression Using EEG
import torch
import torch.nn as nn

class DeprNet(nn.Module):
    def __init__(self, chans, cfg):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, cfg['conv_kernel_1'], kernel_size=(1, cfg['conv_kernel_1']), stride=(1, cfg['conv_stride_1'])),
            nn.BatchNorm2d(cfg['conv_kernel_1']),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, cfg['pool_kernel_1']), stride=(1, cfg['pool_stride_1'])),

            # Block 2
            nn.Conv2d(cfg['conv_kernel_1'], cfg['conv_kernel_2'], kernel_size=(1, cfg['conv_kernel_2']), stride=(1, cfg['conv_stride_2'])),
            nn.BatchNorm2d(cfg['conv_kernel_2']),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, cfg['pool_kernel_2']), stride=(1, cfg['pool_stride_2'])),

            # Block 3
            nn.Conv2d(cfg['conv_kernel_2'], cfg['conv_kernel_3'], kernel_size=(1, cfg['conv_kernel_3']), stride=(1, cfg['conv_stride_3'])),
            nn.BatchNorm2d(cfg['conv_kernel_3']),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, cfg['pool_kernel_3']), stride=(1, cfg['pool_stride_3'])),

            # Block 4
            nn.Conv2d(cfg['conv_kernel_3'], cfg['conv_kernel_4'], kernel_size=(1, cfg['conv_kernel_4']), stride=(1, cfg['conv_stride_4'])),
            nn.BatchNorm2d(cfg['conv_kernel_4']),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, cfg['pool_kernel_4']), stride=(1, cfg['pool_stride_4'])),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, 2000)
            feat = self.features(dummy)
            flatten_dim = feat.numel()  # = C * H * W

        self.linear_size = flatten_dim
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(self.linear_size, cfg['fc_1'])
        self.fc2 = nn.Linear(cfg['fc_1'], cfg['fc_2'])
        self.fc3 = nn.Linear(cfg['fc_2'], 2)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
        x = self.features(x.unsqueeze(dim=1))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
