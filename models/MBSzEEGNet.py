# JBHI 2025 Enhancing EEG-Based Schizophrenia Diagnosis  with Explainable Multi-Branch Deep Learning

import torch.nn as nn
import torch
import math


def padding_same_tuple(kernel):
    kernel -= 1
    if kernel % 2 == 0:
        target = kernel / 2
    else:
        target = (kernel + 1) / 2
    target = int(target)
    if kernel % 2 == 0:
        return (target, target, 0, 0)
    else:
        return (target, target - 1, 0, 0)


class SCC(nn.Module):
    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x ** 2
        x = self.Drop1(x)
        x = self.pool(x)
        return x

    def __init__(self, channels, samples, sfreq, spatial_channel, temporal_channel, ksize, embedded_size, pool_size, pool_stride, drop_rate):
        super().__init__()
        self.channels = channels
        self.samples = samples

        self.conv1 = nn.Conv2d(1, spatial_channel, (channels, 1))
        self.pad2 = nn.ZeroPad2d(padding_same_tuple(int(sfreq * ksize)))
        self.conv2 = nn.Conv2d(spatial_channel, temporal_channel, (1, int(sfreq * ksize)))
        self.bn1 = nn.BatchNorm2d(spatial_channel)
        self.bn2 = nn.BatchNorm2d(temporal_channel)
        self.pool = nn.AvgPool2d((1, math.ceil(min(samples, sfreq * pool_size))), (1, math.ceil(sfreq * pool_stride)))
        self.Drop1 = nn.Dropout(drop_rate)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.channels, 2000)
            feat = self.extract_features(dummy)
            flatten_dim = feat.numel()  # = C * H * W
        self.fc = nn.Linear(flatten_dim, embedded_size)

    def forward(self, x):
        x = self.extract_features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


class EEG(nn.Module):
    def extract_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def __init__(self, channels, sfreq, eeg_ksize, embedded_size, filter_1 = 8, filter_2 = 16, filter_3 = 16,
                 pool_kernel_1=4, pool_stride_1=4, pool_kernel_2=4, pool_stride_2=4, drop_1=0.5, drop_2=0.5):
        super().__init__()
        self.ch = channels
        self.sf = sfreq

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d(padding_same_tuple(math.floor(self.sf * eeg_ksize))),
            nn.Conv2d(1, filter_1, (1, math.floor(self.sf * eeg_ksize)), bias=False),
            nn.BatchNorm2d(filter_1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(filter_1, filter_2, (self.ch, 1), groups=filter_1, bias=False),
            nn.BatchNorm2d(filter_2),
            nn.ELU(),
            nn.AvgPool2d((1, pool_kernel_1), (1, pool_stride_1)),
            nn.Dropout(drop_1)
        )
        self.conv3 = nn.Sequential(
            nn.ZeroPad2d(padding_same_tuple(math.ceil(math.floor(self.sf * eeg_ksize) / 4))),
            nn.Conv2d(filter_2, filter_2, (1, math.ceil(math.floor(self.sf * eeg_ksize) / 4)), groups=filter_2,
                      bias=False),
            nn.Conv2d(filter_2, filter_3, (1, 1), bias=False),
            nn.BatchNorm2d(filter_3),
            nn.ELU(),
            nn.AvgPool2d((1, pool_kernel_2), (1, pool_stride_2)),
            nn.Dropout(drop_2)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.ch, 2000)
            feat = self.extract_features(dummy)
            flatten_dim = feat.numel()  # = C * H * W
        self.fc = nn.Linear(flatten_dim, embedded_size)

    def forward(self, x):
        x = self.extract_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Stack(nn.Module):
    def __init__(self, channels, cfg):
        super().__init__()
        self.stacks_1 = SCC(channels, samples=2000, sfreq=200, spatial_channel=channels,
                            temporal_channel=cfg['scc_1_temp_chan'], ksize=cfg['scc_1_ksize'],
                            embedded_size=cfg['emb_size'], pool_size=cfg['scc_1_pool_size'],
                            pool_stride=cfg['scc_1_pool_stride'], drop_rate=cfg['scc_1_drop_rate'])

        self.stacks_2 = SCC(channels, samples=2000, sfreq=200, spatial_channel=channels,
                            temporal_channel=cfg['scc_2_temp_chan'], ksize=cfg['scc_2_ksize'],
                            embedded_size=cfg['emb_size'], pool_size=cfg['scc_2_pool_size'],
                            pool_stride=cfg['scc_2_pool_stride'], drop_rate=cfg['scc_2_drop_rate'])

        self.eeg_stacks_1 = EEG(channels, sfreq=200, eeg_ksize=cfg['eeg_1_eeg_ksize'], embedded_size=cfg['emb_size'],
                                filter_1=cfg['eeg_1_filter_1'], filter_2 =cfg['eeg_1_filter_2'], filter_3 =cfg['eeg_1_filter_3'],
                                pool_kernel_1=cfg['eeg_1_pool_kernel_1'], pool_stride_1=cfg['eeg_1_pool_stride_1'],
                                pool_kernel_2=cfg['eeg_1_pool_kernel_2'], pool_stride_2=cfg['eeg_1_pool_stride_2'],
                                drop_1=cfg['eeg_1_drop_1'], drop_2=cfg['eeg_1_drop_2'])

        self.eeg_stacks_2 = EEG(channels, sfreq=200, eeg_ksize=cfg['eeg_2_eeg_ksize'], embedded_size=cfg['emb_size'],
                                filter_1=cfg['eeg_2_filter_1'], filter_2 =cfg['eeg_2_filter_2'], filter_3 =cfg['eeg_2_filter_3'],
                                pool_kernel_1=cfg['eeg_2_pool_kernel_1'], pool_stride_1=cfg['eeg_2_pool_stride_1'],
                                pool_kernel_2=cfg['eeg_2_pool_kernel_2'], pool_stride_2=cfg['eeg_2_pool_stride_2'],
                                drop_1=cfg['eeg_2_drop_1'], drop_2=cfg['eeg_2_drop_2'])

    def forward(self, x):
        x_output = []
        out = self.stacks_1(x)
        x_output.append(out)
        out = self.stacks_2(x)
        x_output.append(out)

        out = self.eeg_stacks_1(x)
        x_output.append(out)
        out = self.eeg_stacks_2(x)
        x_output.append(out)

        x_output = torch.stack(x_output, 1)
        x_output = torch.sum(x_output, 1)

        return x_output


class MBSzEEGNet(nn.Module):
    def __init__(self, chans, cfg):
        super(MBSzEEGNet, self).__init__()
        self.module = Stack(chans, cfg)
        self.Drop1 = nn.Dropout(cfg['drop_linear'])
        self.classifier = nn.Linear(cfg['emb_size'], 2)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.module(x)
        x = self.Drop1(x)
        x = self.classifier(x)
        return x

