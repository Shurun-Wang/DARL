# applied science 2019 Deep convolutional neural network model for automated diagnosis of schizophrenia using eeg signals

import torch.nn as nn
import torch


class OhCNN(nn.Module):
    def __init__(self, channel, cfg):
        super(OhCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channel, out_channels=cfg['conv1_channel'], kernel_size=cfg['conv1_kernel_size'], stride=cfg['conv1_stride'])
        self.maxpool1 = nn.MaxPool1d(kernel_size=cfg['maxpool1_kernel_size'], stride=cfg['maxpool1_stride'])
        self.conv2 = nn.Conv1d(in_channels=cfg['conv1_channel'], out_channels=cfg['conv2_channel'], kernel_size=cfg['conv2_kernel_size'], stride=cfg['conv2_stride'])
        self.maxpool2 = nn.MaxPool1d(kernel_size=cfg['maxpool2_kernel_size'], stride=cfg['maxpool2_stride'])
        self.drop1 = nn.Dropout(cfg['drop_1'])
        self.conv3 = nn.Conv1d(in_channels=cfg['conv2_channel'], out_channels=cfg['conv3_channel'], kernel_size=cfg['conv3_kernel_size'], stride=cfg['conv3_stride'])
        self.avgpool1 = nn.AvgPool1d(kernel_size=cfg['avgpool1_kernel_size'], stride=cfg['avgpool1_stride'])
        self.drop2 = nn.Dropout(cfg['drop_2'])
        self.conv4 = nn.Conv1d(in_channels=cfg['conv3_channel'], out_channels=cfg['conv4_channel'], kernel_size=cfg['conv4_kernel_size'], stride=cfg['conv4_stride'])
        self.avgpool2 = nn.AvgPool1d(kernel_size=cfg['avgpool2_kernel_size'], stride=cfg['avgpool2_stride'])
        self.conv5 = nn.Conv1d(in_channels=cfg['conv4_channel'], out_channels=cfg['conv5_channel'], kernel_size=cfg['conv5_kernel_size'], stride=cfg['conv5_stride'])
        globalInputSize = self.get_size(channel, 2000)[2]
        self.globalpool1 = nn.AvgPool1d(kernel_size=globalInputSize)
        self.classifier = nn.Linear(in_features=cfg['conv5_channel'], out_features=2)
        self.leakyReLU = nn.LeakyReLU()
        self.drop3 = nn.Dropout(cfg['drop_3'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyReLU(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.leakyReLU(x)
        x = self.maxpool2(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.leakyReLU(x)
        x = self.avgpool1(x)
        x = self.drop2(x)
        x = self.conv4(x)
        x = self.leakyReLU(x)
        x = self.avgpool2(x)
        x = self.conv5(x)
        x = self.leakyReLU(x)
        x = self.globalpool1(x)
        x = x.view(x.size()[0], -1)
        x = self.drop3(x)
        x = self.classifier(x)
        return x

    def get_size(self, channel, timepoint):
        data = torch.ones((1, channel, timepoint))
        x = self.conv1(data)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.avgpool1(x)
        x = self.conv4(x)
        x = self.avgpool2(x)
        x = self.conv5(x)

        return x.size()