import torch.nn as nn
import torch
from torch import device

import models.self_attention as Att

import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化高宽为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化高宽为1

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化---》1*1卷积层降维----》激活函数----》卷积层升维
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化---》1*1卷积层降维----》激活函数----》卷积层升维
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out  # 加和操作
        return self.sigmoid(out)  # sigmoid激活操作


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        # 经过一个卷积层，输入维度是2，输出维度是1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # sigmoid激活操作

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的平均值  b,1,h,w
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的最大值  b,1,h,w
        x = torch.cat([avg_out, max_out], dim=1)  # 在第一维度上拼接，变为 b,2,h,w
        x = self.conv1(x)  # 转换为维度，变为 b,1,h,w
        return self.sigmoid(x)  # sigmoid激活操作


class cbamblock(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(cbamblock, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)  # 将这个权值乘上原输入特征层
        x = x * self.spatialattention(x)  # 将这个权值乘上原输入特征层
        return x

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.bilstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Embedding input sequence
        embedded = self.embedding(x)

        # Forward propagate BiLSTM
        out, _ = self.bilstm(embedded, (h0, c0))

        # Concatenate the hidden states of both directions
        out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)

        # Pass the concatenated hidden states to the fully connected layer
        out = self.fc(out)
        return out

class dsac(nn.Module):

    def __init__(self):
        super(dsac, self).__init__()

        self.convolution1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 16), stride=(1, 1)),#输入通道1 输出通道128 卷积核4*16 步长1*1
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=128)

        )
        self.dropout = nn.Dropout(0.5)
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2))

        self.attention_1=cbamblock(128)
        self.attention_2 = cbamblock(128)

        # self.attention_1 = Att.SelfAttention(101, 4, 0.2)
        # self.attention_2 = Att.SelfAttention(42, 128, 0.7)

        self.convolution2 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), stride=(1, 1)),
        )

        self.output = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

    def _forward_impl(self, seq):
        seq = seq.float()
        conv_seq_1 = self.convolution1(seq)
        pool_seq_1 = self.max_pooling_1(conv_seq_1)
        pool_seq_1 = self.dropout(pool_seq_1)

        att= self.attention_1(pool_seq_1 )
        conv_seq_2 = self.convolution2(att)

        att = self.attention_2(pool_seq_1)
        att_seq_2 = self.convolution2(att)

        return self.output(att_seq_2),self.output(conv_seq_2)
    def forward(self, seq):
        return self._forward_impl(seq)

