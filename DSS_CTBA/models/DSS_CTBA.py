import time

import torch.nn as nn
import torch
import models.Transformer as TF
from torch import device

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

class dss_ctba(nn.Module):

    def __init__(self):
        super(dss_ctba, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.convolution_seq_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 16), stride=(1, 1)),#输入通道1 输出通道128 卷积核4*16 步长1*1
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=128)

        )
        self.convolution_shape_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 16), stride=(1, 1)),
            # 输入通道1 输出通道128 卷积核4*16 步长1*1
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=128)

        )
        # self.bilstm = BiLSTMModel(128, 200, 2, 2)
        self.dropout = nn.Dropout(0.5)
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2))
        self.transformer_shape = TF.Transformer(101, 8, 8, 128, 128, 0.1)
        self.lstm = nn.LSTM(42, 21, 6, bidirectional=True, batch_first=True, dropout=0.2)
        self.convolution2 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), stride=(1, 1)),
        )
        self.dropout = nn.Dropout(0.5)
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))

        self.attention_1 = cbamblock(128)
        self.attention_2 = cbamblock(128)

        # self.attention_1 = Att.SelfAttention(101, 4, 0.2)
        # self.attention_2 = Att.SelfAttention(42, 128, 0.7)



        self.output = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

    def _forward_impl(self, seq, shape):
        seq = seq.float()  # torch.Size([64, 1, 4, 101])
        conv_seq_1 = self.convolution_seq_1(seq) # torch.Size([64, 128, 1, 86])
        pool_seq_1 = self.max_pooling_1(conv_seq_1)  # torch.Size([64, 128, 1, 42])
        pool_seq_1 = self.dropout(pool_seq_1)  # torch.Size([64, 128, 1, 42])
        conv_seq_2 = self.convolution2(pool_seq_1)  # torch.Size([64, 128, 1, 40])
        pool_seq_2 = self.max_pooling_2(conv_seq_2)  # torch.Size([64, 128, 1, 37])
        pool_seq_2 = self.dropout(pool_seq_2)  # torch.Size([64, 128, 1, 37])
        att_seq_1 = self.attention_1(pool_seq_2)
        # att_seq_2 = self.attention_2(att_seq_1)


        shape = shape.float()  # torch.Size([64, 1, 5, 101])
        shape = shape.squeeze(1)  # torch.Size([64, 5, 101])
        encoder_shape_output = self.transformer_shape(shape)  # torch.Size([64, 5, 101])
        encoder_shape_output = encoder_shape_output.unsqueeze(1)  # torch.Size([64, 1, 5, 101])
        conv_shape_1 = self.convolution_shape_1(encoder_shape_output)  # torch.Size([64, 128, 1, 86])
        pool_shape_1 = self.max_pooling_1(conv_shape_1)  # torch.Size([64, 128, 1, 42])
        pool_shape_1 = self.dropout(pool_shape_1)  # torch.Size([64, 128, 1, 42])
        pool_shape_1 = pool_shape_1.squeeze(2)  # torch.Size([64, 128, 42])
        out_shape, _ = self.lstm(pool_shape_1.to(self.device))  # torch.Size([64, 128, 42])
        out_shape1 = out_shape.unsqueeze(2)  # torch.Size([64, 128, 1, 42])
        conv_shape_2 = self.convolution2(out_shape1)  # torch.Size([64, 128, 1, 40])
        pool_shape_2 = self.max_pooling_2(conv_shape_2)  # torch.Size([64, 128, 1, 37])  torch.Size([64, 128, 1, 39])
        pool_shape_2 = self.dropout(pool_shape_2)  # torch.Size([64, 128, 1, 37])  torch.Size([64, 128, 1, 39])
        att_shape_1 = self.attention_1(pool_shape_2)  # torch.Size([64, 128, 1, 37])  torch.Size([64, 128, 1, 39])

        # att_shape_2 = self.attention_2(att_shape_1)

        # bilstm_seq = self.bilstm(pool_seq_1)
        # bilstm_shape = self.bilstm(pool_shape_1)

        # 模型CNN1----ATT1-----ATT2-----CNN2
        # DNASeq
        # att_seq_1 = self.attention_1(pool_seq_1)
        # att_seq_2 = self.attention_2(att_seq_1)  # torch.Size([64, 128, 1, 42])
        # conv_seq_2 = self.convolution2(att_seq_2)  # torch.Size([64, 128, 1, 40])
        # DNAShape
        # att_shape_1 = self.attention_1(pool_shape_1)
        # att_shape_2 = self.attention_2(att_shape_1)
        # conv_shape_2 = self.convolution2(att_shape_2)

        # bilstm_seq = self.bilstm(att_seq_1)
        # bilstm_shape = self.bilstm(att_shape_1)

        # att_seq_1= self.attention_1(pool_seq_1 )
        # # print("----------------------")
        # # print(att.shape)  # torch.Size([64, 128, 1, 42])
        # # print("----------------------")
        # # time.sleep(1000)
        # conv_seq_2 = self.convolution2(att_seq_1)
        # # print("----------------------")
        # # print(conv_seq_2.shape)  # torch.Size([64, 128, 1, 40])
        # # print("----------------------")
        # # time.sleep(1000)

        # return self.output(att_seq_2),self.output(conv_seq_2)
        return self.output(att_seq_1), self.output(att_shape_1)
    def forward(self, seq, shape):
        return self._forward_impl(seq, shape)

