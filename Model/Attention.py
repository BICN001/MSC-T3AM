import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Filter_Gate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=10, num_layers=1):
        super(Filter_Gate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]

        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i+1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i+1), nn.BatchNorm1d(gate_channels[i+1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i+1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))
        self.Avg_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, in_tensor):
        avg_pool = self.Avg_pool(in_tensor)
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)


class Temporal_Gate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=10, dilation_conv_num=1, dilation_val=4):
        super(Temporal_Gate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio,
                                                                kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i, nn.Conv2d(gate_channel//reduction_ratio,
                                                                      gate_channel//reduction_ratio, kernel_size=(1, 25),
                                                                      padding='same'))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel//reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1,
                                                              kernel_size=1))

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


class FT_A(nn.Module):
    def __init__(self, gate_channel):
        super(FT_A, self).__init__()
        self.filter_att = Filter_Gate(gate_channel)
        self.temporal_att = Temporal_Gate(gate_channel)

    def forward(self, in_tensor):
        att = 1 + torch.sigmoid(self.filter_att(in_tensor) + self.temporal_att(in_tensor))
        return att * in_tensor
