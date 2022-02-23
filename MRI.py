import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from ASoftMax import AngleLinear
from Config import Config

import gc

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss = nn.CrossEntropyLoss()

num_classes = config.num_classes


def garbagecollected():
    gc.collect()
    torch.cuda.empty_cache()


class SeqAttention(nn.Module):
    def __init__(self, units=32, attention_activation=True):
        super(SeqAttention, self).__init__()
        self.units = units
        feature_dim = 6272
        self.Wt = nn.Parameter(torch.Tensor(feature_dim, 32))
        self.Wx = nn.Parameter(torch.Tensor(feature_dim, 32))
        self.bh = nn.Parameter(torch.Tensor(32, ))
        self.Wa = nn.Parameter(torch.Tensor(32, 1))
        self.ba = nn.Parameter(torch.Tensor(1, ))

        self.Wt.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.Wx.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.bh.data.uniform_(-1, 1)
        self.Wa.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.ba.data.uniform_(-1, 1)
        self.attention_activation = attention_activation

    def forward(self, inputs):
        alpha = self._emission(inputs)

        if self.attention_activation is not None:
            alpha = F.sigmoid(alpha)
        alpha = torch.exp(alpha - torch.max(alpha, dim=-1, keepdim=True)[0])
        a = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        c_r = torch.matmul(a, inputs)
        return c_r

    def _emission(self, inputs):
        input_shape = inputs.size()
        batch_size, input_len = input_shape[0], input_shape[1]
        q = torch.matmul(inputs, self.Wt).unsqueeze(2)
        k = torch.matmul(inputs, self.Wt).unsqueeze(1)
        beta = F.tanh(q + k + self.bh)
        a = torch.matmul(beta, self.Wa) + self.ba
        alpha = torch.reshape(torch.matmul(beta, self.Wa) + self.ba, (batch_size, input_len, input_len))
        return alpha


class NetRVLAD(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim, **kwargs):
        super(NetRVLAD, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.cluster_size = cluster_size
        self.cluster_weights = nn.Parameter(torch.Tensor(self.feature_size, self.cluster_size))
        self.cluster_biases = nn.Parameter(torch.randn(self.cluster_size, ))
        self.Wn = nn.Parameter(torch.randn(self.feature_size * self.cluster_size, self.output_dim))
        self.cluster_weights.data.normal_(0, 1).renorm(2, 0, 1e-5).mul(1e5)
        self.cluster_biases.data.normal_(0, 1)
        self.Wn.data.normal_(0, 1).renorm(2, 0, 1e-5).mul(1e5)

    def forward(self, x):
        activation = torch.matmul(x, self.cluster_weights)
        activation += self.cluster_biases

        activation = F.softmax(activation)
        activation = torch.reshape(activation, (-1, self.max_samples, self.cluster_size))
        activation = activation.permute(0, 2, 1)
        x = torch.reshape(x, (-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = vlad.permute(0, 2, 1)
        vlad = F.normalize(vlad, p=2, dim=1)
        vlad = torch.reshape(vlad, (-1, self.cluster_size * self.feature_size))
        Nv = F.normalize(vlad, p=2, dim=1)
        vlad = torch.matmul(Nv, self.Wn)
        print(self.cluster_weights.grad)
        # [8, 100]
        return vlad


class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, channel * ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * ratio, channel, bias=False),
            nn.Sigmoid()
        )
        self.fc.apply(self.weight_init)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = F.softmax(y, dim=1)
        return y

    def weight_init(self, p):  # 初始化权重
        print(type(p), type(self))
        if type(p) == nn.Linear:
            nn.init.kaiming_normal_(p.weight, mode='fan_in', nonlinearity='relu')


def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.AvgPool2d):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out + x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.attention = se_block(4)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        att = self.attention(out1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5, att

    def initialize(self):
        '''
        if config.continue_train:
            if os.path.exists(config.model_dict_path):

                self.load_state_dict(torch.load(config.model_dict_path, map_location=device), strict=False)
            else:
                raise Exception("无已保存的文件!")

        else:
        '''
        if not config.continue_train:
            self.load_state_dict(torch.load(config.resnet_path, map_location=device), strict=False)


class MRI(nn.Module):
    def __init__(self):
        super(MRI, self).__init__()
        self.bkbone = ResNet()
        self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(512, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(256, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        ##空洞扩张5
        # self.squeeze5_conv_1= nn.Sequential(nn.Conv2d(128, 128, 1),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze5_dial1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze5_dial2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=3, dilation=3),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze5_dial3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=5, dilation=5),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fusion5_1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True))
        # self.fusion5_2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))

        ##空洞扩张4
        # self.squeeze4_conv_1 = nn.Sequential(nn.Conv2d(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze4_dial1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze4_dial2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=2, dilation=2),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze4_dial3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=3, dilation=3),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fusion4_1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True))
        # self.fusion4_2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))

        ##空洞扩张3
        self.squeeze3_dial1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze3_dial2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=2, dilation=2),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze3_dial3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=3, dilation=3),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fusion3_1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True))
        # self.fusion3_2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        # self.fusion3_3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        ##空洞扩张2
        self.squeeze2_dial1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze2_dial2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=2, dilation=2),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze2_dial3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=3, dilation=3),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fusion2_1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True))
        # self.fusion2_2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        # self.fusion2_3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))

        ##第一次融合
        self.fusion_23 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True))
        self.fusion_34 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True))
        self.fusion_45 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True))

        ##第二次融合
        self.sfusion_12 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.sfusion_23 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True))

        ##第三次融合
        self.tfusion_12 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True))
        ##反馈机制
        self.backfusion1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.backfusion2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.backfusion = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True))

        self.linearr = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linearr2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.angleLinear = AngleLinear(128, num_classes)
        self.avg2 = nn.AdaptiveAvgPool1d(1)
        self.bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, num_classes)
        self.vlad = NetRVLAD(feature_size=128, max_samples=19, cluster_size=32, output_dim=100)
        self.seqAttention = SeqAttention()

    def forward(self, x, targets):
        out2h, out3h, out4h, out5v, att = self.bkbone(x)
        out2h, out3h, out4h, out5v = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(
            out5v)
        att = torch.mean(att, dim=0)
        att = att * 4
        out2h = out2h * att[0]
        out3h = out3h * att[1]
        out4h = out4h * att[2]
        out5v = out5v * att[3]
        out5dia1 = self.squeeze5_dial1(out5v)
        out5dia2 = self.squeeze5_dial2(out5v)
        out5dia3 = self.squeeze5_dial3(out5v)
        out5_ = torch.add(out5dia1, out5dia2)
        out5_ = torch.add(out5_, out5dia3)
        out_5 = self.fusion5_1(out5_)
        out4dia1 = self.squeeze4_dial1(out4h)
        out4dia2 = self.squeeze4_dial2(out4h)
        out4dia3 = self.squeeze4_dial3(out4h)
        out4_ = torch.add(out4dia1, out4dia2)
        out4_ = torch.add(out4_, out4dia3)
        out_4 = self.fusion4_1(out4_)

        out3dia1 = self.squeeze3_dial1(out3h)
        out3dia2 = self.squeeze3_dial2(out3h)
        out3dia3 = self.squeeze3_dial3(out3h)
        out3_ = torch.add(out3dia1, out3dia2)
        out3_ = torch.add(out3_, out3dia3)
        out_3 = self.fusion3_1(out3_)

        out2dia1 = self.squeeze2_dial1(out2h)
        out2dia2 = self.squeeze2_dial2(out2h)
        out2dia3 = self.squeeze2_dial3(out2h)
        out2_ = torch.add(out2dia1, out2dia2)
        out2_ = torch.add(out2_, out2dia3)
        out_2 = self.fusion2_1(out2_)

        # 第一次融合
        out5_up = F.interpolate(out_5, size=out_4.size()[2:], mode='bilinear')
        out_45 = torch.add(out_4, out5_up)
        fusion45 = self.fusion_45(out_45)

        out4_up = F.interpolate(out_4, size=out_3.size()[2:], mode='bilinear')
        out_34 = torch.add(out_3, out4_up)
        fusion34 = self.fusion_34(out_34)

        out3_up = F.interpolate(out_3, size=out_2.size()[2:], mode='bilinear')
        out_23 = torch.add(out_2, out3_up)
        fusion23 = self.fusion_23(out_23)

        # 第二次融合
        fusion45_up = F.interpolate(fusion45, size=fusion34.size()[2:], mode='bilinear')
        sout_12 = torch.add(fusion45_up, fusion34)
        secondfusion_12 = self.sfusion_12(sout_12)

        fusion34_up = F.interpolate(fusion34, size=fusion23.size()[2:], mode='bilinear')
        sout_23 = torch.add(fusion34_up, fusion23)
        secondfusion_23 = self.sfusion_23(sout_23)

        # 第三次融合
        secondfusion_12_up = F.interpolate(secondfusion_12, size=secondfusion_23.size()[2:], mode='bilinear')
        tout12 = torch.add(secondfusion_12_up, secondfusion_23)
        thirdfusion12 = self.tfusion_12(tout12)
        x1 = self.bilinear(thirdfusion12, x.size(0), 42, 42)
        x1 = self.Crop(x1)
        x1 = self.seqAttention(x1)
        x1 = torch.reshape(x1, (x1.size(0), 128, 931))
        x1 = self.avg2(x1)
        x1 = torch.flatten(x1, 1)
        x = self.avg(thirdfusion12)
        x = torch.flatten(x, 1)
        x1 = torch.div(x1, 100)
        x = torch.add(x, x1)
        x = self.angleLinear(x)

        ce_loss = loss(x, targets)
        return x, att, ce_loss

    def initialize(self):
        weight_init(self)

    def bilinear(self, _input, b_size, h_out, w_out):
        gridY = torch.linspace(-1, 1, steps=h_out).view(1, -1, 1, 1).expand(b_size, h_out, w_out, 1).to(device)
        gridX = torch.linspace(-1, 1, steps=w_out).view(1, 1, -1, 1).expand(b_size, h_out, w_out, 1).to(device)
        grid = torch.cat((gridX, gridY), dim=3).to(device)
        return F.grid_sample(_input, grid, mode='bilinear', padding_mode='zeros')

    def Crop(self, _input, _interval=14, _square=42):
        _mx = _square // _interval
        output = []
        for i in range(2, _mx + 1):
            for j in range(0, _mx - i + 1):
                for k in range(0, 3):
                    output.append(_input[:, :, k * _interval: (k + 1) * _interval, j * _interval: (j + i) * _interval])
        for i in range(2, _mx + 1):
            for j in range(0, _mx - i + 1):
                for k in range(0, 3):
                    output.append(_input[:, :, j * _interval: (j + i) * _interval, k * _interval: (k + 1) * _interval])
        output.append(_input)
        _output = []
        for i in range(0, len(output)):
            _output.append(self.bilinear(output[i], output[i].size(0), 7, 7))

        val = torch.tensor([item.cpu().detach().numpy() for item in _output]).cuda()
        val = val.permute(1, 0, 2, 3, 4)
        val = val.resize_(val.size(0), 19, 128 * 7 * 7)
        # print(val.size())
        # [19,8,128,7,7]
        return val

