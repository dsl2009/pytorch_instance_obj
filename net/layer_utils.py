import torch
import torch.nn as nn


class ConBnRelu(nn.Module):
    def __init__(self, inp_dim, out_dim, kenel,  stride=1, with_bn=True):
        super(ConBnRelu, self).__init__()
        pad = (kenel - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (kenel, kenel), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class Fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(Fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn = self.bn(linear) if self.with_bn else linear
        relu = self.relu(bn)
        return relu


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel, stride=1, with_bn=True):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)

class MergeUp(nn.Module):
    def __init__(self):
        super(MergeUp,self).__init__()
    def forward(self, up1, up2):
        return up1 + up2


class CornerPool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(CornerPool, self).__init__()
        self.p1_conv1 = ConBnRelu(dim, 128, kenel=3)
        self.p2_conv1 = ConBnRelu(dim, 128, kenel=3)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = ConBnRelu(dim, dim, kenel=3)
        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2

def gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat