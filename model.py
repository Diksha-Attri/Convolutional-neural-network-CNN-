import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=s, padding=1, groups=in_ch, bias=False)
        self.bn1= nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2= nn.BatchNorm2d(out_ch)
        self.act= nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x

class SmallConvNet(nn.Module):
    """
    A compact CNN from scratch:
    - Initial conv
    - 3 depthwise-separable stages with downsampling
    - Global average pooling + classifier
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.stem = ConvBNReLU(3, 32, k=3, s=2, p=1)     # 224 -> 112
        self.stage1 = nn.Sequential(
            DepthwiseSeparable(32, 64, s=1),
            DepthwiseSeparable(64, 64, s=1),
            DepthwiseSeparable(64, 128, s=2),             # 112 -> 56
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparable(128, 128, s=1),
            DepthwiseSeparable(128, 256, s=2),            # 56 -> 28
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparable(256, 256, s=1),
            DepthwiseSeparable(256, 512, s=2),            # 28 -> 14
            DepthwiseSeparable(512, 512, s=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x
