import torch
import torch.nn as nn


def conv_bn_block(cin, cout, dropout):
    """两个 Conv-BN-ReLU + MaxPool + Dropout，和 Keras 版本一一对应。"""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class TrafficSignNet(nn.Module):
    """对应 output/trafficsignnet_v2.keras，输入 64x64x3 RGB，输出 43 类 logits。"""

    def __init__(self, num_classes=43):
        super().__init__()
        self.block1 = conv_bn_block(3, 32, 0.10)
        self.block2 = conv_bn_block(32, 64, 0.15)
        self.block3 = conv_bn_block(64, 128, 0.20)
        self.block4 = conv_bn_block(128, 256, 0.25)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 256)
        self.drop = nn.Dropout(0.40)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)
