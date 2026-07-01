import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    """
    通道注意力：
    让模型判断哪些特征通道更重要。
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


class SpatialAttention(nn.Module):
    """
    空间注意力：
    让模型判断图片中哪些位置更重要。
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))

        return x * attention


class CBAM(nn.Module):
    """
    CBAM = 通道注意力 + 空间注意力
    """

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


class ResNet18CBAM(nn.Module):
    """
    最小改动版 ResNet18 + CBAM。

    做法：
    1. 使用 torchvision 的 ResNet18 主体；
    2. 在 layer4 后面加入 CBAM；
    3. 再接 avgpool 和 fc 输出多标签结果。

    这样改动最小，适合你现在的工程。
    """

    def __init__(self, num_labels, pretrained=True):
        super().__init__()

        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                base_model = models.resnet18(weights=weights)
            except Exception:
                base_model = models.resnet18(weights=None)
        else:
            base_model = models.resnet18(weights=None)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.cbam = CBAM(in_channels=512)

        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(512, num_labels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.cbam(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def build_resnet18_cbam(num_labels, pretrained=True):
    return ResNet18CBAM(num_labels=num_labels, pretrained=pretrained)