import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models


class ASPP_MultiScale(nn.Module):
    def __init__(self, in_channels=2048, out_channels=512):
        super(ASPP_MultiScale, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )



        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(256 * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)

        feat4 = self.branch4(x)
        feat4 = F.interpolate(feat4, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        out = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        out = self.fusion(out)
        return out


class DAN_ResNet50_ASPP_MHSA(nn.Module):
    """消融实验 2: ResNet50 + ASPP"""

    def __init__(self, num_class=7, pretrained=True, weight_path='./ResNet50_msceleb.pth'):
        super(DAN_ResNet50_ASPP_MHSA, self).__init__()

        resnet = models.resnet50(pretrained=False)
        if pretrained:
            print(f">>> [ASPP_Only] 正在加载预训练模型...")
            try:
                checkpoint = torch.load(weight_path, map_location='cpu')
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if
                                    not k.startswith('fc.')}
                resnet.load_state_dict(clean_state_dict, strict=False)
            except Exception as e:
                print(f"!!! 权重加载失败: {e}")

        # === 关键修复：加入这一行，截取 ResNet50 的前段特征提取网络 ===
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # 引入 ASPP，将 2048 维降维并提取多尺度到 512 维
        self.aspp = ASPP_MultiScale(in_channels=2048, out_channels=512)

        # 替代 Transformer 的池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ASPP 输出是 512 维
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)  # [B, 2048, 7, 7]
        feat = self.aspp(x)  # [B, 512, 7, 7]

        pooled = self.avgpool(feat)  # [B, 512, 1, 1]
        pooled = torch.flatten(pooled, 1)  # [B, 512]

        out = self.fc(pooled)
        out = self.bn(out)
        return out, None, None