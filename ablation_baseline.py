import torch
import torch.nn as nn
from torchvision import models


class DAN_ResNet50_ASPP_MHSA(nn.Module):

    def __init__(self, num_class=7, pretrained=True, weight_path='./ResNet50_msceleb.pth'):
        super(DAN_ResNet50_ASPP_MHSA, self).__init__()

        resnet = models.resnet50(pretrained=False)
        if pretrained:
            print(f">>> [Baseline] 正在加载预训练模型: {weight_path} ...")
            try:
                checkpoint = torch.load(weight_path, map_location='cpu')
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if
                                    not k.startswith('fc.')}
                resnet.load_state_dict(clean_state_dict, strict=False)
            except Exception as e:
                print(f"!!! 权重加载失败: {e}")
        # 截取 ResNet50 前段
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # 替代 ASPP 和 MHSA 的基础池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ResNet50 最后一层输出是 2048 维
        self.fc = nn.Linear(2048, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)  # [B, 2048, 7, 7]
        x = self.avgpool(x)  # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)  # [B, 2048]

        out = self.fc(x)
        out = self.bn(out)
        return out, None, None