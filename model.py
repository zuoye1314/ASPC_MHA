from torch import nn
from torch.nn import functional as F
import torch
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
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=4, dilation=4, bias=False),
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
        return self.fusion(out)


class StandardTransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.view(B, C, H * W).permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        # 【修改点 1】：接收 MHSA 返回的注意力权重 attn_weights
        attn_out, attn_weights = self.mhsa(tokens, tokens, tokens)

        tokens = self.norm1(tokens + attn_out)
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        # 【修改点 2】：同时返回 CLS token 和 attn_weights
        return tokens[:, 0, :], attn_weights


class ASPC_MHA(nn.Module):
    def __init__(self, num_class=7, pretrained=True, weight_path='./resnet50_msceleb.pth'):
        super(ASPC_MHA, self).__init__()
        resnet = models.resnet50(pretrained=False)
        if pretrained:
            print(f">>> 正在加载人脸预训练模型: {weight_path}")
            checkpoint = torch.load(weight_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if not k.startswith('fc.')}
            resnet.load_state_dict(clean_state_dict, strict=False)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.aspp_multi_scale = ASPP_MultiScale(in_channels=2048, out_channels=512)
        self.transformer_block = StandardTransformerBlock(embed_dim=512, num_heads=8, ffn_dim=2048)
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)
        feat = self.aspp_multi_scale(x)

        # 【修改点 3】：接收 Transformer 返回的两个参数
        cls_vector, attn_weights = self.transformer_block(feat)

        out = self.bn(self.fc(cls_vector))

        # 【修改点 4】：将实际的 feature map 和 注意力权重 返回给外层的界面
        return out, feat, attn_weights