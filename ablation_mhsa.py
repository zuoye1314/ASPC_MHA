import torch
import torch.nn as nn
from torchvision import models


class StandardTransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()

        # 👑 [CLS] 令牌 (用于收集全局特征并最终用于分类)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # ------------------------------------------
        # 模块 A: 多头自注意力机制 (MHSA)
        # ------------------------------------------
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)  # 对应第一个 Add & Norm

        # ------------------------------------------
        # 模块 B: 前馈神经网络 (FFN)
        # ------------------------------------------
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)  # 对应第二个 Add & Norm

    def forward(self, x):
        # x: [B, 512, 7, 7]
        B, C, H, W = x.shape
        # 展平为序列: [B, 49, 512]
        tokens = x.view(B, C, H * W).permute(0, 2, 1)

        # 拼接 [CLS] 令牌 -> [B, 50, 512]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        # ==========================================
        # 核心前向传播流水线 (论文公式的完美复现)
        # ==========================================
        # 1. MHSA
        attn_out, _ = self.mhsa(tokens, tokens, tokens)
        # 2. Add & Norm 1
        tokens = self.norm1(tokens + attn_out)

        # 3. FFN
        ffn_out = self.ffn(tokens)
        # 4. Add & Norm 2
        tokens = self.norm2(tokens + ffn_out)

        # 最终只提取 [CLS] 令牌作为整个图像的全局表征向量: [B, 512]
        cls_output = tokens[:, 0, :]
        return cls_output


class DAN_ResNet50_ASPP_MHSA(nn.Module):
    """消融实验 3: ResNet50 + MHSA (Transformer)"""

    def __init__(self, num_class=7, pretrained=True, weight_path='./ResNet50_msceleb.pth'):
        super(DAN_ResNet50_ASPP_MHSA, self).__init__()

        resnet = models.resnet50(pretrained=False)
        if pretrained:
            print(f">>> [MHSA_Only] 正在加载预训练模型...")
            try:
                checkpoint = torch.load(weight_path, map_location='cpu')
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if
                                    not k.startswith('fc.')}
                resnet.load_state_dict(clean_state_dict, strict=False)
            except Exception as e:
                print(f"!!! 权重加载失败: {e}")


        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # 关键补丁：用 1x1 卷积替代 ASPP 进行纯粹的降维 (2048 -> 512)
        self.dim_reduction = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 引入 Transformer
        self.transformer_block = StandardTransformerBlock(embed_dim=512, num_heads=8, ffn_dim=2048)

        # [CLS] Token 出来就是 512 维
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)  # [B, 2048, 7, 7]
        x_reduced = self.dim_reduction(x)  # [B, 512, 7, 7]

        cls_vector = self.transformer_block(x_reduced)  # [B, 512]

        out = self.fc(cls_vector)
        out = self.bn(out)
        return out, None, None