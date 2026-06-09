# ASPC-MHA: 面部表情识别

**ASPC-MHA**（空洞空间金字塔卷积 + 多头自注意力）是一个基于 ResNet-50 骨干网络的深度学习模型，用于**面部表情识别 (FER)**。它结合了 ASPP 的多尺度特征提取能力和 Transformer 多头自注意力的全局上下文建模能力，同时支持在 **FER2013** 和 **RAF-DB** 两个数据集上训练。

> [English Documentation (英文文档)](README_EN.md)

---

## 文件功能说明

| 文件 | 功能 |
|---|---|
| **`model.py`** | **核心模型**：`ASPC_MHA` 类。ResNet-50 骨干 + ASPP（空洞空间金字塔池化，多尺度特征提取）+ Transformer 模块（多头自注意力 MHSA）。输出分类 logits、特征图以及注意力权重（可用于可视化）。 |
| **`ablation_baseline.py`** | **消融实验 1**：纯 ResNet-50 基线。全局平均池化 + 全连接层，无 ASPP 和 MHSA。 |
| **`ablation_aspp.py`** | **消融实验 2**：ResNet-50 + ASPP（无 Transformer）。ASPP 提取多尺度特征，用全局平均池化替代 Transformer。 |
| **`ablation_mhsa.py`** | **消融实验 3**：ResNet-50 + MHSA（无 ASPP）。用 1×1 卷积降维替代 ASPP，然后送入 Transformer 模块。 |
| **`train_fer2013.py`** | **FER2013 训练脚本**。采用 RandomCrop + TrivialAugmentWide + RandomErasing 数据增强，验证集使用 10-Crop 测试时增强 (TTA)，余弦退火学习率调度器，Label Smoothing 损失函数。 |
| **`train_rafdb.py`** | **RAF-DB 训练脚本**。实现 **Mixup** 数据增强，SGD 优化器配合阶梯式学习率衰减，准确率超过 85% 时自动保存检查点。 |
| **`confusion_matrix.py`** | **混淆矩阵生成**。分别在 FER2013 和 RAF-DB 上推理并绘制归一化混淆矩阵，生成学术级高清 PDF 对比图（seaborn 热力图，Times New Roman 字体）。 |
| **`gui_inference.py`** | **PyQt5 图形界面应用**，实时面部表情识别。支持：摄像头实时监控、单张图片上传、实时概率柱状图、模型注意力热力图叠加显示、FPS 帧率显示。 |
| **`gradcam_compare.py`** | **GradCAM 热力图对比**。对比基线 ResNet-50 和 ASPC-MHA 的注意力区域，并行处理多张图像并输出对比图（PNG + PDF），直观展示模型关注区域的差异。 |
| **`visualize_mixup.py`** | **Mixup 增强效果可视化**。从 RAF-DB 训练集中随机选取两张不同类别的图片，按可配置比例混合，保存为高清 PDF 效果图。 |

---

## 项目结构

```
ASPC_MHA/
├── model.py                  # 核心模型定义
├── ablation_baseline.py      # 消融实验：纯 ResNet-50
├── ablation_aspp.py          # 消融实验：ResNet-50 + ASPP
├── ablation_mhsa.py          # 消融实验：ResNet-50 + MHSA
├── train_fer2013.py          # FER2013 训练
├── train_rafdb.py            # RAF-DB 训练（含 Mixup）
├── confusion_matrix.py       # 混淆矩阵生成
├── gui_inference.py          # PyQt5 实时推理界面
├── gradcam_compare.py        # GradCAM 热力图对比
├── visualize_mixup.py        # Mixup 效果可视化
├── FER2013/                  # FER2013 数据集
├── RAF/                      # RAF-DB 数据集
├── checkpoints/              # RAF-DB 模型保存（自动创建）
├── checkpoints_fer/          # FER2013 模型保存（自动创建）
├── *.pth                     # 预训练权重
├── README_EN.md
└── README_CN.md
```

---

## 预训练权重

百度网盘下载链接：[点击下载](https://pan.baidu.com/s/1EpNZdxIxTMTUNq8P0SawGA?pwd=eha7) 提取码：`eha7`

| 文件 | 说明 |
|---|---|
| `resnet50_msceleb.pth` | MS-Celeb-1M 预训练 ResNet-50 骨干网络 |
| `resnet18_msceleb.pth` | MS-Celeb-1M 预训练 ResNet-18 骨干网络 |
| `mixup_rafdb_acc0.8950.pth` | RAF-DB 训练权重（准确率 89.50%） |

请将下载的 `.pth` 文件放在项目根目录下。

---

## 运行方式

```bash
# 训练 FER2013
python train_fer2013.py --fer_path ./FER2013/ --batch_size 64 --epochs 50

# 训练完整 ASPC-MHA (RAF-DB)
python train_rafdb.py --raf_path ./RAF/ --batch_size 48 --epochs 40 --model full

# 消融实验：纯 ResNet-50 (RAF-DB)
python train_rafdb.py --raf_path ./RAF/ --batch_size 48 --epochs 40 --model baseline

# 消融实验：ResNet-50 + ASPP (RAF-DB)
python train_rafdb.py --raf_path ./RAF/ --batch_size 48 --epochs 40 --model aspp

# 消融实验：ResNet-50 + MHSA (RAF-DB)
python train_rafdb.py --raf_path ./RAF/ --batch_size 48 --epochs 40 --model mhsa

# 启动 GUI 推理界面
python gui_inference.py

# 生成混淆矩阵
python confusion_matrix.py

# 生成 GradCAM 对比图
python gradcam_compare.py

# 生成 Mixup 效果图
python visualize_mixup.py
```

---

## 环境依赖

- Python 3.8+
- PyTorch 2.12+ (CUDA 12.8)
- torchvision
- OpenCV
- PyQt5
- numpy, matplotlib, seaborn, scikit-learn, tqdm
- pytorch-grad-cam（用于 `gradcam_compare.py`）

---

## 引用

如果您在研究中使用本代码，请引用原始的 ASPC-MHA 论文。
