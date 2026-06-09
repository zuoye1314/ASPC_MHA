# ASPC-MHA: Facial Expression Recognition

**ASPC-MHA** (Atrous Spatial Pyramid Convolution + Multi-Head Attention) is a deep learning model for **Facial Expression Recognition (FER)** built on a ResNet-50 backbone. It combines multi-scale feature extraction via ASPP with global context modeling via Transformer Multi-Head Self-Attention, and supports training on both **FER2013** and **RAF-DB** datasets.

> [中文文档 (Chinese Version)](README_CN.md)

---

## File Overview

| File | Function |
|---|---|
| **`model.py`** | **Core model**: `ASPC_MHA` — ResNet-50 backbone + ASPP (Atrous Spatial Pyramid Pooling for multi-scale feature extraction) + Transformer block with Multi-Head Self-Attention (MHSA). Returns classification logits, feature maps, and attention weights for visualization. |
| **`ablation_baseline.py`** | **Ablation 1**: Pure ResNet-50 baseline. Global average pooling + FC layer, no ASPP or MHSA. |
| **`ablation_aspp.py`** | **Ablation 2**: ResNet-50 + ASPP only. ASPP extracts multi-scale features but replaces the Transformer with adaptive average pooling. |
| **`ablation_mhsa.py`** | **Ablation 3**: ResNet-50 + MHSA only. Uses a 1×1 convolution for channel reduction instead of ASPP, then applies the Transformer block. |
| **`train_fer2013.py`** | Training script for **FER2013**. Features: RandomCrop + TrivialAugmentWide + RandomErasing augmentation, 10-Crop Test-Time Augmentation (TTA) on validation, cosine annealing LR scheduler, Label Smoothing loss. |
| **`train_rafdb.py`** | Training script for **RAF-DB**. Features: **Mixup** data augmentation, SGD optimizer with step LR decay, automatic checkpoint saving when accuracy exceeds 85%. |
| **`confusion_matrix.py`** | Generates side-by-side **normalized confusion matrices** for both FER2013 and RAF-DB. Outputs a publication-quality PDF with seaborn heatmaps, large fonts, and academic styling (Times New Roman). |
| **`gui_inference.py`** | **PyQt5 GUI application** for real-time facial expression recognition. Supports: live camera feed, single image upload, real-time probability bar chart, attention heatmap overlay (from the model's internal attention weights), FPS display. |
| **`gradcam_compare.py`** | Generates **GradCAM visualizations** comparing baseline ResNet-50 vs. ASPC-MHA. Processes multiple images and produces a side-by-side comparison figure (PNG + PDF) highlighting where each model focuses. |
| **`visualize_mixup.py`** | Visualizes the **Mixup data augmentation** effect. Picks two random RAF-DB images of different classes, blends them at a configurable ratio, and saves the result as a PDF figure. |

---

## Project Structure

```
ASPC_MHA/
├── model.py                  # Core model definition
├── ablation_baseline.py      # Ablation: pure ResNet-50
├── ablation_aspp.py          # Ablation: ResNet-50 + ASPP
├── ablation_mhsa.py          # Ablation: ResNet-50 + MHSA
├── train_fer2013.py          # FER2013 training
├── train_rafdb.py            # RAF-DB training (with Mixup)
├── confusion_matrix.py       # Confusion matrix generation
├── gui_inference.py          # PyQt5 real-time GUI
├── gradcam_compare.py        # GradCAM heatmap comparison
├── visualize_mixup.py        # Mixup effect visualization
├── FER2013/                  # FER2013 dataset
├── RAF/                      # RAF-DB dataset
├── checkpoints/              # RAF-DB model checkpoints (auto-created)
├── checkpoints_fer/          # FER2013 model checkpoints (auto-created)
├── *.pth                     # Pretrained weights
├── README_EN.md
└── README_CN.md
```

---

## Pretrained Weights

Download from Baidu Netdisk: [link placeholder - user will fill in]

| File | Description |
|---|---|
| `resnet50_msceleb.pth` | MS-Celeb-1M pretrained ResNet-50 backbone |
| `resnet18_msceleb.pth` | MS-Celeb-1M pretrained ResNet-18 backbone |
| `mixup_rafdb_acc0.8950.pth` | Trained RAF-DB checkpoint (89.50% accuracy) |

Place the downloaded `.pth` files in the project root directory.

---

## How to Run

```bash
# Train on FER2013
python train_fer2013.py --fer_path ./FER2013/ --batch_size 64 --epochs 50

# Train full ASPC-MHA on RAF-DB
python train_rafdb.py --raf_path ./RAF/ --batch_size 48 --epochs 40 --model full

# Ablation: pure ResNet-50 on RAF-DB
python train_rafdb.py --raf_path ./RAF/ --batch_size 48 --epochs 40 --model baseline

# Ablation: ResNet-50 + ASPP on RAF-DB
python train_rafdb.py --raf_path ./RAF/ --batch_size 48 --epochs 40 --model aspp

# Ablation: ResNet-50 + MHSA on RAF-DB
python train_rafdb.py --raf_path ./RAF/ --batch_size 48 --epochs 40 --model mhsa

# Launch GUI inference
python gui_inference.py

# Generate confusion matrices
python confusion_matrix.py

# Generate GradCAM comparison
python gradcam_compare.py

# Visualize Mixup effect
python visualize_mixup.py
```

---

## Requirements

- Python 3.8+
- PyTorch 2.12+ (CUDA 12.8)
- torchvision
- OpenCV
- PyQt5
- numpy, matplotlib, seaborn, scikit-learn, tqdm
- pytorch-grad-cam (for `gradcam_compare.py`)

---

## Citation

If you use this code in your research, please cite the original ASPC-MHA paper.
