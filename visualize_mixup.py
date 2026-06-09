import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

# ==========================================
# 1. 核心配置区 (改为 RAF-DB 专属配置)
# ==========================================
RAF_PATH = './RAF/'  # 你的 RAF-DB 数据集根目录
RAF_CLASSES = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral'] # RAF-DB 标准类别顺序

def plot_mixup_effect(lam=0.8):
    print(f">>> 正在绘制 RAF-DB 的 Mixup 效果图 (Lambda={lam})...")

    # 设置学术界标准字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 只需要基础的 Resize，方便可视化展示
    vis_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 加载 RAF-DB 的训练集
    train_dataset = datasets.ImageFolder(os.path.join(RAF_PATH, 'train'), transform=vis_transform)

    # 随机挑选两张不同类别的图片
    idx1, idx2 = np.random.randint(0, len(train_dataset), 2)
    img1, label1 = train_dataset[idx1]
    img2, label2 = train_dataset[idx2]

    # 生成 Mixup 图像 (修复了原先硬编码的 0.2，改为通用的 1 - lam)
    mixed_img = lam * img1 + (1 - lam) * img2

    # Tensor 转换回便于 matplotlib 显示的 numpy 格式 (H, W, C)
    def tensor_to_img(tensor):
        return tensor.permute(1, 2, 0).numpy()

    # 画图：设置稍大的画布以适应放大后的字体
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(tensor_to_img(img1))
    axes[0].set_title(f"Image A\n(Label: {RAF_CLASSES[label1]})", fontsize=18, pad=10)
    axes[0].axis('off')

    axes[1].imshow(tensor_to_img(img2))
    axes[1].set_title(f"Image B\n(Label: {RAF_CLASSES[label2]})", fontsize=18, pad=10)
    axes[1].axis('off')

    # 使用 Python 的 round() 避免浮点数精度显示问题，例如 1-0.8=0.19999...
    lam_b = round(1 - lam, 2)
    axes[2].imshow(tensor_to_img(mixed_img))
    axes[2].set_title(f"Mixup Result\n({lam}*A + 0.2*B)", fontsize=18, pad=10)
    axes[2].axis('off')

    plt.tight_layout()

    # 保存为高清 PDF 矢量图格式
    save_path = 'rafdb_mixup_effect.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f">>> 完美！Mixup 效果图已保存为 '{save_path}'")
    plt.show()

if __name__ == "__main__":
    # 画出 Mixup 数据增强效果图
    # 这里 lam=0.8 表示图A占80%，图B占20%
    plot_mixup_effect(lam=0.8)