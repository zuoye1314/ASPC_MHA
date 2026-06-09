import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix
from model import ASPC_MHA

# ==========================================
# 1. 全局配置区
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

# --- FER2013 配置 ---
FER_PATH = './FER2013/'
FER_WEIGHT = './checkpoints_fer/fer_best_0.7243.pth'  # 替换为你的 FER2013 最优权重
FER_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- RAF-DB 配置 ---
RAF_PATH = 'RAF/valid'
RAF_WEIGHT = 'checkpoints/mixup_rafdb_acc0.8950.pth'  # 替换为你的 RAF-DB 最优权重
RAF_CLASSES = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']


# ==========================================
# 2. 辅助函数
# ==========================================
# FER2013 10-Crop 必备函数
def stack_crops(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])


# 安全加载权重函数（移除 module. 等多余前缀）
def load_safe_weights(model, weight_path):
    print(f"[*] 正在加载权重: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=DEVICE)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    return model


# ==========================================
# 3. 推理函数 (分别获取矩阵)
# ==========================================
def get_fer2013_cm():
    print(">>> 开始评估 FER2013 (使用 10-Crop)...")
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop(224),
        transforms.Lambda(stack_crops)
    ])
    val_dataset = datasets.ImageFolder(os.path.join(FER_PATH, 'test'), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ASPC_MHA(num_class=7, pretrained=False).to(DEVICE)
    model = load_safe_weights(model, FER_WEIGHT)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            bs, n_crops, c, h, w = imgs.size()
            imgs = imgs.view(-1, c, h, w)
            out, _, _ = model(imgs)
            out = out.view(bs, n_crops, -1)
            out_avg = out.mean(dim=1)
            _, preds = torch.max(out_avg, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


def get_rafdb_cm():
    print(">>> 开始评估 RAF-DB (使用常规 CenterCrop/Resize)...")
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(root=RAF_PATH, transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ASPC_MHA(num_class=7, pretrained=False).to(DEVICE)
    model = load_safe_weights(model, RAF_WEIGHT)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs, _, _ = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


# ==========================================
# 4. 主函数：绘制并排图
# ==========================================
def main():
    # 获取两个数据集的归一化混淆矩阵
    cm_fer = get_fer2013_cm()
    cm_raf = get_rafdb_cm()

    print(">>> 推理完成，正在绘制组合热力图...")
    # 设置全局学术字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 【大跨度修改 1】：画布再次扩大，给超大字体留足空间
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))

    # --- 绘制 (a) FER2013 ---
    # 【大跨度修改 2】：矩阵内部数字调大到 26
    sns.heatmap(cm_fer, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=FER_CLASSES, yticklabels=FER_CLASSES,
                annot_kws={"size": 26}, square=True, ax=axes[0], cbar_kws={"shrink": 0.82})

    # 【大跨度修改 3】：标题字体调大到 38，增加 pad 让标题不挤
    axes[0].set_title('(a) Confusion Matrix on FER2013', fontsize=38, pad=30)

    # 【大跨度修改 4】：坐标轴标签字体调大到 30，增加 labelpad 拉开距离
    axes[0].set_ylabel('True Label', fontsize=30, labelpad=15)
    axes[0].set_xlabel('Predicted Label', fontsize=30, labelpad=15)

    # 【大跨度修改 5】：类别刻度字体调大到 26
    axes[0].tick_params(axis='x', rotation=45, labelsize=26)
    axes[0].tick_params(axis='y', rotation=0, labelsize=26)

    # 【大跨度修改 6】：右侧 Colorbar 的字体调大到 22
    cbar_fer = axes[0].collections[0].colorbar
    cbar_fer.ax.tick_params(labelsize=22)

    # --- 绘制 (b) RAF-DB ---
    # 同理进行超大号字体应用
    sns.heatmap(cm_raf, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=RAF_CLASSES, yticklabels=RAF_CLASSES,
                annot_kws={"size": 26}, square=True, ax=axes[1], cbar_kws={"shrink": 0.82})

    axes[1].set_title('(b) Confusion Matrix on RAF-DB', fontsize=38, pad=30)
    axes[1].set_ylabel('True Label', fontsize=30, labelpad=15)
    axes[1].set_xlabel('Predicted Label', fontsize=30, labelpad=15)
    axes[1].tick_params(axis='x', rotation=45, labelsize=26)
    axes[1].tick_params(axis='y', rotation=0, labelsize=26)

    cbar_raf = axes[1].collections[0].colorbar
    cbar_raf.ax.tick_params(labelsize=22)

    # 自动调整排版，防止超大字体被图边缘吃掉
    plt.tight_layout(pad=3.0)

    # --- 保存为高清矢量图 PDF ---
    save_path_pdf = "combined_confusion_matrices.pdf"
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight', dpi=300)
    print(f"[*] 完美！双混淆矩阵图已保存为: {save_path_pdf}")

    plt.show()


if __name__ == "__main__":
    main()