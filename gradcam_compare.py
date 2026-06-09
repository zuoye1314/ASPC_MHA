import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet50
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 引入你的自定义模型
from model import ASPC_MHA

# ==========================================
# 1. 参数配置区
# ==========================================
# 【修改点1】：将单张图片改为图片路径列表。请填入你想对比的多张图片路径！
# 建议挑选3-4张不同类别的，或者带有遮挡、侧脸的困难样本，对比效果最震撼。
IMAGE_PATHS = [
    "RAF/valid/0/test_0002.jpg",  # 替换为真实的图片路径 1
    "RAF/valid/1/test_0623.jpg",  # 替换为真实的图片路径 2
    "RAF/valid/4/test_0001.jpg",  # 替换为真实的图片路径 3
    # "RAF/valid/4/test_0102.jpg", # 如果需要更多，可以继续取消注释并添加
]

# 模型权重路径
ASPC_WEIGHT_PATH = "checkpoints/mixup_rafdb_acc0.8950.pth"
BASELINE_WEIGHT_PATH = "resnet50_msceleb.pth"

NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASS = None  # None表示自动可视化模型预测得分最高的类别


# ==========================================
# 2. 模型包装器 (解决多返回值问题)
# ==========================================
class ASPCWrapper(nn.Module):
    def __init__(self, model):
        super(ASPCWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        out, _, _ = self.model(x)
        return out


# ==========================================
# 3. 核心逻辑：加载模型与批量生成
# ==========================================
def main():
    print(f"Using device: {DEVICE}")

    # --------------------------------------------------
    # 第一步：只加载一次模型，节省显存和时间
    # --------------------------------------------------
    print(">>> 正在初始化并加载 Baseline (ResNet-50) 模型...")
    model_base = resnet50(pretrained=False)
    model_base.fc = nn.Linear(model_base.fc.in_features, NUM_CLASSES)

    if BASELINE_WEIGHT_PATH and os.path.exists(BASELINE_WEIGHT_PATH):
        checkpoint = torch.load(BASELINE_WEIGHT_PATH, map_location=DEVICE)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if not k.startswith('fc.')}
        model_base.load_state_dict(clean_state_dict, strict=False)

    model_base.to(DEVICE)
    model_base.eval()
    target_layers_base = [model_base.layer4[-1]]
    cam_base = GradCAM(model=model_base, target_layers=target_layers_base)

    print(">>> 正在初始化并加载 ASPC-MHA 模型...")
    model_aspc = ASPC_MHA(num_class=NUM_CLASSES, pretrained=False)
    checkpoint_aspc = torch.load(ASPC_WEIGHT_PATH, map_location=DEVICE)
    if 'state_dict' in checkpoint_aspc:
        state_dict_aspc = checkpoint_aspc['state_dict']
    elif 'model_state_dict' in checkpoint_aspc:
        state_dict_aspc = checkpoint_aspc['model_state_dict']
    else:
        state_dict_aspc = checkpoint_aspc

    clean_state_dict_aspc = {k.replace('module.', ''): v for k, v in state_dict_aspc.items()}
    model_aspc.load_state_dict(clean_state_dict_aspc)

    wrapped_model_aspc = ASPCWrapper(model_aspc)
    wrapped_model_aspc.to(DEVICE)
    wrapped_model_aspc.eval()
    target_layers_aspc = [model_aspc.features[-1]]
    cam_aspc = GradCAM(model=wrapped_model_aspc, target_layers=target_layers_aspc)

    # --------------------------------------------------
    # 第二步：准备绘图画板 (N行 x 3列)
    # --------------------------------------------------
    num_images = len(IMAGE_PATHS)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 根据图片数量动态调整画板高度
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    # 兼容处理：如果只有1张图片，axes是1维数组，我们把它变成2维以便统一索引
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    # --------------------------------------------------
    # 第三步：循环处理每张图片
    # --------------------------------------------------
    for row_idx, img_path in enumerate(IMAGE_PATHS):
        print(f"正在处理第 {row_idx + 1}/{num_images} 张图像: {img_path}")
        if not os.path.exists(img_path):
            print(f"⚠️ 找不到图像跳过: {img_path}")
            continue

        # 1. 图像预处理
        rgb_img_raw = cv2.imread(img_path)
        rgb_img_raw = cv2.cvtColor(rgb_img_raw, cv2.COLOR_BGR2RGB)
        rgb_img_resized = cv2.resize(rgb_img_raw, (224, 224))
        rgb_img_float = np.float32(rgb_img_resized) / 255.0

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(rgb_img_resized).unsqueeze(0).to(DEVICE)
        targets = [ClassifierOutputTarget(TARGET_CLASS)] if TARGET_CLASS is not None else None

        # 2. 生成 Baseline 热力图
        grayscale_cam_base = cam_base(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image_base = show_cam_on_image(rgb_img_float, grayscale_cam_base, use_rgb=True)

        # 3. 生成 ASPC-MHA 热力图
        grayscale_cam_aspc = cam_aspc(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image_aspc = show_cam_on_image(rgb_img_float, grayscale_cam_aspc, use_rgb=True)

        # 4. 绘制到对应的子图中
        ax_orig = axes[row_idx, 0]
        ax_base = axes[row_idx, 1]
        ax_aspc = axes[row_idx, 2]

        ax_orig.imshow(rgb_img_resized)
        ax_orig.axis('off')

        ax_base.imshow(cam_image_base)
        ax_base.axis('off')

        ax_aspc.imshow(cam_image_aspc)
        ax_aspc.axis('off')

        # 【学术排版细节】：只在第一行显示大标题，使整个图矩阵更干净整洁
        if row_idx == 0:
            ax_orig.set_title('Original Image', fontsize=18, pad=15)
            ax_base.set_title('Baseline (ResNet-50)', fontsize=18, pad=15)
            ax_aspc.set_title('ASPC-MHA (Ours)', fontsize=18, pad=15, fontweight='bold')

    # --------------------------------------------------
    # 第四步：保存大图
    # --------------------------------------------------
    plt.subplots_adjust(wspace=0.02, hspace=0.02)  # 极大地缩小图片之间的白边间距
    plt.tight_layout()

    save_path_png = "multi_cam_comparison.png"
    save_path_pdf = "multi_cam_comparison.pdf"

    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', pad_inches=0.1)

    print(f"\n>>> 完美！多图对比热力图已成功保存为 {save_path_png} 和 {save_path_pdf}")
    plt.show()


if __name__ == '__main__':
    main()