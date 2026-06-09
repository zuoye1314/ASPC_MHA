import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms, datasets
from sklearn.metrics import balanced_accuracy_score
from model import ASPC_MHA


# ================= 新增：定义一个全局函数来替代 lambda =================
# 彻底解决 Windows 下多进程 DataLoader 无法序列化 lambda 函数的报错
def stack_crops(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])


# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fer_path', type=str, default='./FER2013/', help='FER2013数据集路径')
    parser.add_argument('--batch_size', type=int, default=64, help='训练集的Batch Size')
    parser.add_argument('--lr', type=float, default=0.01)  # 配合余弦退火，初始学习率设为0.01
    parser.add_argument('--epochs', type=int, default=50)  # 建议训练 50 个 Epoch
    return parser.parse_args()


def run_training():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoints_fer', exist_ok=True)

    # ==========================================
    # 1. 训练集增强：Resize(256) + 随机裁剪(224) (相当于动态 N-Crop)
    # ==========================================
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),  # 每次加载随机切一刀，提供极致的泛化能力
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),  # 高级自动化增强
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
    ])

    # ==========================================
    # 2. 测试集增强：严格的 10-Crop 策略
    # ==========================================
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop(224),  # 切出 10 张图 (5正+5反)
        # 完美支持 Windows 多进程：调用刚才在最外层定义的全局函数
        transforms.Lambda(stack_crops)
    ])

    train_dataset = datasets.ImageFolder(os.path.join(args.fer_path, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.fer_path, 'test'), transform=val_transform)

    # ⚠️ 核心防爆显存：10-Crop 相当于 batch 扩大了 10 倍，所以验证集的 batch_size 必须缩小！
    val_batch_size = max(1, args.batch_size // 4)  # 如果训练是 64，验证就用 16

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4,
                                             pin_memory=True)

    # 加载模型 (这里会加载你写的 MS-Celeb 预训练权重)
    model = ASPC_MHA(num_class=7, pretrained=True).to(device)

    # 优化器配置：主干网络学习率压低(lr*0.1)，权重衰减加大(5e-4)防止大模型过拟合
    optimizer = torch.optim.SGD([
        {'params': model.features.parameters(), 'lr': args.lr * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'features' not in n], 'lr': args.lr}
    ], momentum=0.9, weight_decay=5e-4)

    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 移除 Class Weights，保留 Label Smoothing 对抗错标噪声
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        # --- 训练阶段 ---
        model.train()
        correct, total = 0, 0
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs, _, _ = model(imgs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            total += targets.size(0)
            correct += torch.eq(preds, targets).sum().item()

            # 显示最新的学习率，可以观察到余弦退火的平滑下降过程
            current_lr = optimizer.param_groups[1]['lr']
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}", lr=f"{current_lr:.5f}")

        scheduler.step()

        # --- 验证/测试阶段 (应用 10-Crop) ---
        model.eval()
        all_preds, all_labels = [], []

        # tqdm 加上验证进度条，因为 10-Crop 会比之前慢
        vbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val  ]")
        with torch.no_grad():
            for imgs, targets in vbar:
                imgs, targets = imgs.to(device), targets.to(device)

                # 此时 imgs 的形状是 [Batch, 10, 3, 224, 224]
                bs, n_crops, c, h, w = imgs.size()

                # 1. 把前两个维度融合，变成 [Batch * 10, 3, 224, 224] 送入网络
                imgs = imgs.view(-1, c, h, w)
                out, _, _ = model(imgs)  # 输出形状: [Batch * 10, 7]

                # 2. 恢复维度，变成 [Batch, 10, 7]
                out = out.view(bs, n_crops, -1)

                # 3. 核心：沿着 10个Crop 的维度求平均，得到最终的概率分布 [Batch, 7]
                out_avg = out.mean(dim=1)

                # 4. 根据平均后的概率选出预测类别
                _, preds = torch.max(out_avg, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = np.mean(all_preds == all_labels)
        bacc = balanced_accuracy_score(all_labels, all_preds)

        print(f"\n>>> 验证结果 - Acc: {acc:.4f}, BAcc: {bacc:.4f}")

        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'checkpoints_fer/fer_best_{acc:.4f}.pth')
            print(f"[*] 新的最佳模型已保存，当前最高 Acc: {best_acc:.4f}\n")


if __name__ == "__main__":
    run_training()