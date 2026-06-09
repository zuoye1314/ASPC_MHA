import os
import sys
import warnings
from tqdm import tqdm
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets

from sklearn.metrics import balanced_accuracy_score

from model import ASPC_MHA
from ablation_baseline import DAN_ResNet50_ASPP_MHSA as Baseline
from ablation_aspp import DAN_ResNet50_ASPP_MHSA as ASPPNet
from ablation_mhsa import DAN_ResNet50_ASPP_MHSA as MHSANet


# --- 【新增：Mixup 处理函数】 ---
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''返回混合后的输入、两组标签及其混合比例'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''混合标签的损失函数计算'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ------------------------------

def warn(*args, **kwargs):
    pass


warnings.warn = warn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='./RAF/', help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--workers', default=4, type=int, help='Number of workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--alpha', type=float, default=0.2, help='mixup interpolation coefficient (default: 0.2)')
    parser.add_argument('--model', type=str, default='full', choices=['full', 'baseline', 'aspp', 'mhsa'],
                        help='Model variant to train: full (ASPC-MHA), baseline (ResNet-50 only), aspp (ResNet-50 + ASPP), mhsa (ResNet-50 + MHSA)')
    return parser.parse_args()


def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f">>> 训练设备: {device}")

    os.makedirs('checkpoints', exist_ok=True)

    model_map = {
        'full': ASPC_MHA,
        'baseline': Baseline,
        'aspp': ASPPNet,
        'mhsa': MHSANet,
    }
    model_cls = model_map[args.model]
    print(f">>> 使用模型: {args.model} ({model_cls.__name__})")
    model = model_cls(num_class=7, pretrained=True, weight_path='./resnet50_msceleb.pth')
    model.to(device)

    # 数据增强保持不变
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(args.raf_path, 'train'), transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                               shuffle=True, pin_memory=True)

    val_dataset = datasets.ImageFolder(os.path.join(args.raf_path, 'valid'), transform=data_transforms_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                             shuffle=False, pin_memory=True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([
        {'params': model.features.parameters(), 'lr': args.lr *0.1},
        {'params': [p for n, p in model.named_parameters() if 'features' not in n], 'lr': args.lr}
    ], weight_decay=1e-4, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            imgs, targets = imgs.to(device), targets.to(device)

            # 【核心修改：应用 Mixup】
            # 在训练阶段对输入数据进行混合
            inputs, targets_a, targets_b, lam = mixup_data(imgs, targets, args.alpha, use_cuda=torch.cuda.is_available())

            out, _, _ = model(inputs)

            # 使用 Mixup 专门的损失计算方式
            loss = mixup_criterion(criterion_cls, out, targets_a, targets_b, lam)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算准确率时，我们依然参考原始标签（由于是混合标签，训练集准确率会比平时低一些，这是正常的）
            _, predicts = torch.max(out, 1)
            correct_num = (lam * torch.eq(predicts, targets_a).sum().float() + (1 - lam) * torch.eq(predicts,
                                                                                                    targets_b).sum().float())
            correct_sum += correct_num

        acc = correct_sum / float(len(train_dataset))
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss / iter_cnt,
                                                                                optimizer.param_groups[0]['lr']))

        # 验证逻辑（验证集不使用 Mixup）
        with torch.no_grad():
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            y_true, y_pred = [], []

            for (imgs, targets) in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                out, _, _ = model(imgs)

                _, predicts = torch.max(out, 1)
                bingo_cnt += torch.eq(predicts, targets).sum().cpu()
                sample_cnt += out.size(0)
                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())

            scheduler.step()
            acc = bingo_cnt.float() / float(sample_cnt)
            best_acc = max(acc.item(), best_acc)

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)

            tqdm.write("[Epoch %d] Validation accuracy:%.4f. bacc:%.4f" % (epoch, acc, balanced_acc))
            tqdm.write(f"best_acc: {best_acc:.4f}")

            if acc > 0.85 and acc.item() == best_acc:
                torch.save({'model_state_dict': model.state_dict()},
                           os.path.join('checkpoints', f"{args.model}_rafdb_acc{acc:.4f}.pth"))

if __name__ == "__main__":
    run_training()