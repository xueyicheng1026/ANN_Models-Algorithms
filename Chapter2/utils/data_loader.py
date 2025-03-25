from random import random

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import os
import numpy as np
import random

def load_cifar10(
        data_dir='./data',
        batch_size=64,
        shuffle=True,
        train_limit=None,
        test_limit=None,
        random_seed=42,
):
    random.seed(random_seed)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,         # 亮度调整幅度
            contrast=0.2,           # 对比度调整幅度
            saturation=0.2,         # 饱和度调整幅度
            hue=0.02                # 色相偏移幅度（范围[-0.5,0.5]）
        ),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.485, 0.485), (0.2, 0.2, 0.2))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.485, 0.485), (0.2, 0.2, 0.2))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # 限制训练集数据量
    if train_limit is not None:
        if train_limit <= 0 or train_limit > len(train_dataset):
            raise ValueError(f"train_limit 必须在 (0, {len(train_dataset)}] 范围内")
        # 随机选取子集
        indices = torch.randperm(len(train_dataset))[:train_limit]
        train_dataset = Subset(train_dataset, indices)

    # 限制测试集数据量
    if test_limit is not None:
        if test_limit <= 0 or test_limit > len(test_dataset):
            raise ValueError(f"test_limit 必须在 (0, {len(test_dataset)}] 范围内")
        indices = torch.randperm(len(test_dataset))[:test_limit]
        test_dataset = Subset(test_dataset, indices)

    #  创建dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader

# if __name__ == '__main__':
#     train_loader, test_loader = load_cifar10()



