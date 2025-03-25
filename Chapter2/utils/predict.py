import random
import matplotlib.pyplot as plt
import numpy as np
import torch

# CIFAR10类别标签
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def show_image(dataset):
    # 随机选择一个样本
    idx = random.randint(0, len(dataset) - 1)
    image, label = dataset[idx]

    # 转换图像用于显示（反归一化）
    image_show = image.numpy().transpose((1, 2, 0))  # C,H,W -> H,W,C
    image_show = image_show * 0.2 + 0.485 # 反归一化
    image_show = np.clip(image_show, 0, 1)  # 裁剪到合理范围

    # 显示图像和真实标签
    plt.imshow(image_show)
    plt.title(f'True Label: {classes[label]}')
    plt.axis('off')
    plt.show()

    return image


def predict(model, dataset):
    # 设置模型为评估模式
    model.eval()

    image = show_image(dataset)

    # 获取设备并将数据移至对应设备
    device = next(model.parameters()).device
    image = image.unsqueeze(0).to(device)  # 增加batch维度

    # 预测
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    print(f'模型预测标签: {classes[predicted.item()]}')

