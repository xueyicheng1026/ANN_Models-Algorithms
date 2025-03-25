import torch
import os
import torchvision.models as models
from torch import nn, optim
import torch.nn
from utils.data_loader import load_cifar10
from utils.train import train_model
from utils.predict import *

# 设置自定义下载路径
os.environ['TORCH_HOME'] = './models'

# 配置
config = {
    "data_dir" : './data',
    "device" : torch.device('cuda' if torch.cuda.is_available() else 'mps'
                if torch.backends.mps.is_available() else 'cpu'),
    "train_limit" : 300,
    "test_limit" : 30,
    "model_type" : "densenet121",
}

data_dir = config['data_dir']
device = config['device']
train_limit = config['train_limit']
test_limit = config['test_limit']
model_type = config['model_type']

# 导入DataSet、DataLoader
train_dataset, test_dataset, train_loader, test_loader = load_cifar10(data_dir=data_dir,
                                                                      train_limit=train_limit,
                                                                      test_limit=test_limit)

# 设置模型类别，建立模型
if model_type == 'densenet121':
    model = models.densenet121(pretrained=True).to(device)
elif model_type == 'densenet169':
    model = models.densenet169(pretrained=True).to(device)
elif model_type == 'resnext50':
    model = models.resnext50_32x4d(pretrained=True).to(device)

# 训练超参数设置
learning_rate = 1e-4
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 开始训练
model, _ =train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    device=device,
    scheduler=scheduler,
    criterion=criterion,
    num_epochs=0,
    use_amp=False,
    early_stop_patience=50,
    save_interval=1,
    load_last_model=False,
    load_best_model=True,
)

predict(model, test_dataset)


