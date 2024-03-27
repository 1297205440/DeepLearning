import time
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from src.model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
# %matplotlib inline
# 展示高清图
from matplotlib_inline import backend_inline

backend_inline.set_matplotlib_formats('svg')

# Hyper-parameters
# 训练轮数
epoch = 30

batch_size = 128

num_classes = 10

learning_rate = 1e-1

torchvision.transforms.Pad(2),

trans_train = torchvision.transforms.Compose([torchvision.transforms.Pad(2),
                                              torchvision.transforms.RandomHorizontalFlip(),
                                              torchvision.transforms.RandomCrop(32),
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trans_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_Data = datasets.CIFAR10(
    root='../data',  # 下载路径
    train=True,  # 是 train 集
    download=True,  # 如果该路径没有该数据集，就下载
    transform=trans_train  # 数据集转换参数
)
test_Data = datasets.CIFAR10(
    root='../data',  # 下载路径
    train=False,  # 是 test 集
    download=True,  # 如果该路径没有该数据集，就下载
    transform=trans_test  # 数据集转换参数
)

train_data_size = len(train_Data)
test_data_size = len(test_Data)
print(f"训练集长度{train_data_size}")
print(f"测试集长度{test_data_size}")

train_loder = DataLoader(train_Data, batch_size=batch_size, shuffle=True)
test_loder = DataLoader(test_Data, batch_size=batch_size, shuffle=False)

googlenet = GooGleNet()
googlenet = googlenet.to(device)

X = torch.rand(size=(1, 3, 32, 32))
for layer in GooGleNet().net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

# loss_fn
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# optim

optim = torch.optim.SGD(googlenet.parameters(), lr=learning_rate)

# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0

writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print(f"------第{i + 1}轮训练开始------")
    start_time = time.time()
    for data in train_loder:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = googlenet(imgs)

        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step = total_train_step + 1
        if total_train_step % 50 == 0:
            writer.add_scalar("googlenettrain_loss", loss.item(), total_train_step)
            print(f"训练次数:{total_train_step},Loss:{loss.item()}")
            end_time = time.time()
            print(f"第{total_train_step}个50batch花费时间为{end_time - start_time}")
            start_time = time.time()

    total_test_loss = 0
    total_accuracy = 0
    googlenet.eval()
    with torch.no_grad():
        for data in test_loder:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = googlenet(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            # 本次batch中正确的次数
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print(f"整体的loss：{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy / test_data_size}")
    writer.add_scalar("googlenettest_loss", total_test_loss, total_test_step)
    writer.add_scalar("googlenettest_accurary", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    torch.save(googlenet.state_dict(), f"googlenetnet_{i}.pth")
    print("model_saved")
writer.close()
