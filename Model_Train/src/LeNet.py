import time
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

from src.model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
# %matplotlib inline
# 展示高清图
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(0.1307, 0.3081)
                                        ])
train_Data = datasets.MNIST(
 root = '../data', # 下载路径
 train = True, # 是 train 集
 download = True, # 如果该路径没有该数据集，就下载
 transform = trans # 数据集转换参数
)
test_Data = datasets.MNIST(
 root = '../data', # 下载路径
 train = False, # 是 test 集
 download = True, # 如果该路径没有该数据集，就下载
 transform = trans # 数据集转换参数
)

train_data_size = len(train_Data)
test_data_size = len(test_Data)
print(f"训练集长度{train_data_size}")
print(f"测试集长度{test_data_size}")

train_loder = DataLoader(train_Data,batch_size=256,shuffle=True)
test_loder = DataLoader(test_Data,batch_size=256,shuffle=False)

lenet = LeNet()
lenet = lenet.to(device)

# loss_fn
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# optim
learning_rate = 9e-1
optim = torch.optim.SGD(lenet.parameters(), lr=learning_rate)

# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 100

writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print(f"------第{i + 1}轮训练开始------")
    start_time = time.time()
    for data in train_loder:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = lenet(imgs)

        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step = total_train_step + 1
        if total_train_step % 50 == 0:
            writer.add_scalar("train_loss", loss.item(), total_train_step)
            print(f"训练次数:{total_train_step},Loss:{loss.item()}")
            end_time = time.time()
            print(f"第{total_train_step}个50batch花费时间为{end_time - start_time}")
            start_time = time.time()

    total_test_loss = 0
    total_accuracy = 0
    lenet.eval()
    with torch.no_grad():
        for data in test_loder:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = lenet(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            # 本次batch中正确的次数
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print(f"整体的loss：{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy / test_data_size}")
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accurary",total_accuracy/test_data_size,total_test_step)
    total_test_step = total_test_step + 1
    torch.save(lenet.state_dict(),f"../pth_save/lenet_{i}.pth")
    print("model_saved")
writer.close()