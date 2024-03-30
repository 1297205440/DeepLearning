from torch.utils.tensorboard import SummaryWriter

from src.model import *
import torchvision
from torch import nn
from torch.nn import Flatten, Linear
from torch.utils.data import DataLoader
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练集长度{train_data_size}")
print(f"测试集长度{test_data_size}")

# DataLoader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

cifa10 = Cifa10()
cifa10 = cifa10.to(device)
# loss_fn
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# optim
learning_rate = 1e-2
optim = torch.optim.SGD(cifa10.parameters(), lr=learning_rate)

# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 100

# total_test_loss = 0
# tensorboard
writer = SummaryWriter("../logs_train")
# cifa10.train()
for i in range(epoch):
    print(f"------第{i + 1}轮训练开始------")
    start_time = time.time()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = cifa10(imgs)
        loss = loss_fn(outputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step = total_train_step + 1
        if total_train_step % 50 == 0:
            writer.add_scalar("train_loss", loss.item(), total_train_step)
            print(f"训练次数:{total_train_step},Loss:{loss.item()}")
            end_time = time.time()
            print(end_time - start_time)
            start_time = time.time()

    total_test_loss = 0
    total_accuracy = 0
    cifa10.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = cifa10(imgs)
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
    torch.save(cifa10.state_dict(),f"cifa10_{i}.pth")
    print("model_saved")
    writer.close()
