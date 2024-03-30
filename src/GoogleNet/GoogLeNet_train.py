import json
import os
import sys
import time
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from tqdm import tqdm
import torch.optim as optm

from src.GoogleNet.model import *
from src.GoogleNet.dataset import getData_flowers
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    writer = SummaryWriter("../../logs_train")

    # 是否要冻住模型的前面一些层
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            model = model
            for param in model.parameters():
                param.requires_grad = False
    # 可以采取已经实现的预训练模型
    def GoogLeNet_model(num_classes, feature_extract = False, use_pretrained=True):
        model_ft = models.googlenet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
        return model_ft

    lr = 0.0002
    epochs = 30
    # weight_decay = 1e-3

    Recording_frequency = 30
    save_path = '../../pth_save/GoogLeNet_FLOWER.pth'
    model_name = "GoogLeNet_Flowers1"
    train_loader,train_num,validate_loader,val_num = getData_flowers(32)

    net = GoogLeNet(num_classes=5)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    best_acc = 0.0
    # 记录训练次数
    total_train_step = 0
    #多少个batch录入一次数据
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        # train_bar = tqdm(train_loader, file=sys.stdout)
        print(f"------第{epoch + 1}轮训练开始------")
        start_time = time.time()
        for data in train_loader:
            optimizer.zero_grad()

            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            logits, aux_logits2, aux_logits1 = net(images)
            loss0 = loss_function(logits, labels)
            loss1 = loss_function(aux_logits1, labels)
            loss2 = loss_function(aux_logits2, labels)
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3

            loss.backward()
            optimizer.step()

            # Recording statistics infomation
            running_loss += loss.item()
            total_train_step = total_train_step + 1
            if total_train_step % Recording_frequency == 0:
                writer.add_scalar(f"{model_name}_trainloss", loss.item(), total_train_step)
                print(f"训练次数:{total_train_step},Loss:{loss.item()}")
                end_time = time.time()
                print(f"第{total_train_step / Recording_frequency}个{Recording_frequency}batch花费时间为{end_time - start_time}")
                start_time = time.time()
            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
            #                                                          epochs,
            #                                                          loss)

        # validate
        net.eval()
        acc = 0.0
        total_accuracy_number = 0
        accurate_number = 0
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                outputs = net(val_images)
                # 本次batch中正确的次数
                accurate_number = (outputs.argmax(1) == val_labels).sum()
                total_accuracy_number = total_accuracy_number + accurate_number

        acc = total_accuracy_number / val_num
        print(f"epoch {epoch} 上整体测试集上的正确率：{acc}")
        writer.add_scalar(f"{model_name}_accurary", acc, epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)
    print('Finished Training')
    writer.close()

if __name__ == '__main__':
    main()