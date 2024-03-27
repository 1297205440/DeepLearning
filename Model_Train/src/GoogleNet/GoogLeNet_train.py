import json
import os
import sys
import time
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.optim as optm

from src.GoogleNet.model import *

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    writer = SummaryWriter("../../logs_train")
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = "../../data"  # get data root path
    image_path = os.path.join(data_root, "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=2)

    print(json_str)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print(f'Using {nw} dataloader workers every process')

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True,
                              num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = DataLoader(validate_dataset,
                                 batch_size=4, shuffle=False,
                                 num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = GoogLeNet(num_classes=5)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 20
    save_path = '../../pth_save/GoogLeNet_FLOWER.pth'
    model_name = "GoogLeNet_FLOWER"
    best_acc = 0.0
    # train_steps = len(train_loader)

    # 记录训练次数
    total_train_step = 0
    # 记录测试次数
    total_test_step = 0
    #多少个batch录入一次数据
    Recording_frequency = 30
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