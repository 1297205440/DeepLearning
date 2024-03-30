# 首先导入包
import json

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
import matplotlib.pyplot as plt

# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns



class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, trans = transforms.ToTensor()):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """

        self.file_path = file_path
        self.mode = mode
        self.trans = trans
        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path, header=None)  # header=None是去掉表头部分
        # 计算 length
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            # 第一列包含图像文件的名称
            self.train_image = np.asarray(
                self.data_info.iloc[1:self.train_len, 0])  # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            # 第二列是图像的 label
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)

        # 将RGB三通道的图片转换成灰度图片
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')

        # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train' or self.mode == 'train':
            transform = self.trans
        else:
            # valid和test不做数据增强
            transform = self.trans

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]

            return img_as_img, number_label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len


def getData_flowers(batch_size = 32):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 定义data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=nw
    )
    print("using {} images for training, {} images for validation.".format(len(train_dataset),
                                                                           len(val_dataset)))
    return train_loader,len(train_dataset), val_loader,len(val_dataset)

# def getData(batch_size = 8):
#     nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
#     # 定义data loader
#     train_loader = torch.utils.data.DataLoader(
#         dataset=train_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=nw
#     )
#     val_loader = torch.utils.data.DataLoader(
#         dataset=val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=nw
#     )
#     test_loader = torch.utils.data.DataLoader(
#         dataset=test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=nw
#     )
#     return train_loader,len(train_dataset), val_loader,len(val_dataset), test_loader,len(test_dataset)

# 导入标签信息，并做排序
labels_dataframe = pd.read_csv('../../data/classify-leaves/train.csv')
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
# print(n_classes)
# 把label转成对应的数字
class_to_num = dict(zip(leaves_labels, range(n_classes)))
# print(class_to_num)
# 切换键值对，方便预测时候进行查找
num_to_class = {v : k for k, v in class_to_num.items()}
# print(num_to_class)
# leaves_data
train_path = '../../data/classify-leaves/train.csv'
test_path = '../../data/classify-leaves/test.csv'
# csv文件中已经images的路径了，因此这里只到上一级目录
img_path = '../../data/classify-leaves/'

data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
# train_dataset = LeavesData(train_path, img_path, mode='train',valid_ratio=0.2,trans=data_transform["train"])
# val_dataset = LeavesData(train_path, img_path, mode='valid',valid_ratio=0.2,trans=data_transform["val"])
# test_dataset = LeavesData(test_path, img_path, mode='test',valid_ratio=0.2,trans=data_transform["val"])
# print(train_dataset)
# print(val_dataset)
# print(test_dataset)

# flowers_data
data_root = "../../data/"  # get data root path
image_path = os.path.join(data_root, "flower_data")  # flower data set path
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=2)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])