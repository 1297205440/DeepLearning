import os

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from src.model import *

img_path = "../imgs/dog3.jpg"
class MyData(Dataset):
    def __init__(self,root_dir,lable_dir):
        self.root_dir  = root_dir;
        self.lable_dir  = lable_dir;
        self.path = os.path.join(self.root_dir,self.lable_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name  = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.lable_dir,img_name)
        img = Image.open(img_item_path)

        label = self.lable_dir
        return img,label

    def __len__(self):
        return len(self.img_path)
root_dir = "../imgs"
dog_lable_dir = "5"
air_lable_dir = "0"
dog_dataset = MyData(root_dir,dog_lable_dir)
air_dataset = MyData(root_dir,air_lable_dir)
# test_dataloader = DataLoader(dog_dataset, batch_size=6)
# for data in test_dataloader:
#     imgs,lables = data
# img.show()

#
# img = Image.open(img_path)
# print(img)

model = Cifa10()
model.load_state_dict(torch.load("cifa10_0.pth")) # 导入网络的参数 0 6 9 05 07 04
# print(model)

for data in air_dataset:
    img,target = data
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                                torchvision.transforms.ToTensor()])
    img = transform(img)
    # print(img.shape)
    img = torch.reshape(img,(1,3,32,32))
    model.eval()
    with torch.no_grad():
        output = model(img)
    # print(output)
    print(output.argmax(1))
# nn


