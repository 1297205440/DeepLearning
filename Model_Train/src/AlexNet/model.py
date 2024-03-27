# nn
import torch
from torch import nn


class AlexNet_FLOWER(nn.Module):
    def __init__(self,num_classes=1000):
        super(AlexNet_FLOWER,self).__init__()
        self.features = nn.Sequential(
            # input:(3,224,224)
            # 数据集不大，所以深度都/2
            nn.Conv2d(3,48,kernel_size=11,stride=4,padding=2),nn.ReLU(inplace=True),
            # (224-11+2*2)/4+1=55.25
            # output:(48,55,55)
            nn.MaxPool2d(kernel_size=3,stride=2),
            # output:(48,27,27)
            nn.Conv2d(48,128,kernel_size=5,stride=1,padding=2),nn.ReLU(inplace=True),
            # output:(128, 27, 27)
            nn.MaxPool2d(kernel_size=3,stride=2),
            # output:(128, 13, 13)
            nn.Conv2d(128, 192,kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True),
            # output:(192, 13, 13)
            nn.Conv2d(192, 192,kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True),
            # output:(192, 13, 13)
            nn.Conv2d(192, 128,kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True),
            # output:(128, 13, 13)
            nn.MaxPool2d(kernel_size=3,stride=2),
            # output:(128, 6, 6)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 2048), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x