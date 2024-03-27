import torch
from torch import nn
# nn
class Cifa10(nn.Module):
    def __init__(self) -> None:
        super(Cifa10,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10),
        )
    def forward(self,x):
        x = self.model(x)
        return x
# nn
class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,6,5,1,2),nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6,16,5,1),nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 120, 5, 1),nn.Tanh(),
            # nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(120*1*1,84),nn.Tanh(),
            nn.Linear(84,10)
        )
    def forward(self,x):
        x = self.net(x)
        return x

# nn
class AlexNet_MNIST(nn.Module):
    def __init__(self) -> None:
        super(AlexNet_MNIST,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(256, 384, 3, 1, 1),nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Flatten(),
            nn.Linear(256*5*5,4096),nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,10),nn.Softmax()
        )
    def forward(self,x):
        x = self.model(x)
        return x

# nn
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

# nn
class AlexNet_cifa(nn.Module):
    def __init__(self) -> None:
        super(AlexNet_cifa,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=48,kernel_size=3,stride=1,padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(256, 384, 3, 1, 1),nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Flatten(),
            nn.Linear(256*3*3,1024),nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,1024),nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,10),nn.Softmax()
        )
    def forward(self,x):
        x = self.model(x)
        return x


# Define BasicConv2d
class BasicConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


# 一个 Inception 块
class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.Conv2d(24, 24, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class GooGleNet(nn.Module):
     def __init__(self):
         super(GooGleNet, self).__init__()
         self.net = nn.Sequential(
         nn.Conv2d(3, 10, kernel_size=5), nn.ReLU(),
         nn.MaxPool2d(kernel_size=2, stride=2),
         Inception(in_channels=10),
         nn.Conv2d(88, 20, kernel_size=5), nn.ReLU(),
         nn.MaxPool2d(kernel_size=2, stride=2),
         Inception(in_channels=20),
             nn.Flatten(),
             nn.Linear(1408, 10)
         )

     def forward(self, x):
        y = self.net(x)
        return y

if __name__ == '__main__':
    cifa10 = Cifa10()
    input = torch.ones((64,3,32,32))
    output = cifa10(input)
    print(output.shape)