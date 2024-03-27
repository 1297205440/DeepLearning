import torch
import torchvision
from d2l.torch import d2l
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.optim as optm
class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(BasicConv2d,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,**kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
# num_convs：卷积层的个数
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    layers = []
    layers.append(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding)
    )
    layers.append(nn.ReLU(inplace=True))
    layers.append(
        nn.Conv2d(out_channels, out_channels, kernel_size=1)
    )
    layers.append(nn.ReLU(inplace=True))
    layers.append(
        nn.Conv2d(out_channels, out_channels, kernel_size=1)
    )
    layers.append(nn.ReLU(inplace=True))
    return  nn.Sequential(*layers)

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    #AdaptiveAvgPool2d：全局平均池化
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)，这个数据可以直接输入softmax进行最大似然估计，softmax已经写在train函数里了
    nn.Flatten())

X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())