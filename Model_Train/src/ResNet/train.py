import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from src.ResNet.dataset import getData
def main():
    writer = SummaryWriter("../../logs_train")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 是否要冻住模型的前面一些层
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            model = model
            for param in model.parameters():
                param.requires_grad = False

    # resnet34模型
    def res34_model(num_classes, feature_extract = False, use_pretrained=True):

        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
        return model_ft

    # 超参数
    learning_rate = 3e-4
    weight_decay = 1e-3
    num_epoch = 60
    model_path = '../../pth_save/pre_res_model.pth'

    train_loader,train_num, val_loader, val_num,test_loader,test_num = getData()

    # Initialize a model, and put it on the device specified.
    model = res34_model(176)
    model = model.to(device)


    # model.device = device

    loss_function = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # The number of training epochs.
    n_epochs = num_epoch
    model_name = "pre_res34_leaves"
    best_acc = 0.0

    # 记录训练次数
    total_train_step = 0
    # 记录测试次数
    total_test_step = 0
    # 多少个batch录入一次数据
    Recording_frequency = 20

    for epoch in range(n_epochs):
        # train
        model.train()
        running_loss = 0.0
        print(f"------第{epoch + 1}轮训练开始------")
        start_time = time.time()
        for data in train_loader:
            optimizer.zero_grad()
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs,labels)
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
            #

        # validate
        model.eval()
        acc = 0.0
        total_accuracy_number = 0
        accurate_number = 0
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                outputs = model(val_images)
                # 本次batch中正确的次数
                accurate_number = (outputs.argmax(1) == val_labels).sum()
                total_accuracy_number = total_accuracy_number + accurate_number

        acc = total_accuracy_number / val_num
        print(f"epoch {epoch} 上整体测试集上的正确率：{acc}")
        writer.add_scalar(f"{model_name}_accurary", acc, epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
    print('Finished Training')
    writer.close()

if __name__ == '__main__':
    main()