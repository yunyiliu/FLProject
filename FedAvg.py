import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

train_set = torchvision.datasets.MNIST(root="./data",train=True,transform=transforms.ToTensor(),download=True)
train_set_A=Subset(train_set,range(0,1000))
train_set_B=Subset(train_set,range(1000,2000))
train_set_C=Subset(train_set,range(2000,3000))
train_loader_A = dataloader.DataLoader(dataset=train_set_A,batch_size=1000,shuffle=False)
train_loader_B = dataloader.DataLoader(dataset=train_set_B,batch_size=1000,shuffle=False)
train_loader_C = dataloader.DataLoader(dataset=train_set_C,batch_size=1000,shuffle=False)
test_set = torchvision.datasets.MNIST(root="./data",train=False,transform=transforms.ToTensor(),download=True)
test_set=Subset(test_set,range(0,2000))
test_loader = dataloader.DataLoader(dataset=test_set,shuffle=True)

#普通的训练测试过程。首先定义神经网络的类型，
#这里用的是最简单的三层神经网络（也可以说是两层，不算输入层），
#输入层28×28，隐藏层12个神经元，输出层10个神经元
def train_and_test(train_loader, test_loader):
    class NeuralNet(nn.Module):
        def __init__(self, input_num, hidden_num, output_num):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_num, hidden_num)
            self.fc2 = nn.Linear(hidden_num, output_num)
            # 服从正态分布的权重w
            # torch.nn.init.normal_(tensor, mean=0, std=1)
            # 使值服从正态分布N(mean, std)，默认值为0，1
            nn.init.normal_(self.fc1.weight)
            nn.init.normal_(self.fc2.weight)
            #torch.nn.init.constant_(tensor, val)
            #使值为常数val nn.init.constant_(w, 0.3)
            nn.init.constant_(self.fc1.bias, val=0)  # 初始化bias为0
            nn.init.constant_(self.fc2.bias, val=0)
            self.relu = nn.ReLU()  # 使用Relu激活函数

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            y = self.fc2(x)
            return y

    epoches = 20  # 迭代20轮
    lr = 0.01  # 学习率，即步长
    input_num = 784
    hidden_num = 12
    output_num = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_num, hidden_num, output_num)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()  # 损失函数的类型：交叉熵损失函数
    # optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化，也可以用SGD随机梯度下降法
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epoches):
        flag = 0
        for images, labels in train_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            output = model(images)

            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()  # 误差反向传播，计算参数更新值
            optimizer.step()  # 将参数更新值施加到net的parameters上

            # 以下两步可以看每轮损失函数具体的变化情况
            # if (flag + 1) % 10 == 0:
            # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epoches, loss.item()))
            flag += 1

    params = list(model.named_parameters())  # 获取模型参数

    # 测试，评估准确率
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        output = model(images)
        values, predict = torch.max(output, 1)  # 0是每列的最大值，1是每行的最大值
        total += labels.size(0)
        # predict == labels 返回每张图片的布尔类型
        correct += (predict == labels).sum().item()
    print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
    return params

def fedrated_test(train_loader,test_loader,com_para_fc1,com_para_fc2):
    class NeuralNet(nn.Module):
        def __init__(self, input_num, hidden_num, output_num,com_para_fc1,com_para_fc2):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_num, hidden_num)
            self.fc2 = nn.Linear(hidden_num, output_num)
            self.fc1.weight=Parameter(com_para_fc1)
            self.fc2.weight=Parameter(com_para_fc2)
            nn.init.constant_(self.fc1.bias, val=0)
            nn.init.constant_(self.fc2.bias, val=0)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            y = self.fc2(x)
            return y

    epoches = 20
    lr = 0.01
    input_num = 784
    hidden_num = 12
    output_num = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_num, hidden_num, output_num,com_para_fc1,com_para_fc2)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epoches):
        flag = 0
        for images, labels in train_loader:
            # (images, labels) = data
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            output = model(images)

            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (flag + 1) % 10 == 0:
                # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epoches, loss.item()))
            flag += 1
    params = list(model.named_parameters())#get the index by debuging

    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device) #5,8,7...
        output = model(images)#tensor
        values, predict = torch.max(output, 1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
    print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
    return params


def combine_params(para_A, para_B, para_C):
    fc1_wA = para_A[0][1].data
    fc1_wB = para_B[0][1].data
    fc1_wC = para_C[0][1].data
    print("fc1_wA", fc1_wA)
    print("fc2_wB", fc1_wB)
    print("fc1_wC", fc1_wC)
    fc2_wA = para_A[2][1].data
    fc2_wB = para_B[2][1].data
    fc2_wC = para_C[2][1].data

    com_para_fc1 = (fc1_wA + fc1_wB + fc1_wC) / 3
    com_para_fc2 = (fc2_wA + fc2_wB + fc2_wC) / 3
    return com_para_fc1, com_para_fc2

para_A=train_and_test(train_loader_A,test_loader)
para_B=train_and_test(train_loader_B,test_loader)
para_C=train_and_test(train_loader_C,test_loader)
for i in range(10):
    print("The {} round to be federated!!!".format(i+1))
    com_para_fc1,com_para_fc2=combine_params(para_A,para_B,para_C)
    para_A=fedrated_test(train_loader_A,test_loader,com_para_fc1,com_para_fc2)
    print("------------------------------------------------------------------------", para_A)
    para_B=fedrated_test(train_loader_B,test_loader,com_para_fc1,com_para_fc2)
    para_C=fedrated_test(train_loader_C,test_loader,com_para_fc1,com_para_fc2)

# Step 0: get data, split into n clients.
# Step 1: NuralNet: Build a normal training model for Client data training and test
# Step 2: FedAvg algorithm: Use weights get from NuralNet, get the average wight of n client, and return(w1, w2 ...)
# Step 3: FedModel: Build a training model(no matter what model) by using updated weight;
# For every batch training , get FedAvg weight for layer connection by FedAvg algorithm
