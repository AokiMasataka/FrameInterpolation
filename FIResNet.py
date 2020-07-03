import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.functional import F

import convert


EPOCHS = 16
BATCH_SIZE = 16


class Block(nn.Module):
    def __init__(self, inDeep, outDeep):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inDeep, int(outDeep / 4), kernel_size=1)
        self.conv2 = nn.Conv2d(int(outDeep / 4), int(outDeep / 4), kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(int(outDeep / 4), outDeep, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(int(outDeep / 4))
        self.bn2 = nn.BatchNorm2d(int(outDeep / 4))
        self.bn3 = nn.BatchNorm2d(outDeep)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.cutConv = nn.Conv2d(inDeep, outDeep, kernel_size=1)
        self.cutBn = nn.BatchNorm2d(outDeep)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        cut = self.cutConv(inputs)
        cut = self.cutBn(cut)
        y = self.relu3(x + cut)
        return y


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 96, kernel_size=7, padding=2)
        self.bn1 = nn.BatchNorm2d(96)

        self.block2 = Block(96, 64)
        self.block3 = Block(64, 32)
        self.block4 = Block(32, 16)

        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.conv2(x)
        return self.sigmoid(x)


def train(x, y):
    model.train()

    tensor_x, tensor_y = torch.tensor(x), torch.tensor(y)
    DS = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(DS, batch_size=BATCH_SIZE, shuffle=True)

    for iterate, (batch_x, batch_y) in enumerate(loader):

        x_train = torch.autograd.Variable(batch_x).cuda()
        y_train = torch.autograd.Variable(batch_y).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        lossFunction = nn.MSELoss()

        y = model(x_train)
        loss = lossFunction(y, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del x_train, y_train

        if not (iterate + 1) % 10:
            print(iterate + 1, 'loss : ', loss)


def eval(x, y, epoch):
    model.eval()

    tensor_x, tensor_y = torch.tensor(x), torch.tensor(y)
    DS = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(DS, batch_size=BATCH_SIZE, shuffle=False)

    lossFunction = nn.MSELoss()

    x = x * 255
    Y = np.zeros((1, 3, 240, 240))

    for iterate, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()

        y = model(batch_x)
        print(lossFunction(y, batch_y))

        y = y.cpu().detach().numpy()
        y = y * 255
        Y = np.concatenate((Y, y), axis=0)

    print(x.shape, Y[1:].shape)
    convert.compareVideo('video/generate/Gen' + str(epoch), x, Y[1:])


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet().to(device)
    print('model deploy')

    for epoch in range(EPOCHS):
        for i in range(5):
            print('epoch:' + str(epoch) + ' loading data')
            x, y = np.load('ndarray/trainX' + str(i) + '.npy'), np.load('ndarray/trainY' + str(i) + '.npy')
            print(x.shape, y.shape)
            train(model, x, y)
            del x, y

        X = np.load('ndarray/testX.npy')
        Y = np.load('ndarray/testY.npy')
        eval(model, X[:240], Y[:240], epoch)
        del X, Y

        torch.save(model.state_dict(), 'models/SRNet' + str(epoch))
