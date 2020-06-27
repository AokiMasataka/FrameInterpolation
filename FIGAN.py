import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.functional import F
from torchvision import datasets, models, transforms

import convert

EPOCHS = 32
BATCH_SIZE = 24


class ResBlock(nn.Module):
    def __init__(self, inDeep, outDeep):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inDeep, int(outDeep / 4), kernel_size=1)
        self.conv2 = nn.Conv2d(int(outDeep / 4), int(outDeep / 4), kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(int(outDeep / 4), outDeep, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(int(outDeep / 4))
        self.bn2 = nn.BatchNorm2d(int(outDeep / 4))
        self.bn3 = nn.BatchNorm2d(outDeep)

        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()

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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(6, 96, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(96)

        self.block1 = ResBlock(96, 64)
        self.block2 = ResBlock(64, 64)
        self.block3 = ResBlock(64, 32)
        self.block4 = ResBlock(32, 32)
        self.block5 = ResBlock(32, 16)

        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.conv2(x)
        return self.tanh(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            Flatten(),
            nn.Linear(64 * 20 * 20, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def generator_loss(fakeFrame, frameY, DFake, realLabel):
    vgg = models.vgg16(pretrained=True)
    contentLayers = nn.Sequential(*list(vgg.features)[:31]).cuda().eval()
    for param in contentLayers.parameters():
        param.requires_grad = False

    MSELoss = nn.MSELoss()
    content_loss = MSELoss(contentLayers(fakeFrame), contentLayers(frameY))

    BCELoss = nn.BCELoss()
    adversarial_loss = BCELoss(DFake, realLabel)

    return content_loss + 0.001 * adversarial_loss


def train(x, y):
    tensor_x, tensor_y = torch.tensor(x), torch.tensor(y)
    DS = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(DS, batch_size=BATCH_SIZE, shuffle=True)
    D.train()
    G.train()

    realLabel = torch.ones(BATCH_SIZE, 1).cuda()
    fakeLabel = torch.zeros(BATCH_SIZE, 1).cuda()

    D_loss = 0
    G_loss = 0

    for batch_idx, (frameX, frameY) in enumerate(loader):
        frameX = torch.autograd.Variable(frameX).cuda()
        frameY = torch.autograd.Variable(frameY).cuda()

        fakeFrame = G(frameX)

        D.zero_grad()
        DReal = D(frameY)
        DFake = D(fakeFrame)
        D_real_loss = d_loss(DReal, realLabel)
        D_fake_loss = d_loss(DFake, fakeLabel)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward(retain_graph=True)
        D_optimizer.step()

        G.zero_grad()
        G_loss = generator_loss(fakeFrame, frameY, DFake, realLabel)
        print("G_loss :", G_loss, " D_loss :", D_loss)
        G_loss.backward()
        G_optimizer.step()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    G = Generator()
    D = Discriminator()

    G = G.cuda()
    D = D.cuda()

    GeneratorLR = 0.00025
    DiscriminatorLR = 0.00005

    d_loss = nn.BCELoss()
    print('model deployed')

    for epoch in range(EPOCHS):
        D_optimizer = torch.optim.Adam(D.parameters(), lr=DiscriminatorLR, betas=(0.9, 0.999))
        G_optimizer = torch.optim.Adam(G.parameters(), lr=GeneratorLR, betas=(0.9, 0.999))
        
        train(x, y)

        GeneratorLR *= 0.75
        DiscriminatorLR *= 0.95
