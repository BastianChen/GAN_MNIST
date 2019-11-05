from torch import nn
from utils import get_outputpadding
import torch


class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 128, 3, 2, 1),  # n,128,14,14
            nn.LeakyReLU(),
            ConvolutionalLayer(128, 256, 3, 2, 1),  # n,256,7,7
            ConvolutionalLayer(256, 512, 3, 2),  # n,512,3,3
            nn.Conv2d(512, 1, 3),  # n,1,1,1
            nn.Sigmoid()
        )

    def forward(self, data):
        output = self.layer(data)
        return output


class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            ConvTransposeLayer(128, 256, 2, 1, 0, get_outputpadding(1, 2, 2, 1, 0)),  # n,256,2,2
            ConvTransposeLayer(256, 512, 4, 2, 0, get_outputpadding(2, 7, 4, 2, 0)),  # n,512,7,7
            ConvTransposeLayer(512, 128, 4, 2, 1, get_outputpadding(7, 14, 4, 2, 1)),  # n,128,14,14
            nn.ConvTranspose2d(128, 1, 4, 2, 1, get_outputpadding(14, 28, 4, 2, 1)),  # n,1,28,28
            nn.Tanh(),
        )

    def forward(self, data):
        data = data.reshape(data.size(0), -1, 1, 1)
        output = self.layer(data)
        return output


# 用于CGAN
class CDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, 3, 2, 1),  # n,128,14,14
            nn.LeakyReLU(),
            ConvolutionalLayer(128, 256, 3, 2, 1),  # n,256,7,7
            ConvolutionalLayer(256, 512, 3, 2),  # n,512,3,3
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 10),
            nn.Sigmoid()
        )

    def forward(self, data):
        data = self.layer1(data)
        data = data.reshape(data.size(0), -1)
        output = self.layer2(data)
        return output


class CGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            ConvTransposeLayer(138, 256, 2, 1, 0, get_outputpadding(1, 2, 2, 1, 0)),  # n,256,2,2
            ConvTransposeLayer(256, 512, 4, 2, 0, get_outputpadding(2, 7, 4, 2, 0)),  # n,512,7,7
            ConvTransposeLayer(512, 128, 4, 2, 1, get_outputpadding(7, 14, 4, 2, 1)),  # n,128,14,14
            nn.ConvTranspose2d(128, 1, 4, 2, 1, get_outputpadding(14, 28, 4, 2, 1)),  # n,1,28,28
            nn.Tanh(),
        )

    def forward(self, data):
        data = data.reshape(data.size(0), -1, 1, 1)
        output = self.layer(data)
        return output


class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, data):
        return self.layer(data)


class ConvTransposeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, data):
        return self.layer(data)


if __name__ == '__main__':
    d_input = torch.Tensor(1, 1, 28, 28)
    g_input = torch.randn(1, 128, 1, 1)
    g_net = GNet()
    d_net = DNet()
    g_out = g_net(g_input)
    d_out = d_net(d_input)
    print(g_out.shape)
    print(d_out.shape)
    print(d_out)
