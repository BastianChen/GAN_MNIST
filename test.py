import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch import optim
import torch.nn as nn
import os

class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),
        )#64*14*14
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2))
        )#128*7*7
        self.linear1 = nn.Sequential(
            nn.Linear(128*7*7, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(512, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(128, 1*56*56),
            nn.ReLU(True)
        )  # 1*56*56
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )#128*56*56
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )#64*56*56
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 1, 2, stride=2),
            nn.Tanh()
        )#1*28*28

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

if __name__ == "__main__":
    batch_size = 100
    num_epoch = 10
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    mnist = datasets.MNIST(
        './data', transform=img_transform)
    dataloader = DataLoader(
    mnist, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    d_net = D_Net().to(device)
    g_net = G_Net().to(device)

    if not os.path.exists("./cgan_img"):
        os.mkdir("./cgan_img")
    if not os.path.exists("./cgan_params"):
        os.mkdir("./cgan_params")

    # d_net.state_dict(torch.load(
    #     r'./cgan_params/D_Net.pth'))
    # g_net.state_dict(torch.load(
    #     r'./cgan_params/G_Net.pth'))

    loss_fn = nn.BCELoss()
    d_optimizer = optim.Adam(
        d_net.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(
        g_net.parameters(), lr=0.0002)

    for epoch in range(num_epoch):
        for i,(img, label) in enumerate(dataloader):
            labels_onehot = torch.zeros((batch_size, 10))
            print(labels_onehot)
            labels_onehot[torch.arange(batch_size), label] = 1
            print(labels_onehot)

            img = img.to(device)
            real_label = labels_onehot.to(device)
            fake_label = torch.zeros(batch_size, 10).to(device)

            real_out = d_net(img)
            d_loss_real = loss_fn(real_out, real_label)

            z = torch.randn(batch_size, 128).to(device)
            fake_img = g_net(z)
            fake_out = d_net(fake_img)
            d_loss_fake = loss_fn(fake_out, fake_label)

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 生成器G的训练
            z = torch.randn(batch_size, 118)
            z = torch.cat((z, labels_onehot),dim=1).to(device)

            fake_img = g_net(z)
            output = d_net(fake_img)
            g_loss = loss_fn(output, real_label)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print('Epoch [{}/{}], d_loss: {:.3f}, g_loss: {:.3f} '
                      'D real: {:.3f}, D fake: {:.3f}'
                      .format(epoch, num_epoch, d_loss, g_loss,
                              real_out.data.mean(), fake_out.data.mean()))

                torch.save(g_net.state_dict(), r'./cgan_params/G_Net.pth')
                torch.save(d_net.state_dict(), r'./cgan_params/D_Net.pth')


