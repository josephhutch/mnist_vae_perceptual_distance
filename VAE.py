import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class ResBlock(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(2,2)
        )

    def forward(self,x):
        return self.block(x)


class InvResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, final_activation=False):
        super(InvResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        if final_activation:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.block(x)


class CelebVAE(nn.Module):
    def __init__(self):
        super(CelebVAE, self).__init__()

        # in: 1x28x28
        self.encoding_convs = nn.Sequential(ResBlock(3,32),
                                            ResBlock(32,32),
                                            ResBlock(32,64),
                                            ResBlock(64,64),
                                            ResBlock(64,64))

        self.decoding_convs = nn.Sequential(InvResBlock(64,64),
                                            InvResBlock(64,64),
                                            InvResBlock(64,32),
                                            InvResBlock(32, 32),
                                            InvResBlock(32,3, final_activation=True))

        # out: 32x24x24
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc21 = nn.Linear(1024, 512)
        self.fc22 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 64 * 7 * 7)

    def encode(self, x):
        x = self.encoding_convs(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,.3)
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = F.dropout(z, .3)
        z = F.relu(self.fc4(z))
        z = z.view(-1, 64, 7, 7)
        z = self.decoding_convs(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


