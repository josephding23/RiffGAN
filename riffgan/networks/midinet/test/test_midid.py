from riffgan.structure.random_seed import *
import torch
import torch.nn as nn
from riffgan.networks.midinet.utility import *


class Discriminator(nn.Module):
    def __init__(self, pitch_range):
        super(Discriminator, self).__init__()
        self.df_dim = 64
        self.dfc_dim = 1024
        self.pitch_range = pitch_range

        self.cnet_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=(2, pitch_range),
                      stride=(8, 2)
                      ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        init_weight_(self.cnet_1)

        self.cnet_2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=64,
                      kernel_size=(4, 1),
                      stride=(2, 2)
                      ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        init_weight_(self.cnet_2)

        self.linear1 = nn.Linear(192, 1024)
        self.linear2 = nn.Linear(1024, 1)
        init_weight_(self.linear2)

    def forward(self, x, batch_size):
        print(x.shape)
        # (-1, 1, 64, 60)

        x = self.cnet_1(x)
        print(x.shape)
        # (-1, 16, 8, 1)

        x = self.cnet_2(x)
        print(x.shape)
        # (-1, 64, 3, 1)

        x = x.view(batch_size, -1)
        print(x.shape)
        # (-1, 192)

        x = self.linear1(x)
        print(x.shape)
        # (-1, 1024)

        x = self.linear2(x)
        print(x.shape)
        # (-1, 1)

        return torch.sigmoid(x)


def test_discriminator():
    seed_size = 200
    device = torch.device('cuda')

    data = torch.zeros((1, 1, 64, 60)).to(device=device)

    d = Discriminator(pitch_range=60).to(device=device)

    result = d(data, 1)


if __name__ == '__main__':
    test_discriminator()

