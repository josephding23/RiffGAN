from riffgan.structure.random_seed import *
import torch
import torch.nn as nn
from riffgan.networks.midinet.utility import *


class Discriminator(nn.Module):
    def __init__(self, pitch_range):
        super(Discriminator, self).__init__()
        self.df_dim = 256
        self.dfc_dim = 1024
        self.pitch_range = pitch_range

        self.cnet_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.df_dim,
                      kernel_size=(3, 3),
                      stride=(2, self.pitch_range // 12)
                      ),
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.BatchNorm2d(self.df_dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=self.df_dim,
                      out_channels=self.df_dim,
                      kernel_size=(3, 1),
                      stride=(1, 1)
                      ),
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.BatchNorm2d(self.df_dim),
            nn.ReLU()
        )

        self.cnet_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim,
                      out_channels=self.df_dim,
                      kernel_size=(3, 1),
                      stride=(2, 2)
                      ),
            nn.ZeroPad2d((0, 0, 1, 0)),
            nn.BatchNorm2d(self.df_dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=self.df_dim,
                      out_channels=self.df_dim,
                      kernel_size=(3, 1),
                      stride=(1, 1)
                      ),
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.BatchNorm2d(self.df_dim),
            nn.ReLU()
        )

        self.cnet_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim,
                      out_channels=self.df_dim,
                      kernel_size=(3, 1),
                      stride=(2, 2)
                      ),
            nn.ZeroPad2d((0, 0, 1, 0)),
            nn.BatchNorm2d(self.df_dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=self.df_dim,
                      out_channels=self.df_dim,
                      kernel_size=(3, 1),
                      stride=(1, 1)
                      ),
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.BatchNorm2d(self.df_dim),
            nn.ReLU()
        )

        self.cnet_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim,
                      out_channels=self.df_dim,
                      kernel_size=(3, 1),
                      stride=(2, 3)
                      ),
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.BatchNorm2d(self.df_dim),
            nn.ReLU()
        )

        self.linear1 = nn.Linear(self.df_dim * 4, 1)

    def forward(self, x, batch_size):
        print(x.shape)
        # (-1, 1, 64, 60)
        x = self.cnet_1(x)
        print(x.shape)

        x = self.cnet_2(x)
        print(x.shape)

        x = self.cnet_3(x)
        print(x.shape)

        x = self.cnet_4(x)
        print(x.shape)

        x = x.view(batch_size, -1)
        print(x.shape)

        x = self.linear1(x)
        print(x.shape)

        return torch.sigmoid(x)


def test_discriminator():
    seed_size = 200
    device = torch.device('cuda')

    data = torch.zeros((1, 1, 64, 60)).to(device=device)

    d = Discriminator(pitch_range=60).to(device=device)

    result = d(data, 1)


if __name__ == '__main__':
    test_discriminator()

