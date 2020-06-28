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
                      out_channels=64,
                      kernel_size=(5, pitch_range),
                      stride=(4, 2)
                      ),
            nn.ZeroPad2d((0, 0, 1, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )

        self.cnet_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 1),
                      stride=(2, 1)
                      ),
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )

        self.cnet_3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 1),
                      stride=(2, 1)
                      ),
            nn.ZeroPad2d((0, 0, 1, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )

        self.cnet_4 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=256,
                      kernel_size=(3, 1),
                      stride=(1, 1)
                      ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5)
        )

        self.linear1 = nn.Linear(512, 1)

    def forward(self, x, batch_size):
        x = self.cnet_1(x)

        x = self.cnet_2(x)

        x = self.cnet_3(x)

        x = self.cnet_4(x)

        x = x.view(batch_size, -1)

        x = self.linear1(x)

        return x


