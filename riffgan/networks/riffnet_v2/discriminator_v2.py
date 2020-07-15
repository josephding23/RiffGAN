from riffgan.structure.random_seed import *
import torch
import torch.nn as nn
from riffgan.networks.midinet.utility import *


class Discriminator(nn.Module):
    def __init__(self, pitch_range):
        super(Discriminator, self).__init__()
        self.df_dim = 512
        self.dfc_dim = 1024
        self.pitch_range = pitch_range

        self.cnet_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.df_dim,
                      kernel_size=(5, self.pitch_range),
                      stride=(2, 1),
                      padding=(2, 0)
                      ),
            # nn.ZeroPad2d((0, 0, 1, 1)),
            nn.BatchNorm2d(self.df_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.cnet_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim,
                      out_channels=self.df_dim,
                      kernel_size=(5, 1),
                      stride=(2, 1),
                      padding=(2, 0)
                      ),
            nn.BatchNorm2d(self.df_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.cnet_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim,
                      out_channels=self.df_dim,
                      kernel_size=(3, 1),
                      stride=(2, 1),
                      padding=(1, 0)
                      ),
            nn.BatchNorm2d(self.df_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.cnet_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim,
                      out_channels=self.df_dim,
                      kernel_size=(3, 1),
                      stride=(2, 1),
                      padding=(1, 0)
                      ),
            nn.BatchNorm2d(self.df_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.linear1 = nn.Linear(self.df_dim * 4, 1)

    def forward(self, x, batch_size):

        x = self.cnet_1(x)

        x = self.cnet_2(x)

        x = self.cnet_3(x)

        x = self.cnet_4(x)

        x = x.view(batch_size, -1)

        x = self.linear1(x)

        return x


