import torch
import torch.nn as nn
import numpy as np
from riffgan.networks.midinet.utility import *


class Generator(nn.Module):
    def __init__(self, pitch_range):
        super(Generator, self).__init__()
        self.gf_dim = 64
        self.n_channel = 256
        self.pitch_range = pitch_range

        self.ctnet1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.pitch_range+16,
                               out_channels=self.pitch_range,
                               kernel_size=(2, 1),
                               stride=(2, 2)
                               ),
            nn.BatchNorm2d(self.pitch_range),
            nn.LeakyReLU(0.2)
        )

        self.ctnet2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.pitch_range+16,
                               out_channels=self.pitch_range,
                               kernel_size=(2, 1),
                               stride=(2, 2)
                               ),
            nn.BatchNorm2d(self.pitch_range),
            nn.LeakyReLU(0.2)
        )

        self.ctnet3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.pitch_range+16,
                               out_channels=self.pitch_range,
                               kernel_size=(2, 1),
                               stride=(2, 2)
                               ),
            nn.BatchNorm2d(self.pitch_range),
            nn.LeakyReLU(0.2)
        )

        self.ctnet4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.pitch_range+16,
                               out_channels=1,
                               kernel_size=(4, self.pitch_range),
                               stride=(4, 2)
                               ),
            nn.BatchNorm2d(num_features=1),
            nn.LeakyReLU(0.2)
        )

        self.cnet1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=(4, self.pitch_range),
                      stride=(4, 2)
                      ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU()
        )

        self.cnet2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(2, 1),
                      stride=(2, 2)
                      ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU()
        )

        self.cnet3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(2, 1),
                      stride=(2, 2)
                      ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU()
        )

        self.cnet4 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(2, 1),
                      stride=(2, 2)
                      ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU()
        )

        self.linear1 = nn.Linear(100, 1024)
        self.linear2 = nn.Linear(1024, 120)

    def forward(self, noise, seed, batch_size):
        h0_prev = self.cnet1(seed)
        h1_prev = self.cnet2(h0_prev)
        h2_prev = self.cnet3(h1_prev)
        h3_prev = self.cnet4(h2_prev)

        h0 = self.linear1(noise)

        h1 = self.linear2(h0)
        h1 = h1.view(batch_size, 60, 2, 1)
        h1 = conv_prev_concat(h1, h3_prev)

        h2 = self.ctnet1(h1)
        h2 = conv_prev_concat(h2, h2_prev)

        h3 = self.ctnet2(h2)
        h3 = conv_prev_concat(h3, h1_prev)

        h4 = self.ctnet3(h3)
        h4 = conv_prev_concat(h4, h0_prev)

        x = torch.sigmoid(self.ctnet4(h4))

        return x