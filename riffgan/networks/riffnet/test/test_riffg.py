from riffgan.structure.random_seed import *
import torch
import torch.nn as nn
import numpy as np
from riffgan.networks.midinet.utility import *


class Generator(nn.Module):
    def __init__(self, pitch_range):
        super(Generator, self).__init__()
        self.gf_dim = 128
        self.n_channel = 128
        self.pitch_range = pitch_range

        self.ctnet4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.n_channel + self.gf_dim,
                               out_channels=self.n_channel,
                               kernel_size=(3, 1),
                               stride=(2, 1),
                               padding=(1, 0)
                               ),
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.BatchNorm2d(self.n_channel),
            nn.LeakyReLU(0.2),
        )

        self.ctnet3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.n_channel + self.gf_dim,
                               out_channels=self.n_channel,
                               kernel_size=(3, 1),
                               stride=(2, 1),
                               padding=(1, 0)
                               ),
            nn.ZeroPad2d((0, 0, 1, 0)),
            nn.BatchNorm2d(self.n_channel),
            nn.LeakyReLU(0.2),
        )

        self.ctnet2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.n_channel + self.gf_dim,
                               out_channels=self.n_channel,
                               kernel_size=(3, 1),
                               stride=(2, 1),
                               padding=(1, 0)
                               ),
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.BatchNorm2d(self.n_channel),
            nn.LeakyReLU(0.2),

        )

        self.ctnet1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.n_channel + self.gf_dim,
                               out_channels=1,
                               kernel_size=(5, self.pitch_range),
                               stride=(2, 1),
                               padding=(2, 0)
                               ),
            nn.ZeroPad2d((0, 0, 1, 0)),
            nn.BatchNorm2d(1),
        )

        self.cnet1 = nn.Sequential(

            nn.Conv2d(in_channels=1,
                      out_channels=self.gf_dim,
                      kernel_size=(5, self.pitch_range),
                      stride=(2, 1),
                      ),
            nn.ReflectionPad2d((0, 0, 1, 1)),
            nn.BatchNorm2d(num_features=self.gf_dim),
            nn.ReLU()
        )

        self.cnet2 = nn.Sequential(
            nn.Conv2d(in_channels=self.gf_dim,
                      out_channels=self.gf_dim,
                      kernel_size=(3, 1),
                      stride=(2, 1),
                      ),
            nn.ReflectionPad2d((0, 0, 0, 1)),
            nn.BatchNorm2d(num_features=self.gf_dim),
            nn.ReLU(),
        )

        self.cnet3 = nn.Sequential(
            nn.Conv2d(in_channels=self.gf_dim,
                      out_channels=self.gf_dim,
                      kernel_size=(3, 1),
                      stride=(2, 1),
                      ),
            nn.ReflectionPad2d((0, 0, 1, 0)),
            nn.BatchNorm2d(num_features=self.gf_dim),
            nn.ReLU()
        )

        self.cnet4 = nn.Sequential(
            nn.Conv2d(in_channels=self.gf_dim,
                      out_channels=self.gf_dim,
                      kernel_size=(3, 1),
                      stride=(2, 1),
                      ),
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.BatchNorm2d(num_features=self.gf_dim),
            nn.ReLU()
        )

        self.linear1 = nn.Linear(200, self.n_channel * 4)

    def forward(self, noise, seed, batch_size):
        print(seed.shape)
        # torch.Size([1, 1, 64, 60])
        h4_prev = self.cnet1(seed)
        print(h4_prev.shape)
        # torch.Size([1, 64, 16, 1])
        h3_prev = self.cnet2(h4_prev)
        print(h3_prev.shape)
        # torch.Size([1, 64, 8, 1])
        h2_prev = self.cnet3(h3_prev)
        print(h2_prev.shape)
        # torch.Size([1, 64, 4, 1])
        h1_prev = self.cnet4(h2_prev)
        print(h1_prev.shape)
        # torch.Size([1, 64, 2, 1])

        h1 = self.linear1(noise)
        print(h1.shape)
        # torch.Size([1, 120])

        h1 = h1.view(batch_size, self.n_channel, 4, 1)
        print(h1.shape, h1_prev.shape)
        h1 = conv_prev_concat(h1, h1_prev)
        print(h1.shape)

        h2 = self.ctnet4(h1)
        print(h2.shape, h2_prev.shape)
        h2 = conv_prev_concat(h2, h2_prev)
        print(h2.shape)

        h3 = self.ctnet3(h2)
        print(h3.shape, h3_prev.shape)
        h3 = conv_prev_concat(h3, h3_prev)
        print(h3.shape)

        h4 = self.ctnet2(h3)
        h4 = conv_prev_concat(h4, h4_prev)
        print(h4.shape)

        x = torch.sigmoid(self.ctnet1(h4))
        print(x.shape)

        return x


def test_generator():
    seed_size = 200
    device = torch.device('cuda')

    noise = torch.randn(1, seed_size, device=device)
    seed = torch.unsqueeze(torch.from_numpy(generate_random_seed(1, 'guitar')), 1).to(
        device=device, dtype=torch.float)

    g = Generator(pitch_range=60).to(device=device)

    fake_data = g(noise, seed, 1)


if __name__ == '__main__':
    test_generator()

