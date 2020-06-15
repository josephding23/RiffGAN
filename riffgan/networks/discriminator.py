import torch.nn as nn
import torch
from torch.nn import init
from riffgan.networks.resnet import ResnetBlock


def init_weight_(net):
    for name, param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.02)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # shape = (64, 84, 1)
        # df_dim = 64

        self.net1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                            out_channels=32,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False),
                                  nn.ReLU(),

                                  nn.Conv2d(in_channels=32,
                                            out_channels=32,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False),
                                  nn.InstanceNorm2d(32, eps=1e-5),
                                  nn.ReLU(),

                                  nn.Conv2d(in_channels=32,
                                            out_channels=64,
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            bias=False),
                                  nn.RReLU(lower=0.1, upper=0.2),
                                  nn.Dropout(0.2)
                                  )
        init_weight_(self.net1)

        self.net2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                            out_channels=128,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False),
                                  nn.InstanceNorm2d(128, eps=1e-5),
                                  nn.ReLU(),

                                  nn.Conv2d(in_channels=128,
                                            out_channels=128,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False),
                                  nn.InstanceNorm2d(256, eps=1e-5),
                                  nn.ReLU(),

                                  nn.Conv2d(in_channels=128,
                                            out_channels=256,
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            bias=False),
                                  nn.InstanceNorm2d(256, eps=1e-5),
                                  nn.RReLU(lower=0.1, upper=0.2)
                                  )
        init_weight_(self.net2)

        self.net3 = nn.Sequential(nn.Conv2d(in_channels=256,
                                            out_channels=1,
                                            kernel_size=7,
                                            stride=1,
                                            padding=3,
                                            bias=False)
                                  )
        init_weight_(self.net3)

    def forward(self, tensor_in):
        x = tensor_in
        # (batch * 1 * 64 * 84)

        x = self.net1(x)
        # ↓
        # (batch * 32 * 64 * 84)
        # ↓
        # (batch * 32 * 32 * 42)
        # ↓
        # (batch * 64 * 32 * 42)

        x = self.net2(x)
        # ↓
        # (batch * 128 * 32 * 42)
        # ↓
        # (batch * 256 * 16 * 21)

        x = self.net3(x)
        # ↓
        # (batch * 1 * 16 * 21)

        return x