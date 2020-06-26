import torch.nn as nn
import torch
from torch.nn import init
from riffgan.networks.resnet import ResnetBlock


def init_weight_(net):
    for name, param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.02)


class BarUnit(nn.Module):
    def __init__(self, eta):
        super(BarUnit, self).__init__()

        self.eta = eta
        self.net = nn.Sequential(nn.Conv2d(in_channels=64,
                                           out_channels=128,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           bias=False),
                                 nn.ReLU(),

                                 nn.Conv2d(in_channels=128,
                                           out_channels=64,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           bias=False),
                                 nn.ReLU()
                                 )

    def forward(self, tensor_in):
        x = tensor_in
        # (batch * 64 * 16 * 84)

        out = self.eta * self.net(x) + (1-self.eta) * x
        # ↓
        # (batch * 64 * 16 * 84)

        return out


class Generator(nn.Module):
    def __init__(self, pitch_range):
        super(Generator, self).__init__()


        self.linear1 = nn.Linear(100, 1024)
        self.linear2 = nn.Linear(1037, 256)

        init_weight_(self.linear1)
        init_weight_(self.linear2)

        self.cnet1 = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                   nn.Conv2d(in_channels=1,
                                             out_channels=32,
                                             kernel_size=3,
                                             stride=(4, 3),
                                             padding=0,
                                             bias=False),
                                   nn.BatchNorm2d(32, eps=1e-5),
                                   nn.ReLU()
                                   )

        self.cnet2 = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                   nn.Conv2d(in_channels=32,
                                             out_channels=64,
                                             kernel_size=3,
                                             stride=2,
                                             padding=0,
                                             bias=False),
                                   nn.BatchNorm2d(64, eps=1e-5),
                                   nn.ReLU(),
                                   )

        self.cnet3 = nn.Sequential(nn.Conv2d(in_channels=64,
                                             out_channels=128,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(128, eps=1e-5),
                                   nn.LeakyReLU(0.2)
                                   )

        self.cnet4 = nn.Sequential(nn.Conv2d(in_channels=128,
                                             out_channels=256,
                                             kernel_size=3,
                                             stride=(2, 1),
                                             padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(256, eps=1e-5),
                                   nn.LeakyReLU(0.2)
                                   )

        init_weight_(self.paragraph_cnet2)
        init_weight_(self.paragraph_cnet22)

        self.ctnet4 = nn.Sequential(nn.ConvTranspose2d(in_channels=256,
                                                       out_channels=128,
                                                       kernel_size=3,
                                                       stride=(2, 1),
                                                       padding=1,
                                                       bias=False),
                                    nn.ZeroPad2d((0, 1, 0, 1)),
                                    nn.BatchNorm2d(128, eps=1e-5),
                                    nn.LeakyReLU(0.2)
                                    )

        self.ctnet3 = nn.Sequential(nn.ConvTranspose2d(in_channels=128,
                                                       out_channels=64,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       bias=False),
                                    nn.ZeroPad2d((1, 0, 1, 0)),
                                    nn.BatchNorm2d(64, eps=1e-5),
                                    nn.LeakyReLU(0.2)
                                    )

        self.ctnet2 = nn.Sequential(nn.ConvTranspose2d(in_channels=64,
                                                       out_channels=32,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=0),
                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                    nn.BatchNorm2d(32, eps=1e-5),
                                    nn.LeakyReLU(0.2)
                                    )

        self.ctnet1 = nn.Sequential(nn.ConvTranspose2d(in_channels=32,
                                                       out_channels=1,
                                                       kernel_size=3,
                                                       stride=(4, 3),
                                                       padding=0),
                                    nn.ReflectionPad2d((1, 1, 1, 1)),
                                    nn.LeakyReLU(0.2)
                                    )

        self.linear = nn.Linear(in_features=200, out_features=256*2*pitch_range/12)

        init_weight_(self.paragraph_cnet3)

    def forward(self, noise):
        x = noise
        # (batch * 1 * 64 * 84)

        x = self.paragraph_cnet1(x)
        # ↓
        # (batch * 32 * 64 * 84)
        # ↓
        # (batch * 32 * 64 * 84)
        # ↓
        # (batch * 64 * 64 * 84)

        x1, x2, x3, x4 = x.split([16, 16, 16, 16], dim=2)
        # (batch * 64 * 16 * 84) * 4

        x1 = self.bar_cnet1(x1)
        x2 = self.bar_cnet2(x2)
        x3 = self.bar_cnet3(x3)
        x4 = self.bar_cnet4(x4)
        # ↓
        # (batch * 64 * 16 * 84) * 4

        x = torch.cat([x1, x2, x3, x4], dim=2)

        # (batch * 64 * 64 * 84)

        x = self.paragraph_cnet2(x)

        x = self.paragraph_cnet22(x)
        # ↓
        # (batch * 128 * 32 * 42)
        # ↓
        # (batch * 256 * 16 * 21)

        # x = self.resnet(x)
        # ↓
        # (batch * 256 * 16 * 21)

        x = self.paragraph_ctnet11(x)

        x = self.paragraph_ctnet1(x)

        # ↓
        # (batch * 128 * 32 * 42)
        # ↓
        # (batch * 64 * 64 * 84)

        x = self.paragraph_cnet3(x)
        # ↓
        # (batch * 1 * 64 * 84)

        return x

