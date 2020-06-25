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
    def __init__(self, eta):
        super(Generator, self).__init__()

        self.eta = eta

        self.linear1 = nn.Linear(100, 1024)
        self.linear2 = nn.Linear(1037, 256)

        init_weight_(self.linear1)
        init_weight_(self.linear2)

        self.paragraph_cnet1 = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                             nn.Conv2d(in_channels=1,
                                                       out_channels=32,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=0,
                                                       bias=False),
                                             nn.BatchNorm2d(32, eps=1e-5),
                                             nn.ReLU(),

                                             nn.ReflectionPad2d((1, 1, 1, 1)),
                                             nn.Conv2d(in_channels=32,
                                                       out_channels=64,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=0,
                                                       bias=False),
                                             nn.BatchNorm2d(64, eps=1e-5),
                                             nn.ReLU(),
                                             )
        init_weight_(self.paragraph_cnet1)

        self.bar_cnet1 = BarUnit(self.eta)

        self.bar_cnet2 = BarUnit(self.eta)

        self.bar_cnet3 = BarUnit(self.eta)

        self.bar_cnet4 = BarUnit(self.eta)

        init_weight_(self.bar_cnet1)
        init_weight_(self.bar_cnet2)
        init_weight_(self.bar_cnet3)
        init_weight_(self.bar_cnet4)

        self.paragraph_cnet2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                       out_channels=128,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       bias=False),
                                             nn.BatchNorm2d(128, eps=1e-5),
                                             nn.LeakyReLU(0.2),

                                             nn.Conv2d(in_channels=128,
                                                       out_channels=256,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       bias=False),
                                             nn.BatchNorm2d(256, eps=1e-5),
                                             nn.LeakyReLU(0.2)
                                             )

        self.paragraph_cnet22= nn.Sequential(nn.Conv2d(in_channels=256,
                                                       out_channels=512,
                                                       kernel_size=3,
                                                       stride=(2, 3),
                                                       padding=1,
                                                       bias=False),
                                             nn.BatchNorm2d(512, eps=1e-5),
                                             nn.LeakyReLU(0.2)
                                             )

        init_weight_(self.paragraph_cnet2)
        init_weight_(self.paragraph_cnet22)

        '''
        self.resnet = nn.Sequential()
        for i in range(12):
            self.resnet.add_module('resnet_block', ResnetBlock(dim=256,
                                                               padding_type='reflect',
                                                               use_dropout=False,
                                                               use_bias=False,
                                                               norm_layer=nn.InstanceNorm2d))
        init_weight_(self.resnet)
        '''

        self.paragraph_ctnet11 = nn.Sequential(nn.ConvTranspose2d(in_channels=512,
                                                                 out_channels=256,
                                                                 kernel_size=3,
                                                                 stride=(2, 3),
                                                                 padding=1,
                                                                 bias=False),
                                               nn.ZeroPad2d((0, 2, 0, 1)),
                                              nn.BatchNorm2d(256, eps=1e-5),
                                              nn.LeakyReLU(0.2)
                                              )

        self.paragraph_ctnet1 = nn.Sequential(nn.ConvTranspose2d(in_channels=256,
                                                                 out_channels=128,
                                                                 kernel_size=3,
                                                                 stride=2,
                                                                 padding=1,
                                                                 bias=False),
                                              nn.ZeroPad2d((0, 1, 0, 1)),
                                              nn.BatchNorm2d(128, eps=1e-5),
                                              nn.LeakyReLU(0.2),

                                              nn.ConvTranspose2d(in_channels=128,
                                                                 out_channels=64,
                                                                 kernel_size=3,
                                                                 stride=2,
                                                                 padding=1,
                                                                 bias=False),
                                              nn.ZeroPad2d((1, 0, 1, 0)),
                                              nn.BatchNorm2d(64, eps=1e-5),
                                              nn.LeakyReLU(0.2)
                                              )
        init_weight_(self.paragraph_ctnet11)
        init_weight_(self.paragraph_ctnet1)

        self.paragraph_cnet3 = nn.Sequential(nn.ReflectionPad2d((2, 2, 2, 2)),
                                             nn.Conv2d(in_channels=64,
                                                       out_channels=1,
                                                       kernel_size=5,
                                                       stride=1,
                                                       padding=0),
                                             nn.Sigmoid()
                                             )
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

