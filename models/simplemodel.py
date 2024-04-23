# Generator Code
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Size of z latent vector (i.e. size of generator input)
Z_dim = 100 #same with mysimplegan
# Number of channels: 3 for RGB
Number_Channel = 3
# Size of feature maps in discriminator
Number_Feature_Dis = 64
# Size of feature maps in generator
Number_Feature_Gen = 64
# Number of GPUs available. Use 0 for CPU mode.
Number_GPU = 1

class Generator(nn.Module):
    def __init__(self, Number_GPU):
        super(Generator, self).__init__()
        self.number_GPU = Number_GPU
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(Z_dim, Number_Feature_Gen * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(Number_Feature_Gen * 8),
            nn.ReLU(True),
            # state size. (Number_Feature_Gen*8) x 4 x 4
            nn.ConvTranspose2d(Number_Feature_Gen * 8, Number_Feature_Gen * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Number_Feature_Gen * 4),
            nn.ReLU(True),
            # state size. (Number_Feature_Gen*4) x 8 x 8
            nn.ConvTranspose2d(Number_Feature_Gen * 4, Number_Feature_Gen * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Number_Feature_Gen * 2),
            nn.ReLU(True),
            # state size. (Number_Feature_Gen*2) x 16 x 16
            nn.ConvTranspose2d(Number_Feature_Gen * 2, Number_Feature_Gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Number_Feature_Gen),
            nn.ReLU(True),
            # state size. (Number_Feature_Gen) x 32 x 32
            nn.ConvTranspose2d(Number_Feature_Gen, Number_Channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (Number_Channel) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, Number_GPU):
        super(Discriminator, self).__init__()
        self.number_GPU = Number_GPU
        self.main = nn.Sequential(
            # input is RGB: (3) x 64 x 64
            nn.Conv2d(Number_Channel, Number_Feature_Dis, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(Number_Feature_Dis, Number_Feature_Dis * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Number_Feature_Dis * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (Number_Feature_Dis*2) x 16 x 16
            nn.Conv2d(Number_Feature_Dis * 2, Number_Feature_Dis * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Number_Feature_Dis * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (Number_Feature_Dis*4) x 8 x 8
            nn.Conv2d(Number_Feature_Dis * 4, Number_Feature_Dis * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Number_Feature_Dis * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (Number_Feature_Dis*8) x 4 x 4
            nn.Conv2d(Number_Feature_Dis * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

