import torch
from torch import nn

class Bloco_Disc(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Bloco_Disc, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):

    def __init__(self, in_channels, features=[128, 256, 512]):
        super(Discriminator, self).__init__()
        self.conv_inicial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )

        camadas = []
        in_channels = features[0]
        for feature in features[1:]:
            camadas.append(
                Bloco_Disc(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        
        camadas.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )

        self.model = nn.Sequential(*camadas)
    
    def forward(self, x, y):
        x1 = torch.cat([x, y], dim=1)
        x2 = self.conv_inicial(x1)
        return self.model(x2)

def test_Discriminator():
    x = torch.rand((5, 3, 128, 128))
    y = torch.rand((5, 3, 128, 128))
    discriminator = Discriminator(in_channels=3)
    output = discriminator(x, y)
    print (f'Testando discriminator...')
    print (f'x.shape: {x.shape}, y.shape: {y.shape}, output.shape: {output.shape}')

class Bloco_CNN(nn.Module):

    def __init__(self, in_channels, out_channels, tipo='conv', ativacao='relu', usar_dropout=False):
        super(Bloco_CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tipo = tipo # tipo pode ser 'conv' ou 'convT'
        self.ativacao = ativacao # pode ser relu ou leaky
        self.usar_dropout = usar_dropout

        camadas = []
        camadas.append(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect')
            if tipo == 'conv' else
            nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False))
        camadas.append(nn.BatchNorm2d(num_features=self.out_channels))
        camadas.append(nn.ReLU() if self.ativacao == 'relu' else nn.LeakyReLU(negative_slope=0.2))

        if (self.usar_dropout): camadas.append(nn.Dropout(p=0.5))

        self.conv = nn.Sequential(*camadas)
    
    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):

    def __init__(self, in_channels=3, features=64):
        super(Generator, self).__init__()

        self.in_channels = in_channels

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        self.conv1 = Bloco_CNN(features, features*2, tipo='conv', ativacao='leaky', usar_dropout=False)
        self.conv2 = Bloco_CNN(features*2, features*4, tipo='conv', ativacao='leaky', usar_dropout=False)
        self.conv3 = Bloco_CNN(features*4, features*8, tipo='conv', ativacao='leaky', usar_dropout=False)
        self.conv4 = Bloco_CNN(features*8, features*8, tipo='conv', ativacao='leaky', usar_dropout=False)
        self.conv5 = Bloco_CNN(features*8, features*8, tipo='conv', ativacao='leaky', usar_dropout=False)
        # self.conv6 = Bloco_CNN(features*8, features*8, tipo='conv', ativacao='leaky', usar_dropout=False)

        self.bottleneck = nn.Sequential(nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1), nn.ReLU())

        # self.up1 = Bloco_CNN(features*8, features*8, tipo='convT', ativacao='relu', usar_dropout=True)
        self.up2 = Bloco_CNN(features*8, features*8, tipo='convT', ativacao='relu', usar_dropout=True)
        self.up3 = Bloco_CNN(features*8*2, features*8, tipo='convT', ativacao='relu', usar_dropout=True)
        self.up4 = Bloco_CNN(features*8*2, features*8, tipo='convT', ativacao='relu', usar_dropout=False)
        self.up5 = Bloco_CNN(features*8*2, features*4, tipo='convT', ativacao='relu', usar_dropout=False)
        self.up6 = Bloco_CNN(features*4*2, features*2, tipo='convT', ativacao='relu', usar_dropout=False)
        self.up7 = Bloco_CNN(features*2*2, features*1, tipo='convT', ativacao='relu', usar_dropout=False)

        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(features*2, self.in_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x0):
        x1 = self.initial_conv(x0)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x6 = self.conv5(x5)
        # x7 = self.conv6(x6)
        
        bottleneck = self.bottleneck(x6)

        # up1 = self.up1(bottleneck)
        # up2 = self.up2(torch.cat([up1, x7], 1))
        up2 = self.up2(bottleneck)
        up3 = self.up3(torch.cat([up2, x6], 1))
        up4 = self.up4(torch.cat([up3, x5], 1))
        up5 = self.up5(torch.cat([up4, x4], 1))
        up6 = self.up6(torch.cat([up5, x3], 1))
        up7 = self.up7(torch.cat([up6, x2], 1))

        return self.final_conv(torch.cat([up7, x1], 1))


def test_generator():
    generator = Generator(3, 64)
    random_input = torch.rand(size=(5, 3, 128, 128))
    output = generator(random_input)
    print (f'Testando generator...')
    print (f'random_input.shape: {random_input.shape}, output.shape: {output.shape}')


if __name__ == '__main__':
    test_generator()
    test_Discriminator()