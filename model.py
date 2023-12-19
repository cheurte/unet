import torch
import torch.nn as nn
import torchvision

class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = [Block(channels[i], channels[i+1]) for i in range(len(channels)-1)]
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        enc_feat = []
        for block in self.conv_block:
            x = block(x)
            enc_feat.append(x)
            x = self.down(x)

        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))

class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)])
        self.dec_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.upconvs)):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class Unet(nn.Module):
    def __init__(self, in_channels, nb_classes):
        super().__init__()
        self.channels = [64, 128, 256, 512]
        self.encoder = Encoder([in_channels] + self.channels)
        self.decoder = Decoder(self.channels[::-1] + [nb_classes])
        self.pred = nn.Conv2d(self.channels[-1], nb_classes, 1)

    def forward(self, x):
        enc_feat = self.encoder(x)
        out = self.decoder(enc_feat[::-1][-1], enc_feat[::-1][1:])
        out = self.pred(out)
        return out
