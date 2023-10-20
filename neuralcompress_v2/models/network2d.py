"""
2d Autoencoder for TPC data compression
"""

import torch
from torch import nn

class ResidualBlock(nn.Module):
    """
    Description
    ========
    The standard residual block with a possible 1x1 conv on
    the skip path.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 rezero = True):

        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels,
                                             out_channels,
                                             kernel_size=3,
                                             padding=1),
                                   # nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU(negative_slope=.2))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels,
                                             out_channels,
                                             kernel_size=3,
                                             padding=1),)
                                   # nn.BatchNorm2d(out_channels))
        self.skip = nn.Identity()
        if out_channels != in_channels:
            self.skip = nn.Sequential(nn.Conv2d(in_channels,
                                                out_channels,
                                                kernel_size=1),)
                                      # nn.BatchNorm2d(out_channels))
        self.relu = nn.LeakyReLU(negative_slope=.2)

        if rezero:
            self.rezero_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.rezero_alpha = 1

    def forward(self, data):
        """
        Input
        ========
        data: tensor of shape (batch_size, num_channels, height, width)
        """
        residual = self.skip(data)
        data = self.conv1(data)
        data = self.conv2(data)
        return self.relu(self.rezero_alpha * data + residual)


class Encoder(nn.Module):
    """
    Description
    ========
    Compressive encoder
    """
    def __init__(self, in_channels, num_blocks, num_downsamples):
        super().__init__()

        self.init_conv = nn.Conv2d(in_channels,
                                   in_channels * 2,
                                   kernel_size = 7,
                                   padding = 3)
        layers = []
        in_ch = in_channels
        for block_idx in range(num_blocks):

            if block_idx < num_downsamples:
                layers += [nn.AvgPool2d(kernel_size = 2, stride = 2)]

            layers += [ResidualBlock(in_ch * 2, in_ch * 2),
                       ResidualBlock(in_ch * 2, in_ch * 2)]

        self.layers = nn.Sequential(*layers)

    def forward(self, data):
        data = self.init_conv(data)
        data = self.layers(data)
        return data


class Decoder(nn.Module):
    """
    Description
    ========
    Regression decoder
    """
    def __init__(self,
                 in_channels,
                 num_blocks,
                 num_upsamples,
                 output_activ = None):
        super().__init__()

        layers = []
        in_ch = in_channels

        for block_idx in range(num_blocks):

            if block_idx < num_upsamples:
                layers += [nn.Upsample(scale_factor = 2)]

            layers += [ResidualBlock(in_ch * 2, in_ch * 2),
                       ResidualBlock(in_ch * 2, in_ch * 2)]

        self.layers = nn.Sequential(*layers)

        self.output_layer = nn.Conv2d(in_channels * 2,
                                      in_channels,
                                      kernel_size = 1)

        if output_activ == 'sigmoid':
            self.output_activ = nn.Sigmoid()
        else:
            self.output_activ = nn.Identity()

    def forward(self, data):
        data = self.layers(data)
        data = self.output_layer(data)
        return self.output_activ(data)


class BiDecoder(nn.Module):
    def __init__(self, in_channels, num_blocks, num_downsamples):
        super().__init__()
        self.decoder_clf = Decoder(in_channels,
                                   num_blocks,
                                   num_downsamples,
                                   output_activ = 'sigmoid')
        self.decoder_reg = Decoder(in_channels,
                                   num_blocks,
                                   num_downsamples)

    def forward(self, data):
        output_clf = self.decoder_clf(data)
        output_reg = self.decoder_reg(data)
        return output_clf, output_reg
