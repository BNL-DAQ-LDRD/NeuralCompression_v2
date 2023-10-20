"""
Author:
    Yi Huang, yhuang2@bnl.gov
"""
import torch
from torch import nn

from neuralcompress_v2.utils.utils import (get_norm_layer3d,
                                           get_activ_layer)

# default set up
CONV_1 = {
    'out_channels': 8,
    'kernel_size' : [3, 4, 4],
    'padding'     : [1, 1, 1],
    'stride'      : [1, 2, 2]
}
CONV_2 = {
    'out_channels': 16,
    'kernel_size' : [3, 4, 4],
    'padding'     : [1, 1, 1],
    'stride'      : [1, 2, 2]
}
CONV_3 = {
    'out_channels': 32,
    'kernel_size' : [3, 4, 4],
    'padding'     : [1, 1, 1],
    'stride'      : [1, 2, 2]
}
CONV_4 = {
    'out_channels': 32,
    'kernel_size' : [3, 4, 4],
    'padding'     : [1, 1, 1],
    'stride'      : [1, 2, 2]
}

CONV_ARGS_LIST = (CONV_1, CONV_2, CONV_3, CONV_4)

DECONV_1 = {
    'out_channels': 16,
    'kernel_size' : [3, 4, 4],
    'padding'     : [1, 1, 1],
    'stride'      : [1, 2, 2],
    'output_padding': 0
}
DECONV_2 = {
    'out_channels': 8,
    'kernel_size' : [3, 4, 4],
    'padding'     : [1, 1, 1],
    'stride'      : [1, 2, 2],
    'output_padding': 0
}
DECONV_3 = {
    'out_channels': 4,
    'kernel_size' : [3, 4, 4],
    'padding'     : [1, 1, 1],
    'stride'      : [1, 2, 2],
    'output_padding': 0
}
DECONV_4 = {
    'out_channels': 2,
    'kernel_size' : [3, 4, 4],
    'padding'     : [1, 1, 1],
    'stride'      : [1, 2, 2],
    'output_padding': 0
}

DECONV_ARGS_LIST = (DECONV_1, DECONV_2, DECONV_3, DECONV_4)

CODE_CHANNELS = 8


def single_block(block_type, block_args, norm, activ):

    assert block_type in ('conv', 'deconv')

    if block_type == 'conv':
        layer = nn.Conv3d(**block_args)
    else:
        layer = nn.ConvTranspose3d(**block_args)

    out_channels =  block_args['out_channels']
    return nn.Sequential(layer,
                         get_norm_layer3d(norm, out_channels),
                         get_activ_layer(activ))


def double_block(block_type, block_args, norm, activ):

    block_1 = single_block(block_type, block_args, norm, activ)
    block_2 = nn.Conv3d(block_args['out_channels'],
                        block_args['out_channels'],
                        kernel_size = 3,
                        padding = 1)

    return nn.Sequential(*block_1, block_2)


class ResidualBlock(nn.Module):
    """
    A residual block that has a double block on the main path
    and a single block on the side path.
    """
    def __init__(self,
                 main_block,
                 side_block,
                 norm,
                 activ,
                 rezero = True):
        """
        Input:
            - main_block (nn.Module): the network block on the main path
            - side_block (nn.Module): the network block on the side path
            - activ: activation function;
            - norm: normalization function;
        Output:
        """
        super().__init__()

        out_channels = main_block[0].out_channels

        assert main_block[0].in_channels == side_block[0].in_channels, \
            ('main-path block and side-path block'
             'must have the same in_channels')
        assert main_block[0].out_channels == side_block[0].out_channels, \
            ('main-path block and side-path block'
             'must have the same out_channels')

        self.main_block = main_block
        self.side_block = side_block

        self.norm = get_norm_layer3d(norm, out_channels)
        self.activ = get_activ_layer(activ)

        if rezero:
            self.rezero_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.rezero_alpha = 1

    def forward(self, data):
        """
        data shape: (N, C, D, H, W)
            - N = batch_size;
            - C = channels;
            - D, H, W: the three spatial dimensions
        """
        x_side   = self.side_block(data)
        x_main   = self.main_block(data)
        x_output = self.rezero_alpha * x_main + x_side
        return self.activ(self.norm(x_output))


def encoder_block(conv_args, norm, activ, rezero = True):
    """
    Get an encoder residual block.
    """

    main_block = double_block('conv', conv_args, norm, activ)
    side_block = single_block('conv', conv_args, norm, activ)

    return ResidualBlock(main_block = main_block,
                         side_block = side_block,
                         norm       = norm,
                         activ      = activ,
                         rezero     = rezero)


def decoder_block(deconv_args, norm, activ, rezero = True):
    """
    Get an decoder residual block.
    """
    main_block = double_block('deconv', deconv_args, norm, activ)
    side_block = single_block('deconv', deconv_args, norm, activ)

    return ResidualBlock(main_block = main_block,
                         side_block = side_block,
                         norm       = norm,
                         activ      = activ,
                         rezero     = rezero)


class Encoder(nn.Module):
    """
    Encoder with a few downsampling layers plus an output layer.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 in_channels    = 1,
                 out_channels   = CODE_CHANNELS,
                 conv_args_list = CONV_ARGS_LIST,
                 conv_features  = None,
                 norm           = None,
                 activ          = {'name'           : 'leakyrelu',
                                   'negative_slope' : .2},
                 rezero         = True):

        super().__init__()

        # Downsampling layers
        self.layers = nn.Sequential()
        in_ch = in_channels

        for idx, conv_args in enumerate(conv_args_list):

            if conv_features is not None:
                conv_args['out_channels'] = conv_features[idx]

            conv_args['in_channels'] = in_ch

            layer = encoder_block(conv_args, norm, activ, rezero = rezero)

            self.layers.add_module(f'encoder_block_{idx}', layer)
            in_ch = conv_args['out_channels']

        # Encoder output layer
        block_args = {'in_channels'  : in_ch,
                      'out_channels' : out_channels,
                      'kernel_size'  : 3,
                      'padding'      : 1}

        output_layer = single_block('conv', block_args, None, None)
        self.layers.add_module('encoder_output', output_layer)

    def forward(self, data):
        """
        data shape: (N, C, D, H, W)
            - N = batch_size;
            - C = in_channels;
            - D, H, W: the three spatial dimensions
        """
        return self.layers(data)


class Decoder(nn.Module):
    """
    Decoder with a few upsampling layers plus an output layer.
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 in_channels,
                 out_channels,
                 deconv_args_list,
                 norm         = None,
                 activ        = {'name'           : 'leakyrelu',
                                   'negative_slope' : .2},
                 output_activ = None,
                 rezero       = True):
        """
        Input:
            - in_channels (int): in_channels of the first deconvolution
                layers.
            - conv_args_list (list of dictionary): arguments for the
                deconvolution/upsampling layers. Each entry in the list is
                a dictionary contains the following keys:
                - out_channels;
                - kernel_size;
                - stride;
                - padding;
                - output_padding;
            - activ: activation layer;
            - norm: normalization function (a normalization function without
                parameter. Need to be initialized with parameter.)
            - output_channels (int): out_channels in the output layer.
            - output_activ: output activation layer;
        """
        super().__init__()

        # Upsampling layers

        in_ch = in_channels
        self.layers = nn.Sequential()

        for idx, deconv_args in enumerate(deconv_args_list):
            deconv_args['in_channels'] = in_ch

            layer = decoder_block(deconv_args, norm, activ, rezero = rezero)

            self.layers.add_module(f'decoder_block_{idx}', layer)
            in_ch = deconv_args['out_channels']

        # Decoder output layer
        block_args = {'in_channels'  : in_ch,
                      'out_channels' : out_channels,
                      'kernel_size'  : 3,
                      'padding'      : 1}
        output_layer = single_block(block_type = 'conv',
                                    block_args = block_args,
                                    norm       = None,
                                    activ      = output_activ)
        self.layers.add_module('decoder_output', output_layer)

    def forward(self, data):
        """
        data shape: (N, C, D, H, W)
            - N = batch_size;
            - C = in_channels;
            - D, H, W: the three spatial dimensions
        """
        return self.layers(data)


class BiDecoder(nn.Module):
    """
    BCAE decoder with two heads.
    """

    def __init__(self,
                 in_channels      = CODE_CHANNELS,
                 out_channels     = 1,
                 deconv_args_list = DECONV_ARGS_LIST,
                 norm             = None,
                 activ            = {'name'           : 'leakyrelu',
                                     'negative_slope' : .2},
                 rezero           = True):

        """
        in_channels = code_channels;
        output_channels = image_channels;
        """

        super().__init__()

        # set up the network
        args = {'in_channels'      : in_channels,
                'out_channels'     : out_channels,
                'deconv_args_list' : deconv_args_list,
                'norm'             : norm,
                'activ'            : activ}

        self.decoder_clf = Decoder(**args,
                                   output_activ = 'sigmoid',
                                   rezero       = rezero)
        self.decoder_reg = Decoder(**args,
                                   output_activ = None,
                                   rezero       = rezero)

    def forward(self, code):
        """
        data shape: (N, C, D, H, W)
            - N = batch_size;
            - C = image_channels;
            - D, H, W: the three spatial dimensions
        """
        output_clf = self.decoder_clf(code)
        output_reg = self.decoder_reg(code)
        return output_clf, output_reg
