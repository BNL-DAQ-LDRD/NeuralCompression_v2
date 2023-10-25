#!/usr/bin/env python
import os
import argparse
from argparse import RawTextHelpFormatter
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.jit import load
from torch.nn.functional import pad

from neuralcompress_v2.datasets.dataset import DatasetTPC

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

def get_args(description):
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description,
                                     formatter_class = RawTextHelpFormatter)
    # data parameters
    parser.add_argument('--data-path',
                        type    = str,
                        help    = 'Location of the data.\n\n')
    parser.add_argument('--dimension',
                        type    = int,
                        choices = (2, 3),
                        default = 2,
                        help    = ('The dimension the data is loaded as.\n'
                                   '(default = 2)\n\n'))
    parser.add_argument('--log',
                        type    = int,
                        choices = (0, 1),
                        default = 1,
                        help    = ('Whether to use log-scaled ADC value.\n'
                                   '    if log == 0, use raw ADC,\n'
                                   '    if log == 1, use log-scaled ADC value '
                                   'log2(ADC + 1).\n'
                                   '(default = 1).\n\n'))
    parser.add_argument('--transform',
                        type    = int,
                        choices = (0, 1),
                        default = 1,
                        help    = ('Whether to transform the regression output.\n'
                                   '    if transform == 0, no transform is applied.\n'
                                   '    if transform == 1, a transform function '
                                   'will be applied:\n'
                                   '        if log == 0, transform = 64 + 6 * exp(x).\n'
                                   '        if log == 1, transform =  6 + 3 * exp(x),\n'
                                   '(default = 1).\n\n'))
    # model
    parser.add_argument('--checkpoint-path',
                        type    = str,
                        help    = 'Path to the pretrained models.\n\n')
    parser.add_argument('--clf-threshold',
                        type    = float,
                        default = 0.5,
                        help    = ('Threshold for classification output.\n'
                                   '(default = 0.5).\n\n'))
    # device and compute mode
    parser.add_argument('--device',
                        type    = str,
                        default = 'cuda',
                        choices = ('cuda', 'cpu'),
                        help    = 'Device.\n(default = cuda)\n\n')
    parser.add_argument('--gpu-id',
                        type    = int,
                        default = 0,
                        help    = ('ID of GPU card. Only effective when '
                                   'device is cuda.\n(default = 0)\n\n'))
    parser.add_argument('--half',
                        action  = 'store_true',
                        help    = ('Use the flag to turn on inference in '
                                   'half precision\n\n'))
    parser.add_argument('--save-path',
                        type    = str,
                        help    = 'Location to save the results.\n\n')
    parser.add_argument('--num-test-examples',
                        type    = int,
                        default = float('inf'),
                        help    = ('Number of test examples\n'
                                   '(default = inf)\n\n'))

    return parser.parse_args()


def get_metrics(data, reco):
    """
    """
    true = (data > 0).sum()
    pos = (reco > 0).sum()
    true_pos = ((data > 0) * (reco > 0)).sum()

    mse = torch.pow(data - reco, 2).mean()
    psnr = data.max() * torch.rsqrt(mse)

    return {'occupancy': ((data > 0).sum() / np.prod(data.shape)).item(),
            'mse': mse.item(),
            'psnr': psnr.item(),
            'mae': torch.abs(data - reco).mean().item(),
            'precision': (true_pos / pos).item(),
            'recall': (true_pos / true).item()}


def main():

    args = get_args('TPC Data Compression test and inference')

    # data parameters
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise IOException(f'Path does not exist: {data_path}')

    dimension = args.dimension
    log       = args.log
    transform = args.transform

    clf_threshold = args.clf_threshold

    # model path
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise IOException(f'Path does not exist: {checkpoint_path}')

    # device and compute mode
    device = args.device
    if device == 'cuda':
        torch.cuda.set_device(args.gpu_id)
    half   = args.half

    save_path = Path(args.save_path)
    assert not save_path.exists(), \
        f'{save_path} exists, refuse to overwrite.'
    save_path.mkdir(parents=True)
    frame_path = save_path/'frames'
    frame_path.mkdir()
    num_test_examples = args.num_test_examples

    # load data
    dataset = DatasetTPC(data_path,
                         split      = 'test',
                         dimension  = dimension,
                         axis_order = ('layer', 'azimuth', 'beam'))
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)

    # load model
    encoder = load(checkpoint_path/'enc_last.pth').to(device)
    decoder = load(checkpoint_path/'dec_last.pth').to(device)

    if half:
        encoder = encoder.half()

    # compress, decompress, and evaluate
    total = min(len(dataloader), num_test_examples)
    pbar = tqdm('test', total = total)
    result = defaultdict(list)
    for idx, data in tqdm(enumerate(dataloader)):

        if idx == total:
            break

        data = data.to(device)

        if log == 1:
            data = torch.log2(data + 1.)

        if half:
            data = data.half()

        code = encoder(pad(data, (0, 7))).half()

        output_clf, output_reg = decoder(code.to(torch.float32))

        if transform == 1:
            shift = 6. if log else 64.
            expnt = 3. if log else 6.

            output_reg = shift + expnt * torch.exp(output_reg)

        reco = output_reg * (output_clf > clf_threshold)

        reco = reco[..., :-7]

        np.savez_compressed(frame_path/f'sample_{idx}',
                            input          = data.detach().cpu().numpy(),
                            code           = code.detach().cpu().numpy(),
                            reconstruction = reco.detach().cpu().numpy())

        metrics = get_metrics(data, reco)
        pbar.update()
        pbar.set_postfix(metrics)

        for key, val in metrics.items():
            result[key].append(val)

    dataframe = pd.DataFrame(data = result)
    dataframe.to_csv(save_path/'metrics.csv',
                     index = False,
                     float_format = '%.6f')

main()
