"""
Load Au + Au TPC data
"""
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset

AXIS_MAP = {'azimuth': 0,
            'beam': 1,
            'layer': 2}
DEFAULT_ORDER = ('azimuth', 'beam', 'layer')


class DatasetTPC(Dataset):
    """
    Load TPC data as multi-channel 2d dataset
    """
    def __init__(self,
                 dataroot,
                 split      = 'train',
                 dimension  = 2,
                 axis_order = ('azimuth', 'beam', 'layer')):
        """
        Input
        ========
        manifest (str): The filename of the data manifest.
            Each line in file is an absolute path of a data file.
        """
        super().__init__()

        dataroot = Path(dataroot)
        assert dataroot.exists(), f'{dataroot} does not exist!'

        manifest = dataroot/f'{split}.txt'
        assert manifest.exists(), f'{manifest} does not exist!'

        with open(manifest, 'r') as file_handle:
            fnames = file_handle.read().splitlines()
        self.fnames = [dataroot/fname for fname in fnames]

        assert set(axis_order) == AXIS_MAP.keys()
        if tuple(axis_order) != DEFAULT_ORDER:
            self.permute = tuple(AXIS_MAP[axis] for axis in axis_order)

        assert dimension in (2, 3)
        self.dimension = dimension

        print(f'load Au + Au TPC data in {axis_order}')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):

        fname = self.fnames[index]
        datum = torch.tensor(np.load(fname).astype(np.float32))

        if hasattr(self, 'permute'):
            datum = datum.permute(self.permute)

        if self.dimension == 3:
            datum = datum.unsqueeze(0)

        return datum


# from pathlib import Path
#
# ROOT = Path('/data/yhuang2/sphenix/auau/highest_framedata_3d/outer')
# def test():
#
#     for split in ('train', 'test'):
#         dataset = DatasetTPC(ROOT, split, dimension=3)
#         print(f'The {split} dataset contains {len(dataset)} samples.')
#         for data in dataset:
#             print(data.shape)
#             break
#
# test()
