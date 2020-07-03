import torch.utils.data as data
import numpy as np
import random

from riffgan.data.create_dataset import *


class UnitRiffDataset(data.Dataset):
    def __init__(self, instr):

        assert instr in ['guitar', 'bass']
        self.instr = instr

        dataset_dict = {
            'guitar': 'E:/grunge_library/data/guitar_unit_riff.npz',
            'bass': 'E:/grunge_library/data/bass_unit_riff.npz'
        }
        self.dataset_path = dataset_dict[self.instr]

        self.data = generate_from_nonzeros(self.instr)

    def __getitem__(self, item):
        return self.data[item, :, :]

    def __len__(self):
        return self.data.shape[0]

    def get_data(self):
        return self.data

