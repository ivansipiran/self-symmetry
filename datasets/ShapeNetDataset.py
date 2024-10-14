import os
import torch
import glob
import numpy as np
import torch.utils.data as data
import lzma

class ShapeNetDataset(data.Dataset):
    def __init__(self, root, num_points, category=None):
        self.root = root
        self.category = category

        if category is None:
            self.files = sorted(glob.glob(self.root + '/**/*.xz', recursive=True))
        else:
            self.files = sorted(glob.glob(self.root + '/{}/*.xz'.format(category), recursive=True))
        self.num_points = num_points

    def __getitem__(self, index):
        with lzma.open(self.files[index], 'rb') as f:
            data = np.loadtxt(f)
        category = self.category if self.category is not None else self.files[index].split('/')[-2]
        name = self.files[index].split('/')[-1].split('.')[0]
        return torch.from_numpy(data).float(), category, name
    
    def __len__(self):
        return len(self.files)


if __name__=='__main__':
    dataset = ShapeNetDataset('/data/Datasets2/ShapeNetCore.v2/symmetry_pc', 2048, category='table')
    print(len(dataset))
    