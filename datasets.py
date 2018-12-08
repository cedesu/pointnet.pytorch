from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json


class PartDataset(data.Dataset):
    def __init__(self, root, train = True):
        self.root = root
        self.path=os.listdir(root)
        self.maxn=100

        if train:
            self.path = self.path[:int(len(self.path) * 0.9)]
        else:
            self.path = self.path[int(len(self.path) * 0.9):]

    def __getitem__(self, index):
        try:
            print(index)
            pathi = self.path[index]
            f = open(os.path.join(self.root, pathi), 'r')
            sdf = list(map(float, f.readline().split()))
            sdf = np.array(sdf).reshape(self.maxn, self.maxn, self.maxn)
            out_sdf = np.zeros((1, 32, 32, 32))
            for i in range(32):
                for j in range(32):
                    for k in range(32):
                        out_sdf[0, i, j, k] = sdf[i * 3 + 3, j * 3 + 3, k * 3 + 3]
            in_voxel = np.sign(out_sdf)

            in_voxel = torch.from_numpy(in_voxel)
            out_sdf = torch.from_numpy(out_sdf)

            in_voxel = in_voxel.float()
            out_sdf = out_sdf.float()

            return in_voxel, out_sdf
        except:
            print(index)
            index+=1
            pathi = self.path[index]
            f = open(os.path.join(self.root, pathi), 'r')
            sdf = list(map(float, f.readline().split()))
            sdf = np.array(sdf).reshape(self.maxn, self.maxn, self.maxn)
            out_sdf = np.zeros((1, 1, 32, 32, 32))
            for i in range(32):
                for j in range(32):
                    for k in range(32):
                        out_sdf[0, 0, i, j, k] = sdf[i * 3 + 3, j * 3 + 3, k * 3 + 3]
            in_voxel = np.sign(out_sdf)

            in_voxel = torch.from_numpy(in_voxel)
            out_sdf = torch.from_numpy(out_sdf)

            in_voxel = in_voxel.float()
            out_sdf = out_sdf.float()
            print('safe')

            return in_voxel, out_sdf

    def __len__(self):
        return len(self.path)


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'sdf',train=True)
    print(len(d))
    inn,out = d[0]
    print(inn.size(), inn.type(), out.size(),out.type())

