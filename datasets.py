from torch.utils.data import Dataset, DataLoader
import os
from os import listdir
from PIL import Image
import pandas as pd
import torch
import random
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt


class Images(Dataset):
    def __init__(self, img_path, label_path, transform):
        self.img_path = img_path
        self.caption = pd.read_pickle(label_path)
        self.imgs = [f for f in listdir(img_path)]
        self.transfrom = transform

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = Image.open(os.path.join(self.img_path, img_name))
        img = self.transfrom(img)
        caption = self.caption[self.caption['image'] == img_name]
        data = {'image': img, 'img_name': caption['image'].tolist()[0], 'caption': np.stack(caption['caption'].tolist()[0])}
        return data

    def __len__(self):
        return len(self.imgs)


class ImageCodes(Dataset):
    def __init__(self, data_path, split='train'):
        data = pickle.load(open(data_path, 'rb'))
        if split == 'test':
            self.codes = data['image'][0:1000, ...]
            self.captions = data['caption'][0:1000, ...]
            self.imgNames = data['img_name'][0:1000]
        if split == 'val':
            self.codes = data['image'][1000:2000, ...]
            self.captions = data['caption'][1000:2000, ...]
            self.imgNames = data['img_name'][1000:2000]
        if split == 'train':
            self.codes = data['image'][2000:, ...]
            self.captions = data['caption'][2000:, ...]
            self.imgNames = data['img_name'][2000:]

    def __getitem__(self, idx):
        code = self.codes[idx, :]
        caption = self.captions[idx, ...]
        imgNames = self.imgNames[idx]

        caption = caption[random.randint(0, caption.shape[0] - 1), :]
        return code, caption, imgNames

    def __len__(self):
        return len(self.imgNames)

