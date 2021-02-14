import pickle
from collections import Counter
import ast
from mobilenetv2.models.imagenet import mobilenetv2
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import pickle
from collections import Counter
import pandas as pd
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from datasets import Images
import os
import h5py
import numpy as np


def cap_len(x):
    l = max([len(z) for z in x])
    return l


def max_caption_length(captions):
    lengths = captions.apply(cap_len)
    return lengths.max()


def split_words(x):
    filter = ['.', '?', '"', "!", ":", "(", ")", ",", "#", "$", "&", "*", "/", ":", ";", "\n", "\t"]
    captions = [y.split(" ") for y in x]
    captions = [[z for z in y if z not in filter] for y in captions]
    return captions


def pad_sequences(x, length):
    for z in x:
        while len(z) != length:
            z.append(0)
    return x


def word_mapping(captions):
    concat = captions.apply(lambda captions: [word for caption in captions for word in caption]).tolist()
    flat_list = [word.lower() for sublist in concat for word in sublist]
    word_dict = dict(Counter(flat_list))
    word_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}
    word_dict = {k: v for k, v in word_dict.items() if v >= 5}
    word_to_idx = {k: idx + 1 for idx, (k, v) in enumerate(word_dict.items())}
    word_to_idx['_'] = 0
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    return word_to_idx, idx_to_word, word_dict


def preprocess_images():
    img_path = 'archive\\Images'
    label_path = 'data\\targets.pickle'
    model = mobilenetv2().cuda()
    model.load_state_dict(torch.load('mobilenetv2\\pretrained\\mobilenetv2_128x128-fd66a69d.pth'))
    modules1 = [list(model.children())[0]]
    modules2 = [list(model.children())[1][0]]
    modules = modules1 + modules2
    model = nn.Sequential(*modules)
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    images = Images(img_path, label_path, transform)
    dataloader = DataLoader(images, batch_size=128)
    print('preprocessing...')
    code_data = {}
    for idx, (data) in enumerate(dataloader):
        print(f"{round(idx/len(dataloader) * 100, 2)}%", end='\r')
        with torch.no_grad():
            code = model(data['image'].cuda())
            code = torch.flatten(code, 1, -1)
            if idx == 0:
                code_data['image'] = code.cpu()
                code_data['img_name'] = data['img_name']
                code_data['caption'] = data['caption']
            else:
                code_data['image'] = torch.cat((code_data['image'], code.cpu()))
                code_data['img_name'] += data['img_name']
                code_data['caption'] = torch.cat((code_data['caption'], data['caption']))
    print(code_data['image'].shape, len(code_data['img_name']), code_data['caption'].shape)
    pickle.dump(code_data, open('data\\imgCodes.pickle', 'wb'))


def preprocess_labels():
    df = pd.read_csv('archive\\captions.txt')
    df = df.groupby(['image'])['caption'].apply(list).reset_index()
    df['caption'] = df['caption'].apply(split_words)
    word_to_idx, idx_to_word, word_dict = word_mapping(df['caption'])
    df['caption'] = df['caption'].apply(
        lambda captions: [[word.lower() for word in caption if word.lower() in word_dict] for caption in captions])
    df.to_csv('data\\targetsword.csv')
    df['caption'] = df['caption'].apply(lambda captions: [[word_to_idx[word.lower()] for word in caption]
                                                          for caption in captions])
    max_len = max_caption_length(df['caption'])
    df['caption'] = df['caption'].apply(lambda x: pad_sequences(x, max_len))
    pickle.dump(max_len, open("data\\max_len.pickle", "wb"))
    pickle.dump(word_dict, open("data\\word_dict.pickle", "wb"))
    pickle.dump(idx_to_word, open("data\\idx_to_word.pickle", "wb"))
    pickle.dump(word_to_idx, open("data\\word_to_idx.pickle", "wb"))
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_pickle('data\\targets.pickle')
    df.to_csv('data\\targets.csv')


def preprocess():
    preprocess_labels()
    preprocess_images()


if __name__ == '__main__':
    preprocess()


