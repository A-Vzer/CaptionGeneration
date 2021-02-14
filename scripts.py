from model import Model
from datasets import ImageCodes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

data_path = 'data\\imgCodes.pickle'
img_path = 'archive\\Images'
caption_path = 'data\\captions.pickle'
word_dict = pickle.load(open('data\\word_dict.pickle', 'rb'))
idx_to_word = pickle.load(open('data\\idx_to_word.pickle', 'rb'))
max_len = pickle.load(open('data\\max_len.pickle', 'rb'))
img_names = pickle.load(open('data\\imgNames.pickle', 'rb'))

train_dataset = ImageCodes(data_path)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
val_dataset = ImageCodes(data_path, split='val')
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_dataset = ImageCodes(data_path, split='test')
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
criteria = nn.CrossEntropyLoss()
model = Model(code_dim=20480, lstm_dim=500, num_words=len(idx_to_word), embed_dim=512).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100
val_loss_old = 99999
train_log = 10


def show():
    captions = pickle.load(open('data\\inference.pickle', 'rb'))
    names = pickle.load(open('data\\inferenceNames.pickle', 'rb'))
    for i in range(0,30):
        print([idx_to_word[x] for x in captions[i, ...]])
        im = Image.open(os.path.join(img_path, names[i]))
        plt.imshow(im)
        plt.show()


def train():
    model.train()
    train_losses = []
    for idx, (code, cap, img_names) in enumerate(train_loader):
        optimizer.zero_grad()
        code = code.cuda()
        cap = cap.cuda().long()
        out = model(code, cap[..., :-1]).permute(0,2,1)
        loss = criteria(out, cap)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if idx % train_log == 0:
            print(f"{round(idx/len(train_loader), 3)}%: {round(sum(train_losses) / (idx + 1), 3)}")
    return sum(train_losses) / len(train_loader)


def validate():
    model.eval()
    val_losses = []
    with torch.no_grad():
        for idx, (code, cap, img_names) in enumerate(val_loader):
            code = code.cuda()
            cap = cap.cuda().long()
            out = model(code, cap[..., :-1]).permute(0, 2, 1)
            loss = criteria(out, cap)
            val_losses.append(loss.item())
    return sum(val_losses) / len(val_loader)


def decode():
    model.eval()
    model.load_state_dict(torch.load('saves\\model.pt'))
    with torch.no_grad():
        for idx, (code, cap, img_name) in enumerate(test_loader):
            code = code.cuda()
            out = model.decode(code, max_len)
            if idx == 0:
                captions = out
                img_names = img_name
            else:
                captions = torch.cat((captions, out), dim=0)
                img_names += img_name
            print(f"{round(idx / len(test_loader), 3)}%")
    pickle.dump(captions.cpu().numpy(), open('data\\inference.pickle', 'wb'))
    pickle.dump(img_names, open('data\\inferenceNames.pickle', 'wb'))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    argParser = argparse.ArgumentParser(description="start/resume training")
    argParser.add_argument("-m", "--mode", dest="mode", action="store", default='t', type=str)
    cmd_args = argParser.parse_args()
    if not os.path.exists('saves'):
        os.makedirs('saves')
    if cmd_args.mode == 't':
        for epoch in range(epochs):
            train_loss = train()
            print(f"Train loss epoch {epoch+1}: {round(train_loss, 3)}")
            val_loss = validate()
            print(f"Validation loss epoch {epoch+1}: {round(val_loss, 3)}")
            torch.save(model.state_dict(), 'saves/model.pt')
            val_loss_old = val_loss
    elif cmd_args.mode == 'd':
        decode()
    elif cmd_args.mode == 's':
        show()
    elif cmd_args.mode == 'dev':
        print(len(idx_to_word), len(word_dict))

