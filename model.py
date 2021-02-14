import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, code_dim, lstm_dim, num_words, embed_dim):
        super().__init__()
        self.fc_code = nn.Linear(code_dim, embed_dim)
        self.embedding = nn.Embedding(num_words, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_dim)
        self.fc = nn.Linear(lstm_dim, num_words)

    def forward(self, code, caption):
        x = self.fc_code(code)
        y = self.embedding(caption)
        x = torch.unsqueeze(x, dim=1)
        z = torch.cat((x, y), dim=1)
        out, (c, h) = self.lstm(z)
        z = F.dropout(out)
        z = self.fc(z)
        return z

    def decode(self, code, max_len):
        x = self.fc_code(code)
        x = torch.unsqueeze(x, dim=1)
        out, (h, c) = self.lstm(x)
        z = F.dropout(out)
        z = self.fc(z)
        idx = torch.argmax(z, dim=-1)
        captions = idx
        for i in range(max_len):
            word = self.embedding(idx)
            out, (h, c) = self.lstm(word, (h, c))
            z = F.dropout(out)
            z = self.fc(z)
            idx = torch.argmax(z, dim=-1)
            captions = torch.cat((captions, idx), axis=1)
        return captions