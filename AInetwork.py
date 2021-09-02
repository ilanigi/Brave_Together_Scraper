import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt


embedding_dim = 50


class MyBilstm(nn.Module):
    def __init__(self):
        super(MyBilstm, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(self.createEmb(), norm_type=2, max_norm=2).requires_grad_(True)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=2, bidirectional=True)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(hid_dim * 2, len(vocabL.values()))

    def forward(self, x):  # (sampleLen,)
        emb = self.embedding(x)  # (sampleLen, embDim)
        hn, _ = self.lstm(emb.view(len(x), 1, -1))  # (sampleLen, batchSize, hidDim*2)
        hn = self.dropout(hn)
        hn = torch.squeeze(hn)  # (sampleLen, hidDim*2)
        out = self.fc(hn)  # (len(vocabL.values()), )
        if len(x) == 1:  # the output must has 2 dimensions for loss calculation
            out = out.reshape(1, -1)
        return out
    
    
# The neural network
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # maps each label to an embedding_dim vector
        self.embeddings = nn.Embedding(len(vocabW), embedding_dim).requires_grad_(True)



    def forward(self, x):                                 # (batchSize, winSize)
        x = self.embeddings(x)                            # (batchSize, winSize, embedSize)

        return x


# Trains our model
def train(fiveGrams):
    model.train()
    trainLoader = DataLoader(fiveGrams, batch_size=batchSize, shuffle=True)
    for contextWords, midLabel in trainLoader:
        optimizer.zero_grad()
        output = model(contextWords)
        loss = loss_f(output, midLabel)
        loss.backward()
        optimizer.step()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trainData = 1               # ([list of words], label)
    vocabW = 1                  # {word : serial number}
    model = MyModel()
    optimizer =
