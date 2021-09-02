import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt


embedding_dim = 50

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
