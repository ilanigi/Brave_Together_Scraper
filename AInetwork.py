import numpy as np
import sys
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import pandas as pd

emb_dim = 50
hid_dim = 100
batchSize = 1
lr = 0.01
epochs = 10


# The amazing neural network
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # maps each label to an embedding_dim vector
        self.embeddings = nn.Embedding(len(vocabW), emb_dim).requires_grad_(True)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=3)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)  # (sampleLen, embDim)
        hn, _ = self.lstm(emb.view(len(x), 1, hid_dim))  # (sampleLen, batchSize=1, hidDim)
        out = self.fc(hn[-1][-1])  # (1,)
        return torch.sigmoid(out)


# Trains our model
def train(fiveGrams):
    model.train()
    trainLoader = DataLoader(fiveGrams, batch_size=batchSize, shuffle=True)
    for contextWords, midLabel in trainLoader:
        optimizer.zero_grad()
        output = model(words)
        loss = loss_f(output, label)
        loss.backward()
        optimizer.step()
        lossTotal += loss.item()
        if torch.round(output.data[0]) == label.data[0]:
            corrects += 1
    accuracy = corrects / len(list_of_tuples)
    lossTotal /= len(list_of_tuples)
    return accuracy, lossTotal


# does dev, returns loss and accuracy
def dev(list_of_tuples):
    model.eval()
    devLoader = DataLoader(list_of_tuples, batch_size=batchSize, shuffle=True)
    lossTotal, corrects = 0, 0
    with torch.no_grad():
        for sample, target in devLoader:
            output = model(sample)
            loss = loss_f(output, target)
            lossTotal += loss.item()
            if torch.round(output.data[0]) == target.data[0]:
                corrects += 1
    accuracy = corrects / len(list_of_tuples)
    lossTotal /= len(list_of_tuples)
    return accuracy, lossTotal


# plot 2 given lists of values
def plotMeasurement(measurement, trainMeasure, devMeasure):
    epochsList = [i for i in range(epochs)]
    plt.figure()
    plt.title(measurement)
    plt.plot(epochsList, trainMeasure, label="Train")
    plt.plot(epochsList, devMeasure, label="Dev")
    plt.xlabel("Epochs")
    plt.ylabel(measurement)
    plt.locator_params(axis="x", integer=True, tight=True)  # make x axis to display only whole number (iterations)
    plt.legend()
    plt.savefig(measurement)


# Create vocabulary dict of words or labels
def createVocab():
    dict = {}
    i = 0

    # add words from file
    with open("vocab.txt", "r") as f:
        for word in f:
            word = word.strip()
            dict[word] = i
            i += 1

    # add special tokens
    dict['PAD_BEGIN'] = len(dict)
    dict['PAD_END'] = len(dict)

    return dict


if __name__ == '__main__':
    # read data and vocab from files
    data = pd.read_excel("data.xlsx")
    sentences, labels = data.pop("sentences"), data.pop("labels")
    vocab = createVocab()
    # convert to numbers
    labels = [0 if label == "Real" else 1 for label in labels]
    sentences = [sentence.split(' ') for sentence in sentences]
    sentences = [[vocab[word] for word in sentence] for sentence in sentences]


    trainData = 1  # [([list of words], label)]
    devData = 1
    vocabW = 1  # {word : serial number}

    # model
    model = MyModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.BCELoss()

    # do train and dev, save results
    losses_train, accuracies_train, losses_dev, accuracies_dev = [], [], [], []
    for epoch in range(epochs):
        print("epoch", epoch)
        loss_train, accuracy_train = train(trainData)
        loss_dev, accuracy_dev = dev(devData)
        losses_train.append(loss_train)
        accuracies_train.append(loss_train)
        losses_dev.append(loss_dev)
        accuracies_dev.append(accuracy_dev)

    # plot results
    plotMeasurement("Loss", losses_train, losses_dev)
    plotMeasurement("Accuracy", accuracies_train, accuracies_dev)