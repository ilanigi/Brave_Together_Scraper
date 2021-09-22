import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from AI_utils import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

emb_dim = 300
hid_dim = 100
batchSize = 1
lr = 0.01
epochs = 5


# The amazing neural network
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # maps each label to an embedding_dim vector
        self.embeddings = nn.Embedding(len(vocab), emb_dim).requires_grad_(True)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=1)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = torch.squeeze(x)
        emb = self.embeddings(x)  # (sampleLen, embDim)
        hn, _ = self.lstm(emb.view(len(emb), 1, -1))  # (sampleLen, batchSize=1, hidDim)
        out = self.fc(hn[-1][-1])  # (1,)
        return torch.sigmoid(out)


# does train, returns loss and accuracy
def train(list_of_tuples):
    model.train()
    trainLoader = DataLoader(list_of_tuples, batch_size=batchSize, shuffle=True)
    lossTotal, corrects = 0, 0
    for words, label in trainLoader:
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



if __name__ == '__main__':
    # read data and vocab from files
    data = pd.read_excel("data.xlsx")       # change to csv later
    sentences, labels = data.pop("sentences"), data.pop("labels")
    vocab = createVocab()

    # clean sentences, add word to vocab
    sentences = [stripToTextVector(sentence) for sentence in sentences]
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)


    # encode, split to train and dev
    sentences, labels = myEncode(sentences, labels, vocab)
    allData = [(sentence, label) for sentence, label in zip(sentences, labels)]  # [([list of words], label)]
    trainData, devData = train_test_split(allData)

    # model
    model = MyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.BCELoss()

    # do train and dev, save results
    losses_train, accuracies_train, losses_dev, accuracies_dev = [], [], [], []
    for epoch in range(epochs):
        print("epoch", epoch)
        accuracy_train, loss_train = train(trainData)
        accuracy_dev, loss_dev= dev(devData)
        losses_train.append(loss_train)
        accuracies_train.append(accuracy_train)
        losses_dev.append(loss_dev)
        accuracies_dev.append(accuracy_dev)

    # plot results
    plotMeasurement("Loss", losses_train, losses_dev, epochs)
    plotMeasurement("Accuracy", accuracies_train, accuracies_dev, epochs)

    torch.save(model, "modelFile")
