import re
from math import ceil
import matplotlib.pyplot as plt
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def stripToTextVector(text):
    # Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)

    # Remove Emails
    text = re.sub('\S*@\S*\s?', '', text)

    # Remove new line characters
    text = re.sub('\s+', ' ', text)

    # Remove distracting single quotes
    text = re.sub("\'", "", text)

    # Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])

    # Remove punctuations
    punctuations = ["?", ",", "“", ".", ";", ":", "!", '”', '"', "[", "]", "{", "}", "(", ")"]
    text = ''.join(u for u in text if u not in punctuations)

    # Replace slashes with spaces
    text = text.replace("/", " ")

    # Change string to lower letters
    text = text.lower()

    return text.split()


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


# Encode samples to numbers, and add target pred
def myEncode(sentences, labels, vocab):
    # pad each encoded sample till maxLen, that is roughly max len of all samples. and cut longer samples
    def pad(sample):
        while len(sample) < maxLen:
            sample.append('PAD')
        return sample[:maxLen]

    # determine max sentence length and pad
    maxLen = ceil(max([len(sample) for sample in sentences]) * 0.8)
    sentences = [pad(sentence) for sentence in sentences]

    # convert to numbers
    sentences = [[vocab[word] for word in sentence] for sentence in sentences]
    labels = [0 if label == "Real" else 1 for label in labels]

    # convert to tensors
    sentences = torch.tensor(sentences, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.float32, device=device)
    return sentences, labels


# plot 2 given lists of values
def plotMeasurement(measurement, trainMeasure, devMeasure, epochs):
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
