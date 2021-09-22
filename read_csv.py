import pandas as pd
import csv

with open('words_for_search.txt', 'r') as f:
    words = f.read().splitlines()
    print(words)

def findWords(tweet):
    for item in words:
        if item in tweet:
            return True
    return False

counter = 0

with open ('new_data.csv', 'w', encoding="utf8") as file_w:
    writer = csv.writer(file_w, lineterminator='\r')
    head = ['Data', 'Label']
    writer.writerow(head)
    with open('fake.csv', 'r', encoding="utf8") as file_r:
        reader = pd.read_csv(file_r)
        for row in reader.itertuples():
            tweet = row.Text
            if findWords(tweet):
                #if row.label == 1 or row.label==0 :
                 #   label = 'Fake'
                #else:
                 #   label = 'Real'
                tweet = tweet.replace('\n', '')
                new_row = tweet, 'Fake'
                writer.writerow(new_row)
                counter += 1

        print(counter)