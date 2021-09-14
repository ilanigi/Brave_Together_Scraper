import json
import csv

with open('words_for_search.txt', 'r') as f:
    words = f.read().splitlines()
    #print(words)

def findWords(tweet):
    for i in words:
        if i in tweet:
            return True
    return False


with open('nazi_tweets.json', 'r',encoding="utf8") as file_r:
    data = json.load(file_r)

    counter = 0
    with open('nazi_tweets1.csv','w',encoding="utf8") as file_w:
        writer = csv.writer(file_w, lineterminator='\r')
        for item in data:
            if findWords(item['full_text']):
                tweet = item['full_text'].strip('\n').strip('\t')
                tweet = tweet.replace('\n','')
                row = tweet, 'Fake'
                writer.writerow(row)
                #counter += 1
        #print(counter)
