import pandas as pd
import csv

with open('new_data.csv','r', encoding="utf8") as file_r:
    with open('fakePosts.csv', 'w', encoding="utf8") as fake:
        writer_fake = csv.writer(fake, lineterminator='\r')
        head = ['Data', 'Label']
        writer_fake.writerow(head)
        with open('realPosts.csv','w', encoding="utf8") as real:
            writer_real = csv.writer(real,lineterminator='\r')
            writer_real.writerow(head)
            reader = pd.read_csv(file_r)
            for row in reader.itertuples():
                if row.Label=='Fake':
                    writer_fake.writerow(row)
                else:
                    writer_real.writerow(row)