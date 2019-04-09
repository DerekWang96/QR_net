import nltk
import random
import csv
from tqdm import tqdm

QR_list = list(csv.reader(open('data/QR/QR.csv')))

f = open('data/keywords_note_ele.txt', 'r')
keywords = []
for x in f:
    keywords.append(x.strip())
f.close()

key2Q_dict = {}
key2R_dict = {}
QA_list = []
R_list = []
cell_json = open('data/qa_Electronics.json', 'r')

len_qa = 0

for i, line in enumerate(cell_json):
    line = eval(line)
    QA_list.append(line)
    pro_ID = line['asin']
    len_qa = i
    for key in keywords:
        if key in line['question']:
            if key not in key2Q_dict.keys():
                key2Q_dict[key] = [(line['question'],pro_ID)]
            else:
                key2Q_dict[key].append((line['question'],pro_ID))

cell_json.close()
for entry in key2Q_dict.keys():
    print('keywords: {}, count: {}'.format(entry,str(len(key2Q_dict[entry]))))

len_r = 0
cell_r_json = open('data/Electronics_5.json', 'r')
for line in tqdm(cell_r_json):
    line = eval(line)
    R_list.append(line)
    sentences = nltk.sent_tokenize(line['reviewText'])  # to find review level match ,dont split this into sentences.
    pro_ID = line['asin']
    for key in keywords:
        for sent in sentences:
            if key in sent:
                if key not in key2R_dict.keys():
                    key2R_dict[key] = [(sent,pro_ID)]
                else:
                    key2R_dict[key].append((sent,pro_ID))
cell_r_json.close()
cell_json.close()
for entry in key2Q_dict.keys():
    print('keywords: {}, count: {}'.format(entry,str(len(key2R_dict[entry]))))

positive_pairs = []
for key in keywords:
    for q in key2Q_dict[key]:
        for r in key2R_dict[key]:
            if q[1] == r[1]:
                positive_pairs.append((q[0],r[0],q[1],1))

print(len(positive_pairs))
negative_pairs = []
for i in range(len(positive_pairs)):
    qa_pick = random.choice(QA_list)
    r_pick =random.choice(R_list)
    sents = nltk.sent_tokenize(r_pick['reviewText'])
    if len(sents) > 0:
        not_match = False
        r_sent_pick = random.choice(sents)
        for key in keywords:
            if key in qa_pick['question'] and key in r_sent_pick:
                i -= 1
                not_match = True
                continue
        if not_match:
            continue
    else:
        i -= 1
        continue
    q_sent_pick = qa_pick['question']
    pro_ID = qa_pick['asin']
    negative_pairs.append((q_sent_pick,r_sent_pick,pro_ID,0))

all_pairs = positive_pairs + negative_pairs

for i, item in enumerate(QR_list):
    for item_weak in all_pairs:
        if len(item) >= 4:
            if item[1].strip() == item_weak[1].strip() and item[2].strip() == item_weak[2].strip():
                print(i,' :',item)
                continue


random.shuffle(all_pairs)

with open('data/QR_weak/QR_weak_ele.csv', 'a') as f:
    writer_QR_weak = csv.writer(f)
    for pair in all_pairs:
        writer_QR_weak.writerow([pair[2],pair[0],pair[1],pair[3]])

        



