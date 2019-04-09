import csv
from tqdm import tqdm
data_set_path = '../data/QA/QA.csv'
data_set = open(data_set_path, 'r+')
QR_weak_list = list(csv.reader(data_set))

QR_list = list(csv.reader(open('../data/QR/QR.csv')))
items = []
for i, item_weak in tqdm(enumerate(QR_weak_list)):
    for item in QR_list:
        if len(item) >= 4:
            if item[1].strip().lower() == item_weak[1].strip().lower() and item[2].strip().lower() == item_weak[2].strip().lower():
                items.append(item)
                continue

print(items)
print(len(items))