import csv

data_set_path = 'data/QR_weak/QR_weak.csv'
data_set = open(data_set_path, 'r+')
data_set_list = list(csv.reader(data_set))
data_set_train = open(data_set_path[:-4] + '_train' + data_set_path[-4:], 'a')
data_set_dev = open(data_set_path[:-4] + '_dev' + data_set_path[-4:], 'a')
data_set_test = open(data_set_path[:-4] + '_test' + data_set_path[-4:], 'a')

count = 0
writer_train = csv.writer(data_set_train)
writer_dev = csv.writer(data_set_dev)
writer_test = csv.writer(data_set_test)

for i, line in enumerate(data_set_list):
    if i < int(len(data_set_list)*0.6):
        writer_train.writerow(line)
    elif int(len(data_set_list)*0.6) <= i < int(len(data_set_list)*0.8):
        writer_dev.writerow(line)
    else:
        writer_test.writerow(line)

data_set.close()
data_set_train.close()
data_set_dev.close()
data_set_test.close()