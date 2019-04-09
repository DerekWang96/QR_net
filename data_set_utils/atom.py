import  csv , random
QR_list = list(csv.reader(open('../data/QR/QR_dev.csv')))
positive = []
negative = []
for qr in QR_list:
    if qr[4] == '1':
        positive.append(qr)
    elif qr[4] == '0':
        negative.append(qr)


print(len(positive), len(negative))
# pos_len = len(positive)
# out_list = [positive + negative[i*pos_len*2:(i+1)*pos_len*2] for i in range(7)]
out = [positive[5] for i in range(150)]

# for i, o in enumerate(out_list):
#     random.shuffle(o)
#     with open('../data/QR/dev_3/' + str(i) + '.csv', 'a+') as f:
#         writer = csv.writer(f)
#         for line in o:
#             writer.writerow(line)

for i in range(7):
    with open('../data/QR/dev_3/' + str(i) + '.csv', 'a+') as f:
        writer = csv.writer(f)
        for i in out:
            writer.writerow(i)