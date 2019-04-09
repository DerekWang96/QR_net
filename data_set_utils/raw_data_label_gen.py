# generate new data for training
import nltk,pickle
cell_json = open('../data/QA_Cell_Phones_and_Accessories.json', 'r')

len_qa = 0
QA_list = []
proID_2_Qs = {}
proID_2_Rs = {}
for i, line in enumerate(cell_json):
    line = eval(line)
    pro_ID = line['asin']
    questions = line['questions']
    proID_2_Qs[pro_ID] = [q['questionText'] for q in questions]


cell_json.close()

cell_r_json = open('../data/Cell_Phones_and_Accessories_5.json', 'r')
for i, line in enumerate(cell_r_json):
    line = eval(line)
    sentences = nltk.sent_tokenize(
        line['reviewText'])  # to find review level match ,dont split this into sentences.
    pro_ID = line['asin']
    if pro_ID not in proID_2_Rs.keys():
        proID_2_Rs[pro_ID] = sentences
    else:
        proID_2_Rs[pro_ID] += sentences
cell_r_json.close()

proID_2_QRs = {}
for key in proID_2_Qs.keys():
    if key in proID_2_Rs.keys():
        proID_2_QRs[key] = [proID_2_Qs[key],proID_2_Rs[key]]

for k in proID_2_QRs.keys():
    print(k, proID_2_QRs[k])

pickle.dump(proID_2_QRs,file=open('../data/singel product QR/Cell_proID_2_QRs.pkl','wb'))


if __name__ == '__main__':
    proID_2_QRs = pickle.load(open('../data/singel product QR/Cell_proID_2_QRs.pkl', 'rb'))
    print(proID_2_QRs['B0037M8AEQ'])