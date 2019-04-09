import time, os, csv
import utils


os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

timestamp = str(int(time.time()))
# Data path
# QA_weak_train_path = "data/QR_weak/QR_weak_cell.csv"
# QA_weak_dev_path = "data/QR_weak/QR_weak_dev.csv"
QA_weak_train_path = "data/QR_weak/QR_weak_cell.csv"
QA_weak_dev_path = "data/QA/QA_dev.csv"

QR_train_path = "data/QR/QR.csv"
QR_dev_path = "data/QR/QR_dev_large.csv"
QR_dev_dir = "data/QR/dev_3/"

embedding_path = "glove.twitter.27B.50d.txt"
word_dict_path = "word_dict.txt"
save_dir = 'saves/' + timestamp
os.mkdir(save_dir)
log_dir = save_dir + '/logs'
os.mkdir(log_dir)
preTrained_model = save_dir + '/model/preTrained_model.ckpt'
config_file = open(save_dir + '/configs.txt', 'w')
restore_model = '1551607709'
restore_model_path = 'saves/' + restore_model + '/model/preTrained_model.ckpt'

# Model parameters
word_embedding_dim = 50  # set according to embedding_path
max_sequence_length = 80
learning_rate = 0.001
rnn_dim = 32
k = 32
batch_size = 128
preTraining_epochs = 10  # pre-training epochs: on QR weak pairs based on keywords
fineTune_epochs = 100  # fine-tune iterations: on QR pairs
threshold = 0.5  # obtain a binary result from prediction result(continuous values), default=0.5
Mu = 0.1
Eta = 0.1
QR_weak_train_slice = 60000  # use batch_size * slice pairs
QR_weak_dev_slice = 5000
finetune_flag = True
QR_train_slice = 5000
use_self_att = True
mode = 'finetune'  # pretrain, finetune, pred

embedding_index, word_id, id_word = utils.read_word_embedding(embedding_path, word_embedding_dim, word_dict_path)
embedding_matrix = utils.pretrained_embedding_matrix(word_id, embedding_index, word_embedding_dim)
vocab_size = len(word_id) + 1

QR_dev = utils.generate_train_test('data/QR/dev_3/0.csv', word_id, "QR")
print(QR_dev[2])
Q_dev_batch, R_dev_batch, Y_dev_batch, last_start = utils.batch_triplet_shuffle(QR_dev[0],
                                                                        QR_dev[1],
                                                                        QR_dev[2],
                                                                        batch_size,
                                                                        True)
print(Y_dev_batch[0])


