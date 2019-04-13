import time, os, csv
import tensorflow as tf
import utils
import pickle,argparse
from model.QR_net_base import QR_net
from model.QR_net_BERT import QR_net_BERT
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
# add configs
parser = argparse.ArgumentParser()
parser.add_argument('--QR_weak_train_path',type=str, default="data/QR_weak/QR_weak_cell.csv")
parser.add_argument('--QR_weak_dev_path',type=str, default="data/QR_weak/QR_weak_dev.csv")
parser.add_argument('--QR_train_path',type=str, default="data/QR/QR_train.csv")
parser.add_argument('--QR_dev_path',type=str, default="data/QR/QR_dev_large.csv")
parser.add_argument('--QR_dev_dir',type=str, default="data/QR/dev_2/")
parser.add_argument('--restore_model',type=str, default='1551767099')
parser.add_argument('--use_BERT',type=bool, default=False)
parser.add_argument('--max_sequence_length',type=int, default=80)
parser.add_argument('--rnn_dim',type=int, default=32)
parser.add_argument('--batch_size',type=int, default=128)
parser.add_argument('--use_self_att',type=bool, default=True)
parser.add_argument('--threshold',type=float, default=0.5)
parser.add_argument('--finetune_flag',type=bool, default=True)
parser.add_argument('--preTraining_epochs',type=int, default=2)
parser.add_argument('--fineTune_epochs',type=int, default=55)
parser.add_argument('--QR_weak_train_slice',type=int, default=60000)
parser.add_argument('--QR_weak_dev_slice',type=int, default=5000)
parser.add_argument('--QR_train_slice',type=int, default=-1)
parser.add_argument('--mode',type=str, default='pretrain',help='pretrain, finetune')
args = parser.parse_args()

paths = {}
timestamp = str(int(time.time())); paths['timestamp'] = timestamp
embedding_path = "glove.twitter.27B.50d.txt"
word_dict_path = "word_dict.txt"
save_dir = 'saves/' + timestamp; paths['save_dir'] = save_dir; os.mkdir(save_dir)
log_dir = save_dir + '/logs'; paths['log_dir'] = log_dir; os.mkdir(log_dir)
preTrained_model = save_dir + '/model/preTrained_model.ckpt'; paths['preTrained_model'] = preTrained_model
config_file = open(save_dir + '/configs.txt', 'w')
restore_model_path = 'saves/' + args.restore_model + '/model/preTrained_model.ckpt'
paths['restore_model_path'] = restore_model_path

if args.mode == 'pretrain':
    print('preTraining_epos: {0}\n'
          'QR_weak_train_slice: {1}\n'
          'QR_train_slice: {2}\n'
          'pre_train_dataset: {3}\n'
          'use_selfAttenstion: {4}'
          .format(args.preTraining_epochs, args.QR_weak_train_slice, args.QR_train_slice,args.QR_weak_train_path,args.use_self_att)
          , file=config_file)

if args.mode == 'finetune':
    print('restore model id: {0}\n'.format(args.restore_model), file=config_file)
    print('Pretrain model info: \n' +
          open('saves/' + args.restore_model + '/configs.txt').read(), file=config_file)

config_file.close()

# Prepare Glove and Data
word_embedding_dim = 50  # set according to embedding_path
embedding_index, word_id, id_word = utils.read_word_embedding(embedding_path, word_embedding_dim, word_dict_path)
embedding_matrix = utils.pretrained_embedding_matrix(word_id, embedding_index, word_embedding_dim)
vocab_size = len(word_id) + 1

if args.use_BERT == False:
    if args.mode == 'pretrain':
        model = QR_net(args, embedding_matrix, word_id, paths)
        model.build_graph()
        model.pretrain()

    elif args.mode == 'finetune':
        model = QR_net(args, embedding_matrix, word_id, paths)
        model.build_graph()
        model.finetune()
else:
    if args.mode == 'pretrain':
        model = QR_net_BERT(args, embedding_matrix, word_id, paths)
        model.build_graph()
        model.pretrain()





