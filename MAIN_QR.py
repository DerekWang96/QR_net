import time, os, csv
import tensorflow as tf
import utils
from BIGRU import BiGRU
from ATTENTION import BiAttention, SelfAttention
from FUSION import NonLinearFusion, NonLinear_concat_Fusion
from CLASSIFICATION import BinaryClassificaion
from METRIC import mertic
import pickle

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

if mode == 'pretrain':
    print('preTraining_epos: {0}\n'
          'QR_weak_train_slice: {1}\n'
          'QR_train_slice: {2}\n'
          'pre_train_dataset: {3}\n'
          'use_selfAttenstion: {4}'
          .format(preTraining_epochs, QR_weak_train_slice, QR_train_slice,QA_weak_train_path,use_self_att)
          , file=config_file)

if mode == 'finetune':
    print('restore model id: {0}\n'.format(restore_model), file=config_file)
    print('Pretrain model info: \n' +
          open('saves/' + restore_model + '/configs.txt').read(), file=config_file)

config_file.close()

# Prepare Glove and Data
embedding_index, word_id, id_word = utils.read_word_embedding(embedding_path, word_embedding_dim, word_dict_path)
embedding_matrix = utils.pretrained_embedding_matrix(word_id, embedding_index, word_embedding_dim)
vocab_size = len(word_id) + 1

# Read weak QR training set
QR_weak_train = utils.generate_train_test(QA_weak_train_path, word_id, "QA")
QR_weak_q_train = QR_weak_train[0][:QR_weak_train_slice]
QR_weak_a_train = QR_weak_train[1][:QR_weak_train_slice]
QR_weak_y_train = QR_weak_train[2][:QR_weak_train_slice]

QR_weak_q_dev = QR_weak_train[0][QR_weak_train_slice: QR_weak_train_slice + QR_weak_dev_slice]
QR_weak_a_dev = QR_weak_train[1][QR_weak_train_slice: QR_weak_train_slice + QR_weak_dev_slice]
QR_weak_y_dev = QR_weak_train[2][QR_weak_train_slice: QR_weak_train_slice + QR_weak_dev_slice]

# Read QR training set and dev set
# QR_train = utils.generate_train_test(QR_train_path, word_id, "QR")
# QR_q_train = QR_train[0]
# QR_r_train = QR_train[1]
# QR_y_train = QR_train[2]
#
# QR_dev = utils.generate_train_test(QR_dev_path, word_id, "QR")
# QR_q_dev = QR_dev[0]
# QR_r_dev = QR_dev[1]
# QR_y_dev = QR_dev[2]

QR_train = utils.generate_train_test(QR_train_path, word_id, "QR")
QR_q_train = QR_train[0][:QR_train_slice]
QR_r_train = QR_train[1][:QR_train_slice]
QR_y_train = QR_train[2][:QR_train_slice]

QR_q_dev = QR_train[0][QR_train_slice:]
QR_r_dev = QR_train[1][QR_train_slice:]
QR_y_dev = QR_train[2][QR_train_slice:]

########Model#########
input_length = tf.fill([batch_size], max_sequence_length)
Q_ori = tf.placeholder(tf.int32, shape=[batch_size, max_sequence_length], name="question_ids")
R_ori = tf.placeholder(tf.int32, shape=[batch_size, max_sequence_length], name="review_ids")
labels = tf.placeholder(tf.float32, shape=[batch_size, 1], name="review_ids")

# Embedding
with tf.variable_scope("Embedding"):
    W = tf.Variable(tf.to_float(embedding_matrix), trainable=False, name="W_Glove")
    Q_embedding = tf.nn.embedding_lookup(W, Q_ori, name="Q_embedding_lookup")
    R_embedding = tf.nn.embedding_lookup(W, R_ori, name="R_embedding_lookup")

# BiGRU
Q_gru = BiGRU(Q_embedding, rnn_dim, input_length, batch_size, "Q_GRU")
R_gru = BiGRU(R_embedding, rnn_dim, input_length, batch_size, "R_GRU")

# Bi-Attention
with tf.variable_scope("BiAttention"):
    Q_att, R_att, r_Q, r_R = BiAttention(Q_gru, R_gru, rnn_dim, batch_size)

# Self-Attention
with tf.variable_scope("SelfAttention_Q"):
    Q_s_att, r_s_Q = SelfAttention(Q_gru, rnn_dim, batch_size, max_sequence_length, k)

with tf.variable_scope("SelfAttention_R"):
    R_s_att, r_s_R = SelfAttention(R_gru, rnn_dim, batch_size, max_sequence_length, k)

# concat and fusion
r_1 = tf.concat([r_Q, r_s_Q], 1)
r_2 = tf.concat([r_R, r_s_R], 1)
if use_self_att:
    u = NonLinear_concat_Fusion(r_1, r_2, rnn_dim, batch_size)
else:
    u = NonLinearFusion(r_Q, r_R, rnn_dim, batch_size)

# Aux-Task (QA-subnet)
with tf.variable_scope("QR_Task"):
    loss_QR, predict_QR = BinaryClassificaion(u, labels, rnn_dim, batch_size)
    # norm_1 = tf.norm(tf.matmul(Q_att, tf.transpose(R_att, [1, 0])), ord=2)
    # norm_2 = tf.norm([r_Q - r_R], ord="euclidean")
    all_loss = loss_QR  # regulazition term not added !
    tf.summary.histogram('predictions', predict_QR)
    tf.summary.scalar('cost', all_loss)

optimizer = tf.train.AdamOptimizer().minimize(all_loss)

saver = tf.train.Saver()
init = tf.initialize_all_variables()

merged = tf.summary.merge_all()

# Transfer training
if mode == 'pretrain':
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(init)
        step = 0
        csv_file = open(log_dir + '/pretrain_log.csv', "a+")
        pretrain_writer = csv.writer(csv_file)
        print("#########QR weak training begin.#########")
        for epo in range(preTraining_epochs):
            # train on QR weak pairs
            Q_batch, R_batch, Y_batch, last_start = utils.batch_pos_neg(QR_weak_q_train, QR_weak_a_train,
                                                                        QR_weak_y_train, batch_size, 0.5,
                                                                        True,True)
            prediction_weak = []
            y_weak_true = []
            for i in range(len(Y_batch)):
                _QA, Loss_aux, pred_weak, summary = sess.run([optimizer, all_loss, predict_QR, merged],
                                                             {Q_ori: Q_batch[i], R_ori: R_batch[i], labels: Y_batch[i]})

                y_weak_true += [pre[0] for pre in Y_batch[i]]
                pred_weak = [pre[0] for pre in pred_weak]
                prediction_weak += pred_weak
                train_writer.add_summary(summary, step)
                step += 0
                print("Pre-training epoch:", epo + 1, "--iter:", i + 1, "/", len(Y_batch), "--Loss:", Loss_aux)
            # uncomment this to see whether weights have been uodated.
            # precision_weak, recall_weak, f1_weak ,_ = mertic(prediction_weak, y_weak_true, threshold)
            # print("[Validate on QR_weak_train ]: --precision: ", precision_weak, "--recall: ", recall_weak, "--f1: ",
            #       f1_weak)
            # writer.writerow([epo, 'On QR_weak_train', precision_weak, recall_weak, f1_weak])

            # validation on QR_weak dev when training on QR_weak pairs
            Q_weak_dev_batch, R_weak_dev_batch, Y_weak_dev_batch, last_start = utils.batch_triplet_shuffle(
                QR_weak_q_dev, QR_weak_a_dev,
                QR_weak_y_dev, batch_size,
                True)
            prediction_weak_dev = []
            y_weak_dev_true = []
            for s in range(len(Y_weak_dev_batch)):
                Loss_dev, pred = sess.run([all_loss, predict_QR],
                                          {Q_ori: Q_weak_dev_batch[s], R_ori: R_weak_dev_batch[s],
                                           labels: Y_weak_dev_batch[s]})
                pred = [pre[0] for pre in pred]
                if s == len(Y_weak_dev_batch) - 1:
                    prediction_weak_dev += pred[last_start: batch_size]
                    y_weak_dev_true += [pre[0] for pre in Y_weak_dev_batch[s]][last_start: batch_size]
                else:
                    prediction_weak_dev += pred
                    y_weak_dev_true += [pre[0] for pre in Y_weak_dev_batch[s]]
            precision, recall, f1, _ = mertic(prediction_weak_dev, y_weak_dev_true, threshold)

            pretrain_writer.writerow([epo + 1, 'On QR_weak_dev', precision, recall, f1])
            print("[Validate on QR_weak dev]: --precision: ", precision, "--recall: ", recall, "--f1: ", f1)

            # validation on QR when training on QR_weak pairs
            Q_dev_batch, R_dev_batch, Y_dev_batch, last_start = utils.batch_triplet_shuffle(QR_q_dev, QR_r_dev, QR_y_dev,
                                                                                    batch_size,
                                                                                    True)
            prediction_dev_onWeak = []
            y_dev_true_onWeak = []
            for j in range(len(Y_dev_batch)):
                Loss_dev, pred = sess.run([all_loss, predict_QR],
                                          {Q_ori: Q_dev_batch[j], R_ori: R_dev_batch[j], labels: Y_dev_batch[j]})
                pred = [pre[0] for pre in pred]
                if j == len(Y_weak_dev_batch) - 1:
                    prediction_dev_onWeak += pred[last_start: batch_size]
                    y_dev_true_onWeak += [pre[0] for pre in Y_dev_batch][last_start: batch_size]
                else:
                    prediction_dev_onWeak += pred
                    y_dev_true_onWeak += [pre[0] for pre in Y_dev_batch]

            precision, recall, f1, d = mertic(prediction_dev_onWeak, y_dev_true_onWeak, threshold)
            print("[Validate on QR]: --precision: ", precision, "--recall: ", recall, "--f1: ", f1, "--d:", d)
            pretrain_writer.writerow([epo + 1, 'On QR_dev', precision, recall, f1])

        saver.save(sess, preTrained_model)
        print('Pretrained model saved on {}.'.format(timestamp))
        csv_file.close()
        # finetune epoches
        if finetune_flag:
            print("#########Finetune on QR pairs.#########")
            for epo in range(fineTune_epochs):
                # train on QR_train
                ratio = 0.5
                Q_train_batch, R_train_batch, Y_train_batch, (l_s_p, batch_slice) = utils.batch_pos_neg(QR_q_train,
                                                                                                        QR_r_train,
                                                                                                        QR_y_train,
                                                                                                        batch_size,
                                                                                                        ratio, True,True)
                prediction_QR_train = []
                y_train_true = []
                for j in range(len(Y_train_batch)):
                    _, Loss_QR_train, pred = sess.run([optimizer, all_loss, predict_QR],
                                                      {Q_ori: Q_train_batch[j], R_ori: R_train_batch[j],
                                                       labels: Y_train_batch[j]})
                    print("Pre-training epoch:", epo + 1, "--iter:", j + 1, "/", len(Y_train_batch), "--Loss:",
                          Loss_QR_train)

                    pred = [pre[0] for pre in pred]
                    y_train_true += [pre[0] for pre in Y_train_batch[j]]
                    prediction_QR_train += pred
                precision, recall, f1, d = mertic(prediction_QR_train, y_train_true, threshold)
                print("[Train on QR]: --precision: ", precision, "--recall: ", recall, "--f1: ", f1, "--d:", d)

                # un comment these if want to validate on QR weak set.
                # validation on QR_weak dev when training on QR_weak pairs
                # precision_num, recall_num, f1_num = 0, 0, 0
                # for i in range(3):
                #     Q_weak_dev_batch, R_weak_dev_batch, Y_weak_dev_batch, last_start = preprocess.batch_triplet_shuffle(
                #         QR_weak_q_dev, QR_weak_a_dev,
                #         QR_weak_y_dev, batch_size,
                #         False)
                #     prediction_weak_dev = []
                #     y_weak_dev_true = []
                #     for s in range(len(Y_weak_dev_batch)):
                #         Loss_dev, pred = sess.run([all_loss, predict_QR],
                #                                   {Q_ori: Q_weak_dev_batch[s], R_ori: R_weak_dev_batch[s],
                #                                    labels: Y_weak_dev_batch[s]})
                #         pred = [pre[0] for pre in pred]
                #         y_weak_dev_true += [pre[0] for pre in Y_weak_dev_batch[s]]
                #         prediction_weak_dev += pred
                #     precision, recall, f1, _ = mertic(prediction_weak_dev, y_weak_dev_true, threshold)
                #     precision_num += precision
                #     recall_num += recall
                #     f1_num += f1
                # finetune_writer.writerow([epo + 1, 'On QR_weak_dev', precision_num / 3, recall_num / 3, f1_num / 3])
                # print("[Validate on QR_weak dev]: --precision: ", precision_num / 3, "--recall: ", recall_num / 3, "--f1: ",
                #       f1_num / 3)

                # Validate on QR_dev
                precision_num, recall_num, f1_num = 0, 0, 0
                for i in range(7):
                    QR_dev = utils.generate_train_test(QR_dev_dir + str(i) + '.csv', word_id, "QR")
                    Q_dev_batch, R_dev_batch, Y_dev_batch, last_start = utils.batch_triplet_shuffle(QR_dev[0],
                                                                                                      QR_dev[1],
                                                                                                      QR_dev[2],
                                                                                                      batch_size,
                                                                                                      True)

                    prediction_dev = []
                    y_dev_true = []

                    for j in range(len(Y_dev_batch)):
                        Loss_dev, pred = sess.run([all_loss, predict_QR],
                                                  {Q_ori: Q_dev_batch[j], R_ori: R_dev_batch[j],
                                                   labels: Y_dev_batch[j]})

                        pred = [pre[0] for pre in pred]
                        if j == len(Y_dev_batch) - 1:
                            prediction_dev += pred[last_start: batch_size]
                            y_dev_true += [pre[0] for pre in Y_dev_batch[j]][last_start: batch_size]
                        else:
                            prediction_dev += pred
                            y_dev_true += [pre[0] for pre in Y_dev_batch[j]]
                    precision, recall, f1, d = mertic(prediction_dev, y_dev_true, threshold)
                    precision_num += precision
                    recall_num += recall
                    f1_num += f1
                print("[Validate on QR]: --precision: ", precision_num / 7, "--recall: ", recall_num / 7, "--f1: ",
                      f1_num / 7, "--d: ", d)

elif mode == 'finetune':
    csv_file = open(log_dir + '/finetune_log.csv', "a+")
    finetune_writer = csv.writer(csv_file)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(init)
        step = 0
        saver.restore(sess, restore_model_path)
        print('#########Fine tune with pre-trained weights.#########')
        for epo in range(fineTune_epochs):
            # train on QR_train
            ratio = 0.5
            Q_train_batch, R_train_batch, Y_train_batch, (l_s_p, batch_slice) = utils.batch_pos_neg(QR_q_train,
                                                                                                    QR_r_train,
                                                                                                    QR_y_train,
                                                                                                    batch_size,
                                                                                                    ratio, True,True)
            prediction_QR_train = []
            y_train_true = []
            for j in range(len(Y_train_batch)):
                _, Loss_QR_train, pred = sess.run([optimizer, all_loss, predict_QR],
                                                  {Q_ori: Q_train_batch[j], R_ori: R_train_batch[j],
                                                   labels: Y_train_batch[j]})
                print("Finetuning epoch:", epo + 1, "--iter:", j + 1, "/", len(Y_train_batch), "--Loss:",
                      Loss_QR_train)
                pred = [pre[0] for pre in pred]
                y_train_true += [pre[0] for pre in Y_train_batch[j]]
                prediction_QR_train += pred
            precision, recall, f1, d = mertic(prediction_QR_train, y_train_true, threshold)
            print("[Train on QR]: --precision: ", precision, "--recall: ", recall, "--f1: ", f1, "--d:", d)

            # un comment these if want to validate on QR weak set.
            # validation on QR_weak dev when training on QR_weak pairs
            # precision_num, recall_num, f1_num = 0, 0, 0
            # for i in range(3):
            #     Q_weak_dev_batch, R_weak_dev_batch, Y_weak_dev_batch, last_start = preprocess.batch_triplet_shuffle(
            #         QR_weak_q_dev, QR_weak_a_dev,
            #         QR_weak_y_dev, batch_size,
            #         False)
            #     prediction_weak_dev = []
            #     y_weak_dev_true = []
            #     for s in range(len(Y_weak_dev_batch)):
            #         Loss_dev, pred = sess.run([all_loss, predict_QR],
            #                                   {Q_ori: Q_weak_dev_batch[s], R_ori: R_weak_dev_batch[s],
            #                                    labels: Y_weak_dev_batch[s]})
            #         pred = [pre[0] for pre in pred]
            #         y_weak_dev_true += [pre[0] for pre in Y_weak_dev_batch[s]]
            #         prediction_weak_dev += pred
            #     precision, recall, f1, _ = mertic(prediction_weak_dev, y_weak_dev_true, threshold)
            #     precision_num += precision
            #     recall_num += recall
            #     f1_num += f1
            # finetune_writer.writerow([epo + 1, 'On QR_weak_dev', precision_num / 3, recall_num / 3, f1_num / 3])
            # print("[Validate on QR_weak dev]: --precision: ", precision_num / 3, "--recall: ", recall_num / 3, "--f1: ",
            #       f1_num / 3)
            precision_num, recall_num, f1_num = 0, 0, 0
            # Validate on QR_dev
            for i in range(7):
                QR_dev = utils.generate_train_test(QR_dev_dir + str(i) + '.csv' , word_id, "QR")

                Q_dev_batch, R_dev_batch, Y_dev_batch, last_start = utils.batch_triplet_shuffle(QR_dev[0],
                                                                                                  QR_dev[1],
                                                                                                  QR_dev[2],
                                                                                                  batch_size,
                                                                                                  True)
                # Q_dev_batch, R_dev_batch, Y_dev_batch, last_start = utils.batch_pos_neg(QR_dev[0],
                #                                                                                   QR_dev[1],
                #                                                                                   QR_dev[2],
                #                                                                                   batch_size,0.333333,
                #                                                                                   True,True)

                prediction_dev = []
                y_dev_true = []
                for j in range(len(Y_dev_batch)):

                    Loss_dev, pred = sess.run([all_loss, predict_QR],
                                              {Q_ori: Q_dev_batch[j], R_ori: R_dev_batch[j], labels: Y_dev_batch[j]})
                    pred = [pre[0] for pre in pred]
                    if epo == 16 and j == 0:
                        print(pred[:64])
                        print(pred[64:])
                    if j == len(Y_dev_batch) - 1:
                        # prediction_dev += pred[last_start: batch_size]
                        # y_dev_true += [pre[0] for pre in Y_dev_batch[j]][last_start: batch_size]
                        prediction_dev += pred
                        y_dev_true += [pre[0] for pre in Y_dev_batch[j]]
                    else:
                        prediction_dev += pred
                        y_dev_true += [pre[0] for pre in Y_dev_batch[j]]
                precision, recall, f1, d = mertic(prediction_dev, y_dev_true, threshold)
                precision_num += precision
                recall_num += recall
                f1_num += f1
            finetune_writer.writerow(
                [epo + 1, 'On QR_dev', round(precision_num / 7, 6), round(recall_num / 7, 6), round(f1_num / 7, 6)])
            print("[Validate on QR]: --precision: ", precision_num / 7, "--recall: ", recall_num / 7, "--f1: ",
                  f1_num / 7, "--d: ", d)

        # output the probability
        # Q_dev_batch, R_dev_batch, Y_dev_batch, (l_s_p, batch_slice) = utils.batch_pos_neg(QR_q_dev,
        #                                                                                   QR_r_dev,
        #                                                                                   QR_y_dev,
        #                                                                                   batch_size,
        #                                                                                   0.3, True)
        #
        # prediction_dev = []
        # y_dev_true = []
        # q_dev = []
        # r_dev = []
        # for j in range(len(Y_dev_batch)):
        #     Loss_dev, pred = sess.run([all_loss, predict_QR],
        #                               {Q_ori: Q_dev_batch[j], R_ori: R_dev_batch[j], labels: Y_dev_batch[j]})
        #     if j == len(Y_dev_batch) - 1:
        #         y_dev_true += [pre[0] for pre in Y_dev_batch[j]][l_s_p:]
        #         prediction_dev += [pre[0] for pre in pred][l_s_p:]
        #         q_dev += [list(pre) for pre in list(Q_dev_batch)[j]][l_s_p:]
        #         r_dev += [list(pre) for pre in list(R_dev_batch)[j]][l_s_p:]
        #     else:
        #         y_dev_true += [pre[0] for pre in Y_dev_batch[j]]
        #         prediction_dev += [pre[0] for pre in pred]
        #         q_dev += [list(pre) for pre in list(Q_dev_batch)[j]]
        #         r_dev += [list(pre) for pre in list(R_dev_batch)[j]]
        #
        #     for q, r, y, pre in zip(q_dev, r_dev, y_dev_true, prediction_dev):
        #         if y == 1 and pre > 0.5:
        #             print('FP preds, Q: {}, R: {}, sigmoid output: {}'.format(
        #                 utils.id_to_words(id_sequence=q, id_word=id_word),utils.id_to_words(id_sequence=r, id_word=id_word),pre))

    csv_file.close()
    print('Fintuning detail saved in {0}.'.format(timestamp))

elif mode == 'pred':
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(init)
        step = 0
        saver.restore(sess, restore_model_path)
        print('#########Fine tune with pre-trained weights.#########')
        for epo in range(fineTune_epochs):
            # train on QR_train
            ratio = 0.5
            Q_train_batch, R_train_batch, Y_train_batch, (l_s_p, batch_slice) = utils.batch_pos_neg(QR_q_train,
                                                                                                    QR_r_train,
                                                                                                    QR_y_train,
                                                                                                    batch_size,
                                                                                                    ratio, True)
            prediction_QR_train = []
            y_train_true = []
            for j in range(len(Y_train_batch)):
                _, Loss_QR_train, pred = sess.run([optimizer, all_loss, predict_QR],
                                                  {Q_ori: Q_train_batch[j], R_ori: R_train_batch[j],
                                                   labels: Y_train_batch[j]})
                print("Pre-training epoch:", epo + 1, "--iter:", j + 1, "/", len(Y_train_batch), "--Loss:",
                      Loss_QR_train)
                if j == len(Y_train_batch) - 1:
                    y_train_true += [pre[0] for pre in Y_train_batch[j]][l_s_p:batch_slice - 1] \
                                    + [pre[0] for pre in Y_train_batch[j]][
                                      - round((batch_slice - l_s_p) * ((1 - ratio) / ratio)):]
                    prediction_QR_train += [pre[0] for pre in pred][l_s_p:batch_slice - 1] \
                                           + [pre[0] for pre in pred][
                                             - round((batch_slice - l_s_p) * ((1 - ratio) / ratio)):]
                else:
                    y_train_true += [pre[0] for pre in Y_train_batch[j]]
                    prediction_QR_train += [pre[0] for pre in pred]
                # pred = [pre[0] for pre in pred]
                # y_train_true += [pre[0] for pre in Y_train_batch[j]]
                # prediction_QR_train += pred
            precision, recall, f1, d = mertic(prediction_QR_train, y_train_true, threshold)
            print("[Train on QR]: --precision: ", precision, "--recall: ", recall, "--f1: ", f1, "--d:", d)
        # Validate on QR_dev
        precision_num, recall_num, f1_num = 0, 0, 0
        ratio = 0.5
        for i in range(5):
            Q_dev_batch, R_dev_batch, Y_dev_batch, (l_s_p, batch_slice) = utils.batch_pos_neg(QR_q_dev,
                                                                                              QR_r_dev,
                                                                                              QR_y_dev,
                                                                                              batch_size,
                                                                                              ratio, True)

            prediction_dev = []
            y_dev_true = []

            for j in range(len(Y_dev_batch)):
                Loss_dev, pred = sess.run([all_loss, predict_QR],
                                          {Q_ori: Q_dev_batch[j], R_ori: R_dev_batch[j], labels: Y_dev_batch[j]})
                if j == len(Y_dev_batch) - 1:
                    y_dev_true += [pre[0] for pre in Y_dev_batch[j]][l_s_p:batch_slice - 1] + [pre[0] for pre in
                                                                                               Y_dev_batch[j]][
                                                                                              - round((
                                                                                                              batch_slice - l_s_p + 1) * (
                                                                                                              (
                                                                                                                      1 - ratio) / ratio)):]
                    prediction_dev += [pre[0] for pre in pred][l_s_p:batch_slice - 1] + [pre[0] for pre in pred][
                                                                                        - round((
                                                                                                        batch_slice - l_s_p + 1) * (
                                                                                                        (
                                                                                                                1 - ratio) / ratio)):]
                else:
                    y_dev_true += [pre[0] for pre in Y_dev_batch[j]]
                    prediction_dev += [pre[0] for pre in pred]
            precision, recall, f1, d = mertic(prediction_dev, y_dev_true, threshold)
            precision_num += precision
            recall_num += recall
            f1_num += f1
        print("[Validate on QR]: --precision: ", precision_num / 5, "--recall: ", recall_num / 5, "--f1: ",
              f1_num / 5, "--d: ", d)

        proID_2_QRs = pickle.load(open('data/singel product QR/Cell_proID_2_QRs.pkl','rb'))
        for i, key in enumerate(proID_2_QRs.keys()):
            if  i < 50:
                print('-------', key, '--------')
                Q_id_batch, R_id_batch, Y_batch, Q_batch, R_batch, last_start = utils.generate_batch_for_prediction(
                    proID_2_QRs, key, word_id, batch_size)
                Q_word = []
                R_word = []
                prediction = []
                for j in range(len(Y_batch)):
                    pred = sess.run([predict_QR], {Q_ori: Q_id_batch[j], R_ori: R_id_batch[j],
                                                   labels: Y_batch[j]})

                    prediction += [pre[0] for pre in pred]
                    Q_word += [pre for pre in Q_batch[j]]
                    R_word += [pre for pre in R_batch[j]]

                for i, p in enumerate(prediction):
                    if p > 0.5:
                        print(Q_word[i], ' ', R_word[i],' ',p)

