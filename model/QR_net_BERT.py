import tensorflow as tf
import time, os, csv
import utils
from METRIC import mertic
class QR_net_BERT():
    def __init__(self,args,embedding_matrix,word_id_dict,paths):
        self.embedding_matrix = embedding_matrix
        self.batch_size = args.batch_size
        self.max_sequence_length = args.max_sequence_length
        self.BERT_feature_dim = 384  # 768/2
        self.use_BERT = args.use_BERT
        self.use_self_att = args.use_self_att
        self.threshold = args.threshold
        self.word_id = word_id_dict
        self.finetune_flag = args.finetune_flag
        self.preTraining_epochs = args.preTraining_epochs
        self.fineTune_epochs = args.fineTune_epochs
        self.QR_weak_train_slice =args.QR_weak_train_slice
        self.QR_weak_dev_slice = args.QR_weak_dev_slice
        self.QR_train_slice = args.QR_train_slice
        self.QR_weak_dev_path =args.QR_weak_dev_path
        self.QR_weak_train_path = args.QR_weak_train_path
        self.QR_train_path = args.QR_train_path
        self.QR_dev_dir = args.QR_dev_dir
        self.preTrained_model = paths['preTrained_model']
        self.timestamp = paths['timestamp']
        self.restore_model_path = paths['restore_model_path']
        self.log_dir = paths['log_dir']


    def prepare_features(self):
        self.pretrain_train_features = utils.generate_features(self.QR_weak_train_path,'QR_weak',self.QR_weak_train_slice)
        self.pretrain_dev_features = utils.generate_features(self.QR_weak_dev_path,'QR_weak',self.QR_weak_dev_slice)
        self.finetune_train_features = utils.generate_features(self.QR_train_path,'QR')
        self.finetune_devs_features = [utils.generate_features(self.QR_dev_dir + str(i) + '.csv', 'QR')  for i in range(7)]

    def build_graph(self):
        self.prepare_features()
        self.add_placeholders()
        self.add_attentions_n_fusion()
        self.add_output_layer()
        self.add_summary()
        self.add_optimizer()
        self.init_op()

    def add_placeholders(self):
        self.Q_ori = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_sequence_length], name="question_ids")
        self.R_ori = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_sequence_length], name="review_ids")
        self.labels = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name="review_ids")

    def add_GRUs(self):
        input_length = tf.fill([self.batch_size], self.max_sequence_length)
        with tf.variable_scope('Q_GRU'):
            fw_cell = tf.contrib.rnn.GRUCell(self.BERT_feature_dim)
            bw_cell = tf.contrib.rnn.GRUCell(self.BERT_feature_dim)
            initial_state_fw = fw_cell.zero_state(self.batch_size, dtype=tf.float32)
            initial_state_bw = bw_cell.zero_state(self.batch_size, dtype=tf.float32)
            (outputs, states) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.Q_embedding, input_length,
                                                                initial_state_fw, initial_state_bw)
            self.Q_gru_out = tf.concat(outputs, 2)
        with tf.variable_scope('R_GRU'):
            fw_cell = tf.contrib.rnn.GRUCell(self.BERT_feature_dim)
            bw_cell = tf.contrib.rnn.GRUCell(self.BERT_feature_dim)
            initial_state_fw = fw_cell.zero_state(self.batch_size, dtype=tf.float32)
            initial_state_bw = bw_cell.zero_state(self.batch_size, dtype=tf.float32)
            (outputs, states) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.R_embedding, input_length,
                                                                initial_state_fw, initial_state_bw)
            self.R_gru_out = tf.concat(outputs, 2)

    def add_attentions_n_fusion(self):
        with tf.variable_scope('Bi-Attention'):
            U = tf.get_variable("U", [2 * self.BERT_feature_dim, 2 * self.BERT_feature_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            G = tf.matmul(tf.matmul(self.Q_ori, tf.tile(tf.expand_dims(U, 0), [self.batch_size, 1, 1])),
                          self.R_ori, adjoint_b=True)
            self.Q_att = tf.nn.softmax(tf.reduce_max(G, 2))
            self.R_att = tf.nn.softmax(tf.reduce_max(G, 1))
            self.r_Q = tf.matmul(tf.transpose(self.Q_ori, [0, 2, 1]), tf.expand_dims(self.Q_att, 2))
            self.r_R = tf.matmul(tf.transpose(self.R_ori, [0, 2, 1]), tf.expand_dims(self.R_att, 2))

        if self.use_self_att:
            with tf.variable_scope("SelfAttention_Q"):
                k = 32
                U_q = tf.get_variable("U_q", [2 * self.BERT_feature_dim, k], initializer=tf.truncated_normal_initializer(stddev=0.1))
                V_q = tf.get_variable("V_q", [k, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
                U_q_expand = tf.tile(tf.expand_dims(U_q, 0), [self.batch_size, 1, 1])
                V_q_expand = tf.tile(tf.expand_dims(V_q, 0), [self.batch_size, 1, 1])
                UH = tf.tanh(tf.matmul(self.Q_ori, U_q_expand))
                self.Q_s_att = tf.nn.softmax(tf.reshape(tf.matmul(UH, V_q_expand), [self.batch_size, self.max_sequence_length]), axis=1)
                self.r_s_Q = tf.matmul(tf.transpose(self.Q_ori, [0, 2, 1]), tf.expand_dims(self.Q_s_att, 2))

            with tf.variable_scope("SelfAttention_R"):
                k = 32
                U_r = tf.get_variable("U_r", [2 * self.BERT_feature_dim, k], initializer=tf.truncated_normal_initializer(stddev=0.1))
                V_r = tf.get_variable("V_r", [k, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
                U_r_expand = tf.tile(tf.expand_dims(U_r, 0), [self.batch_size, 1, 1])
                V_r_expand = tf.tile(tf.expand_dims(V_r, 0), [self.batch_size, 1, 1])
                UH = tf.tanh(tf.matmul(self.R_ori, U_r_expand))
                self.R_s_att = tf.nn.softmax(tf.reshape(tf.matmul(UH, V_r_expand), [self.batch_size, self.max_sequence_length]), axis=1)
                self.r_s_R = tf.matmul(tf.transpose(self.R_ori, [0, 2, 1]), tf.expand_dims(self.R_s_att, 2))
            # concat att outputs
            self.r_1 = tf.concat([self.r_Q, self.r_s_Q], 1)
            self.r_2 = tf.concat([self.r_R, self.r_s_R], 1)
            # fusion
            with tf.variable_scope("Concat_Fusion", reuse=tf.AUTO_REUSE):
                w1 = tf.get_variable("w1", [2 * self.BERT_feature_dim, 4 * self.BERT_feature_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
                w2 = tf.get_variable("w2", [2 * self.BERT_feature_dim, 4 * self.BERT_feature_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.uQR = tf.nn.tanh(
                    tf.matmul(tf.tile(tf.expand_dims(w1, 0), [self.batch_size, 1, 1]), self.r_1)
                    + tf.matmul(tf.tile(tf.expand_dims(w2, 0), [self.batch_size, 1, 1]), self.r_2))

        else:
            with tf.variable_scope("Fusion", reuse=tf.AUTO_REUSE):
                wQ = tf.get_variable("w1", [2 * self.BERT_feature_dim, 2 * self.BERT_feature_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
                wR = tf.get_variable("w2", [2 * self.BERT_feature_dim, 2 * self.BERT_feature_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.uQR = tf.nn.tanh(
                    tf.matmul(tf.tile(tf.expand_dims(wQ, 0), [self.batch_size, 1, 1]), self.r_Q)
                    + tf.matmul(tf.tile(tf.expand_dims(wR, 0), [self.batch_size, 1, 1]), self.r_R))

    def add_output_layer(self):
        with tf.variable_scope("Classification", reuse=tf.AUTO_REUSE):
            w = tf.get_variable("w", [2 * self.BERT_feature_dim, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable("b", [self.batch_size, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
            u = tf.reshape(self.uQR, [self.batch_size, 2 * self.BERT_feature_dim])
            wu_b = (tf.matmul(u, w) + b)
            self.predict = tf.nn.sigmoid(wu_b)
            self.loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=wu_b, name="binary_classification"))

    def add_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def add_summary(self):
        tf.summary.histogram('predictions', self.predict)
        tf.summary.scalar('cost', self.loss)
        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def pretrain(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            sess.run(self.init_op)
            step = 0
            csv_file = open(self.log_dir + '/pretrain_log.csv', "a+")
            pretrain_writer = csv.writer(csv_file)
            QR_weak_train = utils.generate_train_test(self.QR_weak_train_path, self.word_id, "QA")
            print("#########QR weak training begin.#########")
            for epo in range(self.preTraining_epochs):
                # train on QR weak pairs
                Q_batch, R_batch, Y_batch, last_start = utils.batch_pos_neg(QR_weak_train[0][:self.QR_weak_train_slice], QR_weak_train[1][:self.QR_weak_train_slice],
                                                                            QR_weak_train[2][:self.QR_weak_train_slice], self.batch_size, 0.5,
                                                                            True, True)
                prediction_weak = []
                y_weak_true = []
                for i in range(len(Y_batch)):
                    _QR, Loss_aux, pred_weak, summary = sess.run([self.optimizer, self.loss, self.predict, self.merged],
                                                                 {self.Q_ori: Q_batch[i], self.R_ori: R_batch[i],
                                                                  self.labels: Y_batch[i]})

                    y_weak_true += [pre[0] for pre in Y_batch[i]]
                    pred_weak = [pre[0] for pre in pred_weak]
                    prediction_weak += pred_weak
                    train_writer.add_summary(summary, step)
                    step += 0
                    print("Pre-training epoch:", epo + 1, "--iter:", i + 1, "/", len(Y_batch), "--Loss:", Loss_aux)
                self.validate(sess,pretrain_writer,True,epo)
                self.validate(sess,pretrain_writer,False,epo)
                # uncomment this to see whether weights have been updated.
                # precision_weak, recall_weak, f1_weak ,_ = mertic(prediction_weak, y_weak_true, threshold)
                # print("[Validate on QR_weak_train ]: --precision: ", precision_weak, "--recall: ", recall_weak, "--f1: ",
                #       f1_weak)
                # writer.writerow([epo, 'On QR_weak_train', precision_weak, recall_weak, f1_weak])

                # validation on QR_weak dev when training on QR_weak pairs


            saver.save(sess, self.preTrained_model)
            print('Pretrained model saved on {}.'.format(self.timestamp))
            csv_file.close()

            # finetune epoches
            if self.finetune_flag:
                print("#########Finetune on QR pairs.#########")
                QR_train = utils.generate_train_test(self.QR_train_path, self.word_id, "QR")
                for epo in range(self.fineTune_epochs):
                    # train on QR_train
                    ratio = 0.5
                    Q_train_batch, R_train_batch, Y_train_batch, (l_s_p, batch_slice) = utils.batch_pos_neg(QR_train[0][:self.QR_train_slice],
                                                                                                            QR_train[1][:self.QR_train_slice],
                                                                                                            QR_train[2][:self.QR_train_slice],
                                                                                                            self.batch_size,
                                                                                                            ratio, True,
                                                                                                            True)
                    prediction_QR_train = []
                    y_train_true = []
                    for j in range(len(Y_train_batch)):
                        _, Loss_QR_train, pred = sess.run([self.optimizer, self.loss, self.predict],
                                                          {self.Q_ori: Q_train_batch[j], self.R_ori: R_train_batch[j],
                                                           self.labels: Y_train_batch[j]})
                        print("Pre-training epoch:", epo + 1, "--iter:", j + 1, "/", len(Y_train_batch), "--Loss:",
                              Loss_QR_train)

                        pred = [pre[0] for pre in pred]
                        y_train_true += [pre[0] for pre in Y_train_batch[j]]
                        prediction_QR_train += pred
                    precision, recall, f1, d = mertic(prediction_QR_train, y_train_true, self.threshold)
                    print("[Train on QR]: --precision: ", precision, "--recall: ", recall, "--f1: ", f1, "--d:", d)
                    # self.validate(sess, None,True,epo)
                    self.validate(sess, None,False,epo)


    def finetune(self):
        csv_file = open(self.log_dir + '/finetune_log.csv', "a+")
        finetune_writer = csv.writer(csv_file)
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            sess.run(self.init_op)
            step = 0
            self.saver.restore(sess, self.restore_model_path)
            print('#########Fine tune with pre-trained weights.#########')
            QR_train = utils.generate_train_test(self.QR_train_path, self.word_id, "QR")
            for epo in range(self.fineTune_epochs):
                # train on QR_train
                ratio = 0.5
                Q_train_batch, R_train_batch, Y_train_batch, (l_s_p, batch_slice) = utils.batch_pos_neg(
                    QR_train[0][:self.QR_train_slice],
                    QR_train[1][:self.QR_train_slice],
                    QR_train[2][:self.QR_train_slice],
                    self.batch_size,
                    ratio, True,
                    True)
                prediction_QR_train = []
                y_train_true = []
                for j in range(len(Y_train_batch)):
                    _, Loss_QR_train, pred = sess.run([self.optimizer, self.loss, self.predict],
                                                      {self.Q_ori: Q_train_batch[j], self.R_ori: R_train_batch[j],
                                                       self.labels: Y_train_batch[j]})
                    print("Finetuning epoch:", epo + 1, "--iter:", j + 1, "/", len(Y_train_batch), "--Loss:",
                          Loss_QR_train)
                    pred = [pre[0] for pre in pred]
                    y_train_true += [pre[0] for pre in Y_train_batch[j]]
                    prediction_QR_train += pred
                precision, recall, f1, d = mertic(prediction_QR_train, y_train_true, self.threshold)
                print("[Train on QR]: --precision: ", precision, "--recall: ", recall, "--f1: ", f1, "--d:", d)
                # self.validate(sess, finetune_writer, True, epo)
                self.validate(sess, finetune_writer, False, epo)

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
        print('Fintuning detail saved in {0}.'.format(self.timestamp))

    def validate(self,sess,writer,on_QR_weak,epo):
        """
        can only be called by pretrain and finetune
        """
        if on_QR_weak:
            times = 1
        else:
            times = 7
        precision_num, recall_num, f1_num = 0, 0, 0
        for i in range(times):
            if on_QR_weak:
                QR_dev = self.pretrain_dev_features
            else:
                QR_dev = self.finetune_devs_features[i]

            Q_dev_batch, R_dev_batch, Y_dev_batch, last_start = utils.batch_triplet_shuffle(
                QR_dev[0], QR_dev[1],
                QR_dev[2], self.batch_size,
                True)
            prediction_weak_dev = []
            y_weak_dev_true = []

            for s in range(len(Y_dev_batch)):
                Loss_dev, pred = sess.run([self.loss, self.predict],
                                          {self.Q_ori: Q_dev_batch[s], self.R_ori: R_dev_batch[s],
                                           self.labels: Y_dev_batch[s]})
                pred = [pre[0] for pre in pred]
                if s == len(Y_dev_batch) - 1:
                    prediction_weak_dev += pred[last_start: self.batch_size]
                    y_weak_dev_true += [pre[0] for pre in Y_dev_batch[s]][last_start: self.batch_size]
                else:
                    prediction_weak_dev += pred
                    y_weak_dev_true += [pre[0] for pre in Y_dev_batch[s]]
            precision, recall, f1, d = mertic(prediction_weak_dev, y_weak_dev_true, self.threshold)
            precision_num += precision
            recall_num += recall
            f1_num += f1
        if on_QR_weak:
            if writer is not None :
                writer.writerow(
                    [epo + 1, 'On QR_weak_dev', round(precision_num / times, 6), round(recall_num / times, 6),
                     round(f1_num / times, 6)])
            print("[Validate on QR_weak dev]: --precision: ", round(precision_num/times,6), "--recall: ", round(recall_num/times,6), "--f1: ", round(f1_num/times,6))
        else:
            if writer is not None :
                writer.writerow([epo + 1, 'On QR_dev', round(precision_num / times, 6), round(recall_num / times, 6),
                                 f1_num / times])
            print("[Validate on QR dev]: --precision: ", round(precision_num/times,6), "--recall: ", round(recall_num/times,6), "--f1: ", round(f1_num/times,6))



