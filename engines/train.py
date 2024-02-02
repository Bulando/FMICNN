# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py
# @Software: PyCharm
import numpy as np
import time, datetime
import math
import tensorflow as tf
from tqdm import tqdm
from engines.utils.focal_loss import FocalLoss
from engines.utils.metrics import cal_metrics
from config import classifier_config

tf.keras.backend.set_floatx('float32')


def train(data_manager, logger, embed, ele):

    embedding_dim = data_manager.embedding_dim
    seq_length = data_manager.max_sequence_length

    num_classes = data_manager.max_label_number
    # 卷积参数的设置
    checkpoints_dir = classifier_config['checkpoints_dir']
    checkpoint_name = classifier_config['checkpoint_name']
    num_filters = classifier_config['num_filters']
    learning_rate = classifier_config['learning_rate']
    epoch = classifier_config['epoch']
    max_to_keep = classifier_config['max_to_keep']
    print_per_batch = classifier_config['print_per_batch']
    is_early_stop = classifier_config['is_early_stop']
    patient = classifier_config['patient']
    hidden_dim = classifier_config['hidden_dim']
    classifier = classifier_config['classifier']
    file_writer = tf.summary.create_file_writer(
        './logs' + '/' + (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))

    best_f1_val = 0.0
    best_at_epoch = 0
    unprocessed = 0
    batch_size = data_manager.batch_size
    very_start_time = time.time()
    loss_obj = FocalLoss() if classifier_config['use_focal_loss'] else None
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    step = 0
    # 改
    if embed == "word2vec":
        X_train, y_train, X_val, y_val = data_manager.get_training_set(ele=ele)
    elif embed == "sbert":
        X_train, y_train, X_val, y_val = data_manager.get_sentence_set(ele)
    elif embed == "multi-channel":
        # x_train, y_train = data_manager.read_eles_set(data_manager.train_file)
        # x_val, y_val = data_manager.read_eles_set(data_manager.dev_file)
        # num_samples = len(x_train)
        # data_manager.indices = np.arange(num_samples)
        # np.random.shuffle(data_manager.indices)
        # X_train_t, y_train_t, X_val_t, y_val_t = data_manager.get_training_set(embed="tfidf")
        # X_train_w, y_train_w, X_val_w, y_val_w = data_manager.get_training_set(embed="word2vec")
        # X_train_s, y_train_s, X_val_s, y_val_s = data_manager.get_training_set(embed="sbert")
        X_train_t = data_manager.load_npy("X_train_t.npy")
        y_train_t = data_manager.load_npy("y_train_t.npy")
        X_val_t = data_manager.load_npy("X_val_t.npy")
        y_val_t = data_manager.load_npy("y_val_t.npy")

        X_train_w = data_manager.load_npy("X_train_w.npy")
        y_train_w = data_manager.load_npy("y_train_w.npy")
        X_val_w = data_manager.load_npy("X_val_w.npy")
        y_val_w = data_manager.load_npy("y_val_w.npy")
        # x_t = list(X_train_w)  验证word2vec每个句子是否归一化
        # n = 0
        # print(x_t[0][0])
        # for x in x_t[0]:
        #     for x_1 in x:
        #         n+=x_1
        # print("word2vec是否是归一化？", n)
        X_train_s = data_manager.load_npy("X_train_s.npy")
        y_train_s = data_manager.load_npy("y_train_s.npy")
        X_val_s = data_manager.load_npy("X_val_s.npy")
        y_val_s = data_manager.load_npy("y_val_s.npy")

        # X_train_s, y_train_s, X_val_s, y_val_s = data_manager.load_npy(embed="sbert")
    # 载入模型
    if classifier == 'textcnn':
        from engines.models.textcnn import TextCNN
        model = TextCNN(seq_length, num_filters, num_classes, embedding_dim)
    elif classifier == 'textrcnn':
        from engines.models.textrcnn import TextRCNN
        model = TextRCNN(seq_length, num_classes, hidden_dim, embedding_dim)
    elif classifier == 'bi-lstm':
        from engines.models.bilstm import bilstm
        model = bilstm(num_classes=num_classes, embedding_dim=embedding_dim)
    else:
        raise Exception('config model is not exist')
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoints_dir, checkpoint_name=checkpoint_name, max_to_keep=max_to_keep)
    num_iterations = int(math.ceil(1.0 * len(X_train_w) / batch_size))
    num_val_iterations = int(math.ceil(1.0 * len(X_val_w) / batch_size))
    logger.info(('+' * 20) + 'training starting' + ('+' * 20))
    for i in range(epoch):
        start_time = time.time()
        # shuffle train at each epoch
        sh_index = np.arange(len(X_train_w))
        np.random.shuffle(sh_index)
        # X_train = X_train[sh_index]  # MBD
        # y_train = y_train[sh_index]  # MBD
        X_train_w = X_train_w[sh_index]
        y_train_w = y_train_w[sh_index]
        X_train_s = X_train_s[sh_index]
        y_train_s = y_train_s[sh_index]
        X_train_t = X_train_t[sh_index]
        y_train_t = y_train_t[sh_index]
        logger.info('epoch:{}/{}'.format(i + 1, epoch))
        for iteration in tqdm(range(num_iterations)):
            data_manager.indu = []
            step += 1
            # X_train_batch, y_train_batch = data_manager.next_batch(X_train, y_train, start_index=iteration * batch_size)
            X_train_batch_w, y_train_batch_w = data_manager.next_batch(X_train_w, y_train_w, start_index=iteration * batch_size)
            X_train_batch_s, y_train_batch_s = data_manager.next_batch(X_train_s, y_train_s, start_index=iteration * batch_size)
            X_train_batch_t, y_train_batch_t = data_manager.next_batch(X_train_t, y_train_t, start_index=iteration * batch_size)
            with tf.GradientTape() as tape:
                if classifier == 'textcnn':
                    logits = model.call([X_train_batch_w, X_train_batch_s], data_manager=data_manager, training=1)  # MBD
                elif classifier == 'textrcnn':
                    logits = model.call(X_train_batch_w,
                                        training=1)  # MBD
                elif classifier == 'bi-lstm':
                    logits = model.call(X_train_batch_w,
                                        training=1)  # MBD
                # predictions = tf.argmax(logits, axis=-1)
                if classifier_config['use_focal_loss']:
                    loss_vec = loss_obj.call(y_true=y_train_batch_w, y_pred=logits)
                else:
                    loss_vec = tf.keras.losses.categorical_crossentropy(y_true=y_train_batch_w, y_pred=logits)
                loss = tf.reduce_mean(loss_vec)
            # 定义好参加梯度的参数
            gradients = tape.gradient(loss, model.trainable_variables)
            # 反向传播，自动微分计算
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if iteration % print_per_batch == 0 and iteration != 0:
                predictions = tf.argmax(logits, axis=-1)
                y_train_batch_w = tf.argmax(y_train_batch_w, axis=-1)
                measures = cal_metrics(y_true=y_train_batch_w, y_pred=predictions)
                with file_writer.as_default():
                    tf.summary.scalar("loss", loss, step=step)
                res_str = ''
                for k, v in measures.items():
                    res_str += (k + ': %.3f ' % v)
                logger.info('training batch: %5d, loss: %.5f, %s' % (iteration, loss, res_str))

        # validation
        logger.info('start evaluate engines...')
        val_results = {'precision': 0, 'recall': 0, 'f1': 0}
        for iteration in tqdm(range(num_val_iterations)):
            data_manager.indu = []
            # X_val_batch, y_val_batch = data_manager.next_batch(X_val, y_val, start_index=iteration * batch_size)
            X_val_batch_w, y_val_batch_w = data_manager.next_batch(X_val_w, y_val_w, start_index=iteration * batch_size)
            X_val_batch_s, y_val_batch_s = data_manager.next_batch(X_val_s, y_val_s, start_index=iteration * batch_size)
            X_val_batch_t, y_val_batch_t = data_manager.next_batch(X_val_t, y_val_t, start_index=iteration * batch_size)
            if classifier == 'textcnn':
                logits = model.call([X_val_batch_w, X_val_batch_s], data_manager=data_manager)  # MBD
            elif classifier == 'textrcnn':
                logits = model.call(X_val_batch_w,
                                    training=1)  # MBD
            elif classifier == 'bi-lstm':
                logits = model.call(X_val_batch_w,
                                    training=1)  # MBD
            if np.array_equal(y_val_batch_w, y_val_batch_s):
                print("相等的！")
            else:
                index = np.arange(0, batch_size)
                # for i in range(batch_size):
                #     if y_val_batch_w[i] != y_val_batch_s[i]:
                #         print("快快快%d", i)
                for i, w in enumerate(y_val_batch_w.tolist()):
                    s = y_val_batch_s.tolist()[i]
                    if w != s:
                        print(i)
                        print(w)
                        print(s)
                # print(index[y_val_batch_s != y_val_batch_w])
                print("不等的！")
            predictions = tf.argmax(logits, axis=-1)
            ceshi = tf.argmax(y_val_batch_s, axis=-1)
            measures = cal_metrics(y_true=ceshi, y_pred=predictions)
            for k, v in measures.items():
                val_results[k] += v
        with file_writer.as_default():
            tf.summary.scalar('precision', val_results['precision']/num_val_iterations, step=step)
        time_span = (time.time() - start_time) / 60
        val_res_str = ''
        dev_f1_avg = 0
        for k, v in val_results.items():
            val_results[k] /= num_val_iterations
            val_res_str += (k + ': %.3f ' % val_results[k])
            if k == 'f1':
                dev_f1_avg = val_results[k]
        logger.info('time consumption:%.2f(min), %s' % (time_span, val_res_str))
        tf.summary.scalar('f1', np.array(dev_f1_avg).mean(), step=step)
        if np.array(dev_f1_avg).mean() > best_f1_val:
            unprocessed = 0
            best_f1_val = np.array(dev_f1_avg).mean()
            best_at_epoch = i + 1
            checkpoint_manager.save()
            logger.info('saved the new best model with f1: %.3f' % best_f1_val)
        else:
            unprocessed += 1

        if is_early_stop:
            if unprocessed >= patient:
                logger.info('early stopped, no progress obtained within {} epochs'.format(patient))
                logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
                logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
                return
    logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
    logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
