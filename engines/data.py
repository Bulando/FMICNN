# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from engines.utils.word2vec import Word2VecUtils
from config import classifier_config
from sentencebert import get_sentence_embedding
from sklearn.preprocessing import OneHotEncoder
from engines.utils.tfidf import handler_doc
from engines.utils.logger import get_logger
import json


class DataManager:
    def __init__(self, logger, embed):
        self.logger = logger
        if embed == "word2vec":
            # 这里也有问题
            self.train_file = classifier_config['train_file']
            self.dev_file = classifier_config['dev_file']
            self.w2v_util = Word2VecUtils(logger)
            self.w2v_model = Word2Vec.load(self.w2v_util.model_path)
            self.stop_words = self.w2v_util.get_stop_words()
            self.max_sequence_length = classifier_config['max_sequence_length']
            self.embedding_dim = self.w2v_model.vector_size
            # 这里有问题
            self.class_id = classifier_config['classes']
            self.class_list = [name for name, index in classifier_config['classes'].items()]
            # self.max_label_number = len(self.class_id)
        elif embed == "sbert":
            self.max_sequence_length = 16
            self.embedding_dim = 32
            # self.max_label_number = 16
        self.indu = []
        self.indices = None
        self.indices1 = None
        self.max_label_number = classifier_config['max_label']
        self.batch_size = classifier_config['batch_size']
        self.PADDING = '[PAD]'
        self.logger.info('dataManager initialed...')

    def next_batch(self, X, y, start_index):
        """
        下一次个训练批次
        :param X:
        :param y:
        :param start_index:
        :return:
        """
        last_index = start_index + self.batch_size
        X_batch = list(X[start_index:min(last_index, len(X))])
        y_batch = list(y[start_index:min(last_index, len(X))])
        if last_index > len(X):
            left_size = last_index - (len(X))
            if self.indu is None or len(self.indu) == 0:
                for i in range(left_size):
                    index = np.random.randint(len(X))
                    self.indu.append(index)
            for num in self.indu:
                # index = np.random.randint(len(X))
                X_batch.append(X[num])
                y_batch.append(y[num])
        return np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)

    def padding(self, sentence):
        """
        长度不足max_sequence_length则补齐
        :param sentence:
        :return:
        """
        if len(sentence) < self.max_sequence_length:
            sentence += [self.PADDING for _ in range(self.max_sequence_length - len(sentence))]
        else:
            sentence = sentence[:self.max_sequence_length]
        return sentence

    def prepare(self, sentences, labels):
        """
        输出X矩阵和y向量
        """
        labels = [int(x) for x in labels]
        self.logger.info('loading data...')
        X, y = [], []
        embedding_unknown = [0] * self.embedding_dim
        for record in tqdm(zip(sentences, labels)):
            sentence = self.w2v_util.processing_sentence(record[0], self.stop_words)
            sentence = self.padding(sentence)
            label = tf.one_hot(record[1], depth=self.max_label_number)
            vector = []
            for word in sentence:
                if word in self.w2v_model.wv.vocab:
                    vector.append(self.w2v_model[word].tolist())
                else:
                    vector.append(embedding_unknown)
            X.append(vector)
            y.append(label)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def prepare_s(self, sentence, labels):
        '''
        MBD: 获取sentence数据集、验证集
        '''
        X = get_sentence_embedding(sentence)
        Y = []
        labels = [int(x) for x in labels]
        for label in labels:
            label = tf.one_hot(label, depth=self.max_label_number)
            Y.append(label)
        # Y = np.array(label).reshape(len(label), -1)
        # enc = OneHotEncoder()
        # enc.fit(Y)
        # targets = enc.transform(Y).toarray().tolist()
        # Y = targets
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    def prepare_t(self, sentence, labels):
        hd = handler_doc()
        X = hd.handle_sentence(sentence)
        Y = []
        labels = [int(x) for x in labels]
        for label in labels:
            label = tf.one_hot(label, depth=self.max_label_number)
            Y.append(label)
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    def save_npy(self, filename, target):
        path = r"F:\Classifier\comments_classifier\data\npy\eles9_new2v13_sbert_trained\{}".format(filename)
        np.save(path, target)


    def load_npy(self, filename):
        path = r"F:\Classifier\comments_classifier\data\npy\eles9_new2v13_sbert_trained\{}".format(filename)
        result = np.load(path)
        return result

    def read_eles_set(self, filename):
        # BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # path = os.path.join(BASE_DIR, ele, filename)
        dic = dict()
        with open(filename, 'r', encoding='utf-8') as f:
            dic = json.load(f)
        X = list()
        Y = list()
        for key, value in dic.items():
            for word in value:
                X.append(word)
                Y.append(key)
        return X, Y


    def get_training_set(self, embed, train_val_ratio=0.9):
        """
        获取训练数据集、验证集
        """
        x_train, y_train = self.read_eles_set(self.train_file)
        # convert the data in matrix
        if embed == 'word2vec':
            X, y = self.prepare(x_train, y_train)
        elif embed == 'sbert':
            X, y = self.prepare_s(x_train, y_train)
        elif embed == "tfidf":
            X, y = self.prepare_t(x_train, y_train)
        # shuffle the samples
        self.logger.info('shuffling data...')
        num_samples = len(X)
        # indices = np.arange(num_samples)
        # np.random.shuffle(self.indices)
        X = X[self.indices]
        y = y[self.indices]
        self.logger.info('getting validation data...')
        if self.dev_file is not None:
            X_train = X
            y_train = y
            X_val, y_val = self.get_valid_set(embed=embed)
            X_val = X_val[self.indices1]
            y_val = y_val[self.indices1]
        else:
            # split the data into train and validation set
            X_train = X[:int(num_samples * train_val_ratio)]
            y_train = y[:int(num_samples * train_val_ratio)]
            X_val = X[int(num_samples * train_val_ratio):]
            y_val = y[int(num_samples * train_val_ratio):]
            self.logger.info('validating set is not exist, built...')
        self.logger.info('training set size: {}, validating set size: {}'.format(len(X_train), len(y_val)))
        return X_train, y_train, X_val, y_val

    def get_valid_set(self, embed):
        """
        获取验证集
        :return:
        """
        x_val, y_val = self.read_eles_set(self.dev_file)
        # convert the data in matrix
        if embed == "word2vec":
            X_val, y_val = self.prepare(x_val, y_val)
        elif embed == "sbert":
            X_val, y_val = self.prepare_s(x_val, y_val)
        elif embed == "tfidf":
            X_val, y_val = self.prepare_t(x_val, y_val)
        return X_val, y_val

    def prepare_single_sentence(self, sentence):
        """
        把预测的句子转成矩阵和向量
        :param sentence:
        :return:
        """
        embedding_unknown = [0] * self.embedding_dim
        sentence = self.w2v_util.processing_sentence(sentence, self.stop_words)
        sentence = self.padding(sentence)
        vector = []
        for word in sentence:
            if word in self.w2v_model.wv.vocab:
                vector.append(self.w2v_model[word].tolist())
            else:
                vector.append(embedding_unknown)
        return np.array([vector], dtype=np.float32)


def main():
    logger = get_logger('../logs')
    data_manager = DataManager(logger, 'word2vec')
    x_train, y_train = data_manager.read_eles_set(data_manager.train_file)
    num_samples = len(x_train)
    data_manager.indices = np.arange(num_samples)
    np.random.shuffle(data_manager.indices)
    x_val, y_val = data_manager.read_eles_set(data_manager.dev_file)
    num_samples1 = len(x_val)
    data_manager.indices1 = np.arange(num_samples1)
    np.random.shuffle(data_manager.indices1)
    X_train_t, y_train_t, X_val_t, y_val_t = data_manager.get_training_set(embed="tfidf")
    X_train_w, y_train_w, X_val_w, y_val_w = data_manager.get_training_set(embed="word2vec")
    X_train_s, y_train_s, X_val_s, y_val_s = data_manager.get_training_set(embed="sbert")
    data_manager.save_npy("X_train_t.npy", X_train_t)
    data_manager.save_npy("y_train_t.npy", y_train_t)
    data_manager.save_npy("X_val_t.npy", X_val_t)
    data_manager.save_npy("y_val_t.npy", y_val_t)
    data_manager.save_npy("X_train_w.npy", X_train_w)
    data_manager.save_npy("y_train_w.npy", y_train_w)
    data_manager.save_npy("X_val_w.npy", X_val_w)
    data_manager.save_npy("y_val_w.npy", y_val_w)
    data_manager.save_npy("X_train_s.npy", X_train_s)
    data_manager.save_npy("y_train_s.npy", y_train_s)
    data_manager.save_npy("X_val_s.npy", X_val_s)
    data_manager.save_npy("y_val_s.npy", y_val_s)


if __name__ == "__main__":
    main()