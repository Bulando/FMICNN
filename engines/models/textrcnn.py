# -*- coding: utf-8 -*-
# @Time : 2020/11/6 10:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : textrcnn.py
# @Software: PyCharm
from abc import ABC
import tensorflow as tf
from config import classifier_config


class TextRCNN(tf.keras.Model, ABC):
    """
    TextCNN模型
    """

    def __init__(self, seq_length, num_classes, hidden_dim, embedding_dim):
        super(TextRCNN, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.forward = tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True)
        self.backward = tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True, go_backwards=True)
        self.max_pool = tf.keras.layers.GlobalMaxPool1D()
        self.dropout = tf.keras.layers.Dropout(classifier_config['droupout_rate'], name='dropout')
        self.dense1 = tf.keras.layers.Dense(2 * self.hidden_dim + self.embedding_dim, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(num_classes,
                                            activation='softmax',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                            bias_regularizer=tf.keras.regularizers.l2(0.1),
                                            name='dense')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_last', name='flatten')

    @tf.function
    def call(self, inputs, training=None):
        print("inputs", inputs)
        left_embedding = self.forward(inputs)
        print("left_embedding", left_embedding)
        right_embedding = self.backward(inputs)
        print("right_embedding", right_embedding)
        concat_outputs = tf.keras.layers.concatenate([left_embedding, inputs, right_embedding], axis=-1)
        print("concat_outputs", concat_outputs)
        dropout_outputs = self.dropout(concat_outputs, training)
        fc_outputs = self.dense1(dropout_outputs)
        pool_outputs = self.max_pool(fc_outputs)
        outputs = self.dense2(pool_outputs)
        print("outputs", outputs)
        return outputs
