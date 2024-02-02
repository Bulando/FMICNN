# -*- coding: utf-8 -*-

"""
@project: custom words similarity
@author: David
@time: 2021/5/6 17:20
"""
from abc import ABC
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import tensorflow as tf
from config import classifier_config

class bilstm(tf.keras.Model, ABC):
    def __init__(self, num_classes, embedding_dim):
        super(bilstm, self).__init__()
        self.embedding_dim = embedding_dim
        self.layer = Bidirectional(LSTM(64))
        self.dropout = tf.keras.layers.Dropout(classifier_config['droupout_rate'], name='dropout')
        self.dense = tf.keras.layers.Dense(num_classes,
                                            activation='softmax',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                            bias_regularizer=tf.keras.regularizers.l2(0.1),
                                            name='dense')


    @tf.function
    def call(self, inputs, training=None):
        print("inputs", inputs)
        embedding = self.layer(inputs)
        print("left_embedding", embedding)
        dropout_outputs = self.dropout(embedding, training)
        outputs = self.dense(dropout_outputs)
        print("outputs", outputs)
        return outputs