# -*- coding: utf-8 -*-

"""
@project: custom words similarity
@author: David
@time: 2021/5/9 10:55
"""
import numpy
import keras
import tensorflow as tf

from keras import backend as K
from keras import activations
from keras.engine.topology import Layer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional

K.clear_session()


class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)



    # def get_config(self):
    #     config = super().get_config()
    #     config['attention_size'] = self.attention_size
    #     return config


    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                 initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
