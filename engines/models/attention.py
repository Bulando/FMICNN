# -*- coding: utf-8 -*-

"""
@project: custom words similarity
@author: David
@time: 2021/3/23 13:28
"""
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf



class attention_layers(Layer):
    def __init__(self, **kwargs):
        super(attention_layers, self).__init__(**kwargs)

    # @tf.function
    def build(self, inputshape):
        assert len(inputshape) == 3
        # print("inputshape is --- :", inputshape)
        # print("inputshape的长度", len(inputshape))
        # print("inputshape[0] is --- :", inputshape[0])
        # print("inputshape[1] is --- :", inputshape[1])
        # print("inputshape[2] is --- :", inputshape[2])
        self.W = self.add_weight(name='attr_weight', shape=(inputshape[1], inputshape[2]),
                                 initializer='uniform', trainable=True)
        print("self.W ---", self.W)
        self.b = self.add_weight(name='attr_bias', shape=(inputshape[1],),
                                 initializer='uniform', trainable=True)
        print("self.b ---", self.b)
        super(attention_layers, self).build(inputshape)

    # @tf.function
    def call(self, inputs):
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # print("x is ---", x)
        # print("self.W is ---", self.W)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        # print("a is ----", a)
        outputs = K.permute_dimensions(a*x, (0, 2, 1))
        # outputs = K.sum(outputs, axis=1)
        # print("outputs ---", outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class attention_layers1(Layer):
    def __init__(self, **kwargs):
        super(attention_layers1, self).__init__(**kwargs)

    # @tf.function
    def build(self, inputshape):
        assert len(inputshape) == 3
        # print("inputshape is --- :", inputshape)
        # print("inputshape的长度", len(inputshape))
        # print("inputshape[0] is --- :", inputshape[0])
        # print("inputshape[1] is --- :", inputshape[1])
        # print("inputshape[2] is --- :", inputshape[2])
        self.W = self.add_weight(name='attr_weight', shape=(inputshape[1], inputshape[2]),
                                 initializer='uniform', trainable=True)
        # print("self.W ---", self.W)
        self.b = self.add_weight(name='attr_bias', shape=(inputshape[1],),
                                 initializer='uniform', trainable=True)
        # print("self.b ---", self.b)
        super(attention_layers1, self).build(inputshape)

    # @tf.function
    def call(self, inputs):
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # print("x is ---", x)
        # print("self.W is ---", self.W)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        # print("a is ----", a)
        outputs = K.permute_dimensions(a*x, (0, 2, 1))
        # outputs = K.sum(outputs, axis=1)
        # print("outputs ---", outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]