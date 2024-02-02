# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : textcnn.py
# @Software: PyCharm
from abc import ABC
import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
from config import classifier_config
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from BiLstmAttention import BiLSTMAttention, HierarchicalAttentionNetwork
# from attention import attention_layers, attention_layers1
from AttentionLayer import AttentionLayer

class TextCNN(tf.keras.Model, ABC):
    """
    TextCNN模型
    """
    def __init__(self, seq_length, num_filters, num_classes, embedding_dim):
        super(TextCNN, self).__init__()
        # if embed == "word2vec":
        #     self.seq_length = seq_length
        #     self.embedding_dim = embedding_dim
        # elif embed == "sbert":
        #     self.seq_length = 32
        #     self.embedding_dim = 16
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        # self.atten1 = attention_layers1()
        # self.atten = attention_layers()


        if classifier_config['use_attention']:
            self.attention_dim = classifier_config['attention_dim']
            self.attention_W = tf.keras.layers.Dense(classifier_config['attention_dim'], activation='tanh')
            self.attention_V = tf.keras.layers.Dense(1)

            self.attention_dim1 = 32
            self.attention_W1 = tf.keras.layers.Dense(32, activation='tanh')
            self.attention_V1 = tf.keras.layers.Dense(1)

        self.conv1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=[2, 300],
                                            strides=1,
                                            padding='valid',
                                            activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[seq_length-2+1, 1], padding='valid')

        self.conv2 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=[3, 300], strides=1,
                                            padding='valid',
                                            activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[seq_length-3+1, 1], padding='valid')

        self.conv3 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=[4, 300], strides=1,
                                            padding='valid',
                                            activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=[seq_length-4+1, 1], padding='valid')
# ------------------------------------------------------------------------------
        self.conv4 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=[2, 32],
                                            strides=1,
                                            padding='valid',
                                            activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=[16 - 2 + 1, 1], padding='valid')

        self.conv5 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=[3, 32], strides=1,
                                            padding='valid',
                                            activation='relu')
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=[16 - 3 + 1, 1], padding='valid')

        self.conv6 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=[4, 32], strides=1,
                                            padding='valid',
                                            activation='relu')
        self.pool6 = tf.keras.layers.MaxPooling2D(pool_size=[16 - 4 + 1, 1], padding='valid')
# ----------------------------------------------------------------------------------------------------
        self.dropout = tf.keras.layers.Dropout(classifier_config['droupout_rate'], name='dropout')
        self.dense = tf.keras.layers.Dense(num_classes,
                                           activation='softmax',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                           bias_regularizer=tf.keras.regularizers.l2(0.1),
                                           name='dense')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_last', name='flatten')

        self.densef = tf.keras.layers.Dense(12, activation='relu', use_bias=True)
# -------------------------------textrcnn---------------------------------------------------------
#       self.seq_length = seq_length
        self.hidden_dim = 12
        # self.embedding_dim = embedding_dim
        # tf.keras.layers.LSTM.input_shape(12, 1)
        self.forward = tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True)
        self.backward = tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True, go_backwards=True)
        self.max_pool = tf.keras.layers.GlobalMaxPool1D()
        self.dropout_textrcnn = tf.keras.layers.Dropout(classifier_config['droupout_rate'], name='dropout')
        self.dense1 = tf.keras.layers.Dense(2 * self.hidden_dim, activation='tanh')
        # self.dense2 = tf.keras.layers.Dense(num_classes,
        #                                     activation='softmax',
        #                                     kernel_regularizer=tf.keras.regularizers.l2(0.1),
        #                                     bias_regularizer=tf.keras.regularizers.l2(0.1),
        #                                     name='dense')
        # self.flatten = tf.keras.layers.Flatten(data_format='channels_last', name='flatten')
#-----------------------------bi-lstm--------------------------------------------------------------
        self.bilstm = Bidirectional(LSTM(64, dropout=0.2, return_sequences=True))
        self.dropout_bilstm = tf.keras.layers.Dropout(classifier_config['droupout_rate'], name='dropout')
        self.attention = AttentionLayer(attention_size=128)
        self.attn = HierarchicalAttentionNetwork(attention_dim=32)

#-----------------------------------------------------------------------------------------------------------------------

    @tf.function
    def call(self, inputs, data_manager, training=None):
        # print("进入了重写的call方法")
        if classifier_config['use_attention']:
            xh = tf.reshape(inputs[0], [-1, self.embedding_dim * self.seq_length])
            xhg = tf.expand_dims(xh, -1)
            inputs0 = self.atten(xhg)
            shuru = tf.reshape(inputs0, [64, self.seq_length, self.embedding_dim])
            # print("shuru", shuru)
            gh = tf.reshape(inputs[1], [-1, 16*32])
            xhz = tf.expand_dims(gh, -1)
            inputs1 = self.atten1(xhz)
            zz = tf.reshape(inputs1, [64, 16, 32])
        else:
            shuru = inputs[0]
            zz = inputs[1]
# --------------------对inputs1 进行textrcnn------------------------------------------------------------------
        # inputs0 = inputs[0]
        # left_embedding = self.forward(inputs[0])
        # print("left_embedding", left_embedding)
        # right_embedding = self.backward(inputs[0])
        # print("right_embedding", right_embedding)
        # concat_outputs = tf.keras.layers.concatenate([left_embedding, inputs[0], right_embedding], axis=-1)

        # dropout_outputs = self.dropout(concat_outputs, training)
        # print("dropout", dropout_outputs)
        # fc_outputs = self.dense1(dropout_outputs)
# ----------------------------  textrcnn结束------------------------------------------------------------------
        shuru = tf.expand_dims(shuru, -1)
        print("看看扩展后，input第一个元素是什么样：%s" % shuru)

        # print("inputs.shape:", inputs.shape)
        # print("shuru.shape:", shuru.shape)
        # print(inputs)
        pooled_output = []
        # print("self.con1", self.conv1.shape)



        con1 = self.conv1(shuru)
        print("con1:", con1)
        pool1 = self.pool1(con1)
        print("pool1:", pool1)
        pooled_output.append(pool1)

        con2 = self.conv2(shuru)
        print("con2", con2)
        pool2 = self.pool2(con2)
        print("pool2", pool2)
        pooled_output.append(pool2)

        con3 = self.conv3(shuru)
        print("con3", con3)
        pool3 = self.pool3(con3)
        print("pool3", pool3)
        pooled_output.append(pool3)


        # --------Modified by David---------
        # inputs1 = inputs[1]
        zz = tf.expand_dims(zz, -1)
        print("看看扩展后，input第2个元素是什么样：%s" % zz)
        # print(inputs)
        pooled_output1 = []
        con4 = self.conv4(zz)
        print("conv4:", con4)
        pool4 = self.pool4(con4)
        print("pool4:", pool4)
        pooled_output1.append(pool4)

        con5 = self.conv5(zz)
        pool5 = self.pool5(con5)
        pooled_output1.append(pool5)

        con6 = self.conv6(zz)
        pool6 = self.pool6(con6)
        pooled_output1.append(pool6)
        # -------------merge--------------
        print("pooled_output:", pooled_output)
        print("pooled_output1:", pooled_output1)
        concat_outputs = tf.keras.layers.concatenate(pooled_output, axis=-1, name='concatenate')
        print("concat_outputs:", concat_outputs)
        concat_outputs1 = tf.keras.layers.concatenate(pooled_output1, axis=-1, name='concatenate')
        print("concat_outputs1:", concat_outputs1)
        flatten_outputs = self.flatten(concat_outputs)
        print("flatten_outputs:", flatten_outputs)
        flatten_outputs1 = self.flatten(concat_outputs1)
        print("flatten_outputs1:", flatten_outputs1)
        # kk = inputs[2]
        # kk = tf.reshape(kk, [64, 12, 1])
        # print("kk", kk)
# ----------------------------tfidf + textrcnn----------------------------------------------------
#         left_embedding = self.forward(kk)
#         print("left_embedding", left_embedding)
#         right_embedding = self.backward(kk)
#         print("right_embedding", right_embedding)
#         concat_outputs = tf.keras.layers.concatenate([left_embedding, right_embedding], axis=-1)
#         print("concat_outputs", concat_outputs)
#         # dropout_outputs = self.dropout(concat_outputs, training)
#         # print("dropout", dropout_outputs)
#         fc_outputs = self.dense1(concat_outputs)
#         print("dense1", fc_outputs)
#         pool_outputs = self.max_pool(fc_outputs)
#         print("pool_outputs", pool_outputs)
# ------------------------------tfidf + bi-lstm------------------------------------------------------------
#         kk = inputs[2]
#         kk = tf.reshape(kk, [64, 4, 3])
#         print("kk", kk)
#         embedding = self.bilstm(kk)
#         print("embedding", embedding)
#         dropout_outputs = self.dropout_bilstm(embedding, training)
#         print("drop_outputs:", dropout_outputs)
#----------------------------------Bi-LSTM + attention-----------------------------------------------------------------
        kk = inputs[0]
        print("输入到Bilstm前", kk)
        x = self.bilstm(kk)
        print("经bilstm后输出", x)
        # x = self.attention(x)
        x = self.attn(x)
        # out_put = self.bilstmAttention.call(wordEmbedding=kk)
        print("attention_out_out:", x)
#==-----------------------------------------------------------------------------------------------------------------
        # print("++++++++inputs[2]", kk)
        final = tf.keras.layers.concatenate([flatten_outputs, flatten_outputs1, x])
        # final = tf.reshape(final, [64, 22, 18])
        print("final", final)
        # david = self.densef(final)
        # added = tf.keras.layers.Add()([david, kk])
        # print("added", added)
        # david = tf.matmul(tf.transpose(kk), final)
        # david = tf.reshape(david, [64, -1])
        # print("david", final)
        # embedding = self.bilstm(final)
        # print("embedding:", embedding)
        dropout_outputs = self.dropout(final, training)
        outputs = self.dense(dropout_outputs)
        print("最终outputs", outputs)
        return outputs










