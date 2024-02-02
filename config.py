# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py 
# @Software: PyCharm


# [train_classifier, interactive_predict, train_word2vec]
mode = 'train_word2vec'

word2vec_config = {
    'stop_words': 'F:\\Classifier\\comments_classifier\\data/w2v_data\\stop_words.txt',  # 停用词(可为空)
    'train_data': 'F:\\Classifier\\comments_classifier\\同义词new\\(00T1)同义词.xlsx',
    # 'train_data': 'F:/Classifier/comments_classifier/data/w2v_data/comments_data.csv',  # 词向量训练用的数据
    'model_dir': 'F:\\Classifier\\comments_classifier\\model\\w2v_model',  # 词向量模型的保存文件夹
    'model_name': 'w2v_eles13_model.pkl',  # 词向量模型名 w2v_model.pkl
    'word2vec_dim': 300,  # 词向量维度
}

classifier_config = {
    'classifier': 'textcnn',  # 模型选择
    'train_file': 'F:/Classifier/comments_classifier/data/eles9/train.txt',  # 训练数据集
    # 'dev_file': 'data/data/dev_data.csv',  # 验证数据集
    'dev_file': 'F:/Classifier/comments_classifier/data/eles9/test.txt',
    'classes': {'negative': 0, 'positive': 1},  # 类别和对应的id
    'max_label': 9,  # MBD 新添加的一个变量
    'checkpoints_dir': 'model/1w2v-textcnn',  # 模型保存的文件夹
    'checkpoint_name': 'w2v-textcnn_model',  # 模型保存的名字
    'num_filters': 64,  # 卷集核的个数  64 MBD
    'learning_rate': 0.001,  # 学习率
    'epoch': 30,  # 训练epoch
    'max_to_keep': 1,  # 最多保存max_to_keep个模型
    'print_per_batch': 20,  # 每print_per_batch打印
    'is_early_stop': True,  # 是否提前结束 True
    'use_attention': False,  # 是否引入attention True
    'attention_dim': 300,  # attention大小 300
    'patient': 8,
    'batch_size': 64,
    'max_sequence_length': 150,
    'droupout_rate': 0.3,  # 遗忘率
    'hidden_dim': 100,  # 隐藏层维度 200
    'metrics_average': 'macro',  # 若为二分类则使用binary，多分类使用micro或macro
    'use_focal_loss': False,  # 类别样本比例失衡的时候可以考虑使用
    'labels': [0, 1, 2, 3, 4, 5, 6, 7, 8]
}
