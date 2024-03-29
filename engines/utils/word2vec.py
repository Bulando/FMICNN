# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : word2vec_util.py
# @Software: PyCharm
import pandas as pd
from gensim.models.word2vec import Word2Vec
from config import word2vec_config
import jieba
import os
from engines.utils.logger import get_logger # MBD
import numpy as np  # MBD

class Word2VecUtils:
    def __init__(self, logger):
        self.logger = logger
        self.stop_words = word2vec_config['stop_words']
        self.train_data = word2vec_config['train_data']
        model_dir = word2vec_config['model_dir']
        model_name = word2vec_config['model_name']
        self.model_path = os.path.join(model_dir, model_name)
        self.dim = word2vec_config['word2vec_dim']

    @staticmethod
    def processing_sentence(x, stop_words):
        cut_word = jieba.cut(str(x).strip())
        if stop_words:
            words = [word for word in cut_word if word not in stop_words and word != ' ']
        else:
            words = list(cut_word)
            words = [word for word in words if word != ' ']
        return words

    def get_stop_words(self):
        stop_words_list = []
        try:
            with open(self.stop_words, 'r', encoding='utf-8') as stop_words_file:
                for line in stop_words_file:
                    stop_words_list.append(line.strip())
        except FileNotFoundError:
            return stop_words_list
        return stop_words_list

    def train_word2vec(self):
        train_df = pd.read_csv(self.train_data, encoding='utf-8')
        stop_words = self.get_stop_words()
        # 切词
        self.logger.info('Cut sentence...')
        train_df['sentence'] = train_df.sentence.apply(self.processing_sentence, args=(stop_words,))
        # 删掉缺失的行
        train_df.dropna(inplace=True)
        all_cut_sentence = train_df.sentence.to_list()
        # 训练词向量
        self.logger.info('Training word2vec...')
        w2v_model = Word2Vec(size=self.dim, workers=10, min_count=3)
        w2v_model.build_vocab(all_cut_sentence)
        w2v_model.train(all_cut_sentence, total_examples=w2v_model.corpus_count, epochs=100)
        w2v_model.save(self.model_path)

    def train_1_word2vec(self, DIR):
        all = []
        for path in os.listdir(DIR):
            path1 = os.path.join(DIR, path)
            train_df = pd.read_excel(path1, keep_default_na=False)
            Data_Array = np.array(train_df)
            Data_List = Data_Array.tolist()
            li = []
            word_li = []
            for list1 in Data_List:
                # li.extend([x for x in list if x != ''])
                for x in list1:
                    if x != '':
                        cut_word = jieba.lcut(x, cut_all=False)
                        li.extend(cut_word)
            # leng = len(li)
            all.append(list(set(li)))
        with open("9个eles分词语料.txt", "w") as f:
            for li in all:
                for w in li:
                    f.write(w+"\t")
                f.write("\n")
            # all.extend(li[:int(0.3*leng)])
        sh_index = np.arange(len(all))
        np.random.shuffle(sh_index)
        all = np.array(all)[sh_index]
        stop_words = self.get_stop_words()
        for word in all:
            word = self.processing_sentence(word, stop_words)
            word_li.append(word)
        self.logger.info('Training word2vec...')
        w2v_model = Word2Vec(size=self.dim, workers=10, min_count=3)
        w2v_model.build_vocab(word_li)
        w2v_model.train(word_li, total_examples=w2v_model.corpus_count, epochs=100)
        w2v_model.save(self.model_path)



if __name__ == '__main__':
    pwd = os.getcwd()
    # 获取当前文件的前两级目录
    dir = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
    dirs = os.path.join(dir, "w2v训练数据")
    path = os.path.join(dir, 'logs')
    w2v =  Word2VecUtils(get_logger(path))
    w2v.train_1_word2vec(dirs)

