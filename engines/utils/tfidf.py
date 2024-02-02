# -*- coding: utf-8 -*-

"""
@project: custom words similarity
@author: David
@time: 2021/3/17 15:03
"""
from gensim import corpora, models
import jieba, re, json, os
from collections import OrderedDict

class handler_doc(object):
    def __init__(self):
        self.max_sequence_length = 12
        self.PADDING = 0
        pass

    def fetch_chinese(self, word):
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        chinese = re.sub(pattern, '', word)
        return chinese

    def stopwordslist(self):
        '''创建停用词列表'''
        stopwords = [line.strip() for line in open('./stopwords.txt', encoding='utf-8').readlines()]
        return stopwords

    def read_eles_set(self, ele):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        path = os.path.join(BASE_DIR, "data", ele, 'train.txt')
        dic = dict()
        with open(path, 'r', encoding='utf-8') as f:
            dic = json.load(f)
        return dic

    def get_eles_list(self, dic):
        for key, value in sorted(dic.items()):
            keyl = self.doc_splict_words(value)
            dic[key] = keyl
        return dic

    def get_ele_dic_corpus(self, keyl):
        all = []
        for key, value in sorted(keyl.items()):
            all.append(value)
        dictionary = corpora.Dictionary(all)
        new_corpurs = [dictionary.doc2bow(text) for text in all]
        return dictionary, new_corpurs

    def save_dic_corpus(self, dictionary, corpus):
        tfidf = models.TfidfModel(corpus)
        tfidf.save("eles.tfidf")
        dictionary.save("eles.dict")

    def handle_sentence(self, sentence):
        """
        对外API
        """
        tfidf = models.TfidfModel.load(r"F:\Classifier\comments_classifier\engines\utils\eles.tfidf")
        dictionary = corpora.Dictionary.load(r"F:\Classifier\comments_classifier\engines\utils\eles.dict")
        # all = []
        # for sen in sentence:
        #     new = self.doc_splict_words(sen)
        #     all.append(new)
        tfidf_vec = []
        for sen in sentence:
            word = self.fetch_chinese(sen)
            seg_list = self.seg_depart(word)
            sen_bow =dictionary.doc2bow(seg_list)
            sen_tfidf = tfidf[sen_bow]
            tfidf_vec.append([i[1] for i in sen_tfidf])
        for i, vec in enumerate(tfidf_vec):
            tfidf_vec[i] = self.padding(vec)
        return tfidf_vec


    def padding(self, sentence):
        """
        长度不足max_sequence_length则补齐
        :param sentence:
        :return:
        """
        if len(sentence) < self.max_sequence_length:
            sentence.extend([self.PADDING for _ in range(self.max_sequence_length - len(sentence))])
        else:
            sentence = sentence[:self.max_sequence_length]
        return sentence


    def seg_depart(self, sentence):
        '''对大词进行分词'''
        sentence_depart = jieba.cut(sentence)
        # stopwords = self.stopwordslist()
        outstr = []
        # for word in sentence_depart:
        #     if word not in stopwords:
        #         outstr.append(word)
                # outstr += " "
        return sentence_depart

    def doc_splict_words(self, lists):
        '''对文档列表进行分词'''
        new = []
        for word in lists:
            word = self.fetch_chinese(word)
            seg_list = self.seg_depart(word)
            #     new.extend(seg_list)
            # str = "-:|:-".join(line)
            new.extend(seg_list)
        return new

def main1():
    hd = handler_doc()
    vec = hd.handle_sentence(['我是一个中国人', '我爱我的祖国'])
    print(vec)


def main():
    handler = handler_doc()
    dic = handler.read_eles_set('eles9')
    dic = handler.get_eles_list(dic)
    dictionary, new_corpurs = handler.get_ele_dic_corpus(dic)
    handler.save_dic_corpus(dictionary=dictionary, corpus=new_corpurs)
    pass


if __name__ == "__main__":
    main()