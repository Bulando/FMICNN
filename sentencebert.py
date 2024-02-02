# -*- coding: utf-8 -*-

"""
@project: custom words similarity
@author: David
@time: 2021/1/9 9:57
"""
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader
import os, json
import numpy as np
last = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(last, 'data', 'eles9', 'sbert_model')
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
model_trained = SentenceTransformer(model_name_or_path=path)
macro_averge = {'00': 0.65, '01': 0.25, '02': 0.35, '03': 0.30, '04': 0.01, '05': 0.10
                , '06': 0.15, '07': -0.01, '08': 0.05, '11': 0.65, '12': 0.15, '13': 0.20
                , '14': 0.02, '15': 0.06, '16': 0.12, '17': 0.01, '18': -0.01, '22': 0.65
                , '23': 0.25, '24': 0.05, '25': 0.15, '26': 0.20, '27': -0.02, '28': 0.05
                , '33': 0.65, '34': 0.03, '35': 0.07, '36': 0.10, '37': -0.00, '38': 0.00
                , '44': 0.75, '45': 0.10, '46': 0.10, '47': 0.10, '48': 0.10, '55': 0.65
                , '56': 0.35, '57': -0.01, '58': 0.15, '66': 0.65, '67': 0.00, '68': 0.25
                , '77': 0.65, '78': 0.05, '88': 0.65}

def train_sentence_transformers(ele):
    last_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(last_dir, 'data', ele, 'train.txt')
    with open(path, 'r', encoding='utf-8') as f:
        reader = f.readlines()
    train_examples = []
    for index, line in enumerate(reader):
        split_line = line.strip().split('-:|:-')
        text_a = split_line[0]
        text_b = split_line[1]
        text = [text_a, '']
        label = split_line[1]
        train_examples.append(InputExample(texts=text, label=int(label)))
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.SoftmaxLoss(model, sentence_embedding_dimension=512, num_labels=16)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)


class train_sentence_transformers(object):
    def __init__(self):
        self.list = []
        self.group_dict = {}
        self.adjust_dict = {}
        pass

    def reduce_dict_data(self, path):
        dic = dict()
        new_dic = dict()
        with open(path, 'r', encoding='utf-8') as f:
            dic = json.load(f)
        for key, value in dic.items():
            new_dic[key] = value[:100]
        return new_dic

    def get_group_datas(self, dir):
        dic = dict()
        with open(dir, 'r', encoding='utf-8') as f:
            dic = json.load(f)
        all = []
        groups = []
        for key, value in dic.items():
            for val in value:
                all.append([val, key])
        for i, wordA in enumerate(all):
            if i >= len(all)-1:
                break
            for j, wordB in enumerate(all[i+1:]):
                groups.append([wordA[0], wordB[0], wordA[1]+wordB[1]])
        for group in groups:
            self.group_dict.setdefault(group[2], []).append(group)
        folder_path, file_name = os.path.split(dir)
        path = os.path.join(folder_path, 'group.txt')
        with open(path, 'w') as file:
            json.dump(self.group_dict, file, ensure_ascii=False)
        pass

    def score_group_dict(self, dir):
        dic = dict()
        new_dict = {}
        with open(dir, 'r', encoding='utf-8') as f:
            dic = json.load(f)
        for key, value in dic.items():
            new_value = []
            for list in value:
                embeddings = model.encode(list[:-1], convert_to_tensor=True)
                score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
                list.append(score.tolist()[0][0])
                new_value.append(list)
            new_dict[key] = new_value
            # all.extend(value)
        folder_path, file_name = os.path.split(dir)
        path1 = os.path.join(folder_path, 'results.txt')
        with open(path1, 'w') as file:
            json.dump(new_dict, file, ensure_ascii=False)
        pass


    def get_average_score(self, dir):
        dic = {}
        with open(dir, 'r', encoding='utf-8') as f:
            dic = json.load(f)
        for key, value in dic.items():
            sum = 0
            leng = len(value)
            for val in value:
                sum += val[-1]
            print("%s组合的平均相似度是%.2f" % (key, sum/leng))

    def adjust_group_dict(self, dir):
        dic = {}
        with open(dir, 'r', encoding='utf-8') as f:
            dic = json.load(f)
        for key, value in dic.items():
            ave = macro_averge[key]
            for i, val in enumerate(value):
                new_val = val[-1]
                if key[0] == key[1]:
                    if val[-1] < ave:
                        new_val = val[-1] + (ave - val[-1])*0.8
                else:
                    if val[-1] > ave:
                        new_val = val[-1] - (val[-1] - ave)*0.8
                value[i].append(new_val)
        folder_path, file_name = os.path.split(dir)
        path1 = os.path.join(folder_path, 'results_update.txt')
        with open(path1, 'w') as file:
            json.dump(dic, file, ensure_ascii=False)
        print("更新完毕！")

    def make_data_loader(self, dir):
        dic = {}
        all = []
        with open(dir, 'r', encoding='utf-8') as f:
            dic = json.load(f)
        for key, value in dic.items():
            for val in value:
                all.append([val[0], val[1], ("%.2f" % val[-1])])
        indices = np.arange(len(all))
        np.random.shuffle(indices)
        all = np.array(all)[indices]
        train_examples = []
        for line in all:
            train_examples.append(InputExample(texts=[line[0], line[1]], label=float(line[-1])))
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        return train_dataloader


    def defineTrain_a_model(self, dir, train_dataloader):
        train_loss = losses.CosineSimilarityLoss(model)
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
        print("训练结束！")
        model.save(path=dir)
        pass


def train_sentence_transformers1(ele):
    last_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(last_dir, 'data', ele, 'train.txt')
    with open(path, 'r', encoding='utf-8') as f:
        reader = f.readlines()
    train_examples = []
    for index, line in enumerate(reader):
        split_line = line.strip().split('-:|:-')
        text_a = split_line[0]
        text_b = split_line[1]
        text = [text_a, '']
        label = split_line[1]
        train_examples.append(InputExample(texts=text, label=int(label)))
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.SoftmaxLoss(model, sentence_embedding_dimension=512, num_labels=16)
    model.fit(train_objectives=[(train_dataloader, train_loss)], output_path='', epochs=1, warmup_steps=100)


def get_sentence_embedding(sentences):
    corpus_embeddings = model_trained.encode(sentences, convert_to_tensor=True)
    embeddings = corpus_embeddings.view(len(corpus_embeddings), 16, 32).tolist()
    return embeddings

def generate_bert_dataset():
    dic = {}
    all = []
    last_dir = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.join(last_dir, 'data', 'eles9', 'results_update.txt')
    with open(dir, 'r', encoding='utf-8') as f:
        dic = json.load(f)
    for key, value in dic.items():
        for val in value:
            all.append([val[0], val[1], ("%.2f" % val[-1])])
    changdu = len(all)
    folder_path = os.path.join(last_dir, 'data', 'eles9')
    path1 = os.path.join(folder_path, 'bert-train.txt')
    path2 = os.path.join(folder_path, 'bert-dev.txt')
    path3 = os.path.join(folder_path, 'bert-test.txt')
    with open(path1, 'w') as file:
        json.dump(all[:int(changdu*0.8)], file, ensure_ascii=False)
    with open(path2, 'w') as file:
        json.dump(all[int(changdu*0.8):int(changdu*0.9)], file, ensure_ascii=False)
    with open(path3, 'w') as file:
        json.dump(all[int(changdu * 0.9):], file, ensure_ascii=False)

if __name__ == '__main__':
    generate_bert_dataset()

if __name__ == "__main1__":
    '训练sbert'
    last_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(last_dir, 'data', 'eles9', 'train.txt')
    path_100 = os.path.join(last_dir, 'data', 'eles9', 'train100.txt')
    group = os.path.join(last_dir, 'data', 'eles9', 'group.txt')
    results = os.path.join(last_dir, 'data', 'eles9', 'results.txt')
    path_d = os.path.join(last_dir, 'data', 'eles9', 'results_update.txt')
    path_out = os.path.join(last_dir, 'data', 'eles9', 'sbert_model')
    # with open(path, 'w') as file:
    #     json.dump(new_dic, file, ensure_ascii=False)
    tst = train_sentence_transformers()
    train_dataloader = tst.make_data_loader(path_d)
    tst.defineTrain_a_model(path_out, train_dataloader=train_dataloader)
    # tst.score_group_dict(group)
    # train_sentence_transformers('0205')
