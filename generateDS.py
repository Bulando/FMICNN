# -*- coding: utf-8 -*-

"""
@project: custom words similarity
@author: David
@time: 2021/1/8 15:18
"""

import xlrd
import os
import random
from random import shuffle


class ExcelRead(object):
    def __init__(self, excel_name, sheet_name):
        # 获取当前.py文件所在文件夹层
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # 获取excel所在文件目录
        excel_path = os.path.join(dir_path, '同义词new', excel_name)
        workbook = xlrd.open_workbook(excel_path)
        self.table = workbook.sheet_by_name(sheet_name)
        self.columns = self.table.ncols
        self.rows = self.table.nrows  # 获取总行数
        # self.table_key = self.table.row_values(0)
        self.table_key = []
        print(self.table)

    def list_data(self):
        if self.rows <= 1:
            print(u"excel行数小于等于1")
        else:
            list0 = []
            num = 0
            for i in range(0, self.rows):
                values = self.table.row_values(i)
                values = [x for x in values if x != '']
                if len(values) < 10:
                    continue
                for x in range(len(values)):
                    if len(values[x]) > 0:
                        list0.append([values[x], str(num)])
                num += 1
            print("数据集中一共所分类别为{}".format(num+1))
            return list0

    def to_file(self, lists, element_name):
        li = []
        SortList = list(range(0, len(lists)))
        shuffle(SortList)
        for i in range(len(SortList)):
            li.append(lists[SortList[i]])
        path = './data/%s/train.txt' % element_name
        path1 = './data/%s/dev.txt' % element_name
        path2 = './data/%s/test.txt' % element_name
        output = open(path, 'w', encoding='utf-8')
        output1 = open(path1, 'w', encoding='utf-8')
        index = 0
        for row in li[:int(len(li)*0.7)]:
            rowtxt = '{}-:|:-{}'.format(row[0], row[1])
            output.write(rowtxt)
            index += 1
            output.write('\n')
        output.close()
        for row in li[index:]:
            rowtxt = '{}-:|:-{}'.format(row[0], row[1])
            output1.write(rowtxt)
            output1.write('\n')
        output1.close()



def main():
    # 接收？一定是表格数据？
    table = ExcelRead('(00T1)同义词.xlsx', 'Sheet1')
    lists = table.list_data()
    table.to_file(lists, '00T1')

    # zuizhong = table.filter_list_by_new(lists)
    # zuizhong1 = table.add_noise_list_by_new(zuizhong)
    # table.to_file(zuizhong1, '0035')
    # print(zuizhong1)
    # dir_path = os.path.dirname(os.path.realpath(__file__))



if __name__ == '__main__':
    main()