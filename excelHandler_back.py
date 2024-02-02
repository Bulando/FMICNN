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
            for i in range(0, self.rows):
                values = self.table.row_values(i)
                list1 = []
                for x in range(len(values)):
                    if len(values[x]) > 0:
                        list1.append(values[x])
                # self.table_key.append(values[0])
                list0.append(list1)
            return list0

    def filter_list(self, li):
        all = list()
        for i in range(len(li)):
            new = []
            if len(li[i][4]) > 0:
                num = self.table_key.index(li[i][4])
                new.append(li[i][2])
                new.append(li[num][2])
                new.append('1')
                all.append(new)
        return all

    def filter_list_by_new(self, li):
        all = []
        for i, x in enumerate(li):
            zhu = x[0]
            for y in x[1:]:
                new = []
                new.append(zhu)
                new.append(y)
                new.append('1')
                all.append(new)
                new1 = []
                new1.append(zhu)
                num = random.randint(0, len(li) - 1)
                while num == i:
                    num = random.randint(0, len(li) - 1)
                new1.append(li[num][random.randint(0, len(li[num])-1)])
                new1.append('0')
                all.append(new1)
        return all

    def add_noise_list_by_new(self, li):
        # for i in range(0, len(li)):
        #     new = []
        #     zhu = li[i][0]
        #     num = random.randint(0, len(li)-1)
        #     while num == i:
        #         num = random.randint(0, len(li)-1)
        #     cong = li[num][0]
        #     new.append(zhu)
        #     new.append(cong)
        #     new.append('0')
        #     li.append(new)
        SortList = list(range(0, len(li)))
        shuffle(SortList)
        for i in range(len(SortList)):
            SortList[i] = li[SortList[i]]
        return SortList

    def add_noise_list(self, li):
        lists_all = []
        # 生成4万的一个随机数
        for i in range(0, 40000, 1):
            new = []
            num = random.randint(1, 40001)
            new.append(li[i][0])
            new.append(li[num][1])
            new.append('0')
            li.append(new)
        SortList = list(range(0, len(li)))
        shuffle(SortList)
        for i in range(len(SortList)):
            SortList[i] = li[SortList[i]]

        return SortList


    def to_file_by_new(self, li, element_name):
        nums = len(li)
        long_num = 0
        long_seq = []
        for line in li:
            if len(line[0]) + len(line[1]) > 125:
                print(line)
                long_seq.append(line)
                li.remove(line)
                long_num += 1
        number = int(len(long_seq)/0.05)
        li = li[:number]
        li.extend(long_seq)
        shuffle(li)
        print("%s超过最大长度占比%.2f%%,共有数据%d,超过的有%d" % (element_name, long_num/len(li)*100, nums, long_num))
        path = './data/%s/train.txt' % element_name
        path1 = './data/%s/dev.txt' % element_name
        path2 = './data/%s/test_ele.txt' % element_name
        output = open(path, 'w', encoding='utf-8')
        x = 0
        for row in li[:int(len(li)*0.7)]:
            rowtxt = '{}-:|:-{}-:|:-{}'.format(row[0], row[1], row[2])
            output.write(rowtxt)
            x+=1
            output.write('\n')
        output.close()
        output = open(path2, 'w', encoding='utf-8')
        index = x - 1
        end = index+int(len(li)*0.1)
        for row in li[index:]:
            rowtxt = '{}-:|:-{}-:|:-{}'.format(row[0], row[1], row[2])
            output.write(rowtxt)
            x+=1
            output.write('\n')
        output.close()


    def to_file(self, li, element_name):
        path = './sdata/%s/train.txt' % element_name
        path1 = './sdata/%s/dev.txt' % element_name
        path2 = './sdata/%s/test.txt' % element_name
        output = open(path, 'w', encoding='utf-8')
        x = 0
        for row in li[:int(len(li) * 0.9)]:
            rowtxt = '{}-:|:-{}-:|:-{}'.format(row[0], row[1], row[2])
            output.write(rowtxt)
            x += 1
            output.write('\n')
        output.close()
        output = open(path2, 'w', encoding='utf-8')
        index = x - 1
        end = index + int(len(li) * 0.1)
        for row in li[index:]:
            rowtxt = '{}-:|:-{}-:|:-{}'.format(row[0], row[1], row[2])
            output.write(rowtxt)
            x += 1
            output.write('\n')
        output.close()



def main():
    # 接收？一定是表格数据？
    table = ExcelRead('(0205)同义词.xlsx', 'Sheet1')
    lists = table.list_data()
    zuizhong = table.filter_list_by_new(lists)
    zuizhong1 = table.add_noise_list_by_new(zuizhong)
    table.to_file(zuizhong1, '0205')
    # print(zuizhong1)
    # dir_path = os.path.dirname(os.path.realpath(__file__))



if __name__ == '__main__':
    main()
