#将VOC数据集按照train:val:test=8:1:1划分并生成相应的txt文件

import os
import random
import argparse

parser = argparse.ArgumentParser()
# xml文件的路径，Annotations文件夹
parser.add_argument('--xml_path', default=r'C:\Users\28244\Desktop\yolov8\dataset\3\Annotations', type=str, help='input xml label path')
# 数据集的划分，地址选择自己数据下的ImageSets/Main
parser.add_argument('--txt_path', default=r'C:\Users\28244\Desktop\yolov8\dataset\3\ImageSets\Main', type=str, help='output txt label path')
opt = parser.parse_args()

train_percent = 0.8  # 训练集所占比例
val_percent = 0.1  # 验证集所占比例
test_persent = 0.1  # 测试集所占比例

xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)

if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list = list(range(num))

t_train = int(num * train_percent)
t_val = int(num * val_percent)

train = random.sample(list, t_train)
num1 = len(train)
for i in range(num1):
    list.remove(train[i])

val_test = [i for i in list if not i in train]
val = random.sample(val_test, t_val)
num2 = len(val)
for i in range(num2):
    list.remove(val[i])

file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')

for i in train:
    name = total_xml[i][:-4] + '\n'
    file_train.write(name)

for i in val:
    name = total_xml[i][:-4] + '\n'
    file_val.write(name)

for i in list:
    name = total_xml[i][:-4] + '\n'
    file_test.write(name)

file_train.close()
file_val.close()
file_test.close()