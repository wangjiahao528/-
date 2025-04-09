# 将VOC数据集按照txt文件转换并划分为YOLO格式

import xml.etree.ElementTree as ET
import os
import shutil

path = r'C:\Users\28244\Desktop\yolov8\dataset\3'
imagetype = '.jpg'
sets = ['train', 'test', 'val']
classes = []


def gen_classes(image_ids):
    global classes
    for image_id in image_ids:
        in_file = open(f'{path}/Annotations/{image_id}.xml', encoding='utf-8')  # Specify UTF-8 encoding
        tree = ET.parse(in_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            if cls_name not in classes:
                classes.append(cls_name)


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_set, image_ids):
    for image_id in image_ids:
        in_file = open(f'{path}/Annotations/{image_id}.xml', encoding='utf-8')  # Specify UTF-8 encoding
        out_file = open(f'{path}/{image_set}/labels/{image_id}.txt', 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            # Check if 'difficult' tag exists
            difficult = obj.find('difficult')
            if difficult is not None and int(difficult.text) == 1:
                continue  # Skip if the object is marked as difficult

            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(f'{cls_id} ' + ' '.join([str(a) for a in bb]) + '\n')
        out_file.close()


for image_set in sets:
    # Create directories if they don't exist
    os.makedirs(f'{path}/{image_set}/labels/', exist_ok=True)
    os.makedirs(f'{path}/{image_set}/images/', exist_ok=True)

    image_ids = open(f'{path}/ImageSets/Main/{image_set}.txt').read().strip().split()

    # Generate the classes list based on all images in the set
    gen_classes(image_ids)

    # Convert annotations and copy images
    convert_annotation(image_set, image_ids)
    for image_id in image_ids:
        shutil.copy(f'{path}/JPEGImages/{image_id}{imagetype}', f'{path}/{image_set}/images/')

    # Save the classes to a file
    with open(f'{path}/{image_set}/labels/classes.txt', 'w') as classes_file:
        classes_file.write("\n".join(classes))

print('Processing completed.')