import xml.etree.ElementTree as ET
from os import getcwd
import shutil

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(datapath, year, image_id, list_file):
    in_file = open(datapath+'/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

def write_annotation(data_path):
    print(data_path)
    for year, image_set in sets:
        image_ids = open(data_path+'/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        list_file = open('%s_%s.txt'%(year, image_set), 'w')
        for image_id in image_ids:
            list_file.write(data_path+'/VOC%s/JPEGImages/%s.jpg'%(year, image_id))
            convert_annotation(data_path, year, image_id, list_file)
            list_file.write('\n')
        list_file.close()
    
    with open('train.txt','wb') as wfd:
        for f in ['2007_train.txt','2007_val.txt','2007_test.txt']:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)

