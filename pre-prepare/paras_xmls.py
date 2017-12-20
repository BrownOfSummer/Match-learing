# -*- coding:utf-8 -*-
from __future__ import print_function
"""
parse xml annotations 
"""

import os
import sys
import xml.etree.ElementTree as ET
import glob
import time

"""
Paras:
    annotation_dir: dir contains xmls
    labels: list of all classes
"""
def xmls_statistics(annotation_dir, labels):
    print('Parsing for {}'.format(labels))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(annotation_dir)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.xml')
    size = len(annotations)
    
    if not annotations:
        return []
    for i, file in enumerate(annotations):
        #progess bar
        sys.stdout.write('\r')
        percentage = 1.0 * (i + 1) / size
        progress = int(percentage * 20)
        bar_arg = [progress*'=', ' '*(20-progress), percentage*100]
        bar_arg +=[file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()

        # actual parsing
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        jpg = str(root.find('filename').text)
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        
        all = list()

        for obj in root.iter('object'):
            current = list()
            name = obj.find('name').text
            if name not in labels:
                continue

            xmlbox = obj.find('bndbox')
            xn = int(float(xmlbox.find('xmin').text))
            xx = int(float(xmlbox.find('xmax').text))
            yn = int(float(xmlbox.find('ymin').text))
            yx = int(float(xmlbox.find('ymax').text))
            current = [name, xn, yn, xx, yx]
            all += [current]
        add = [[jpg, [w, h, all]]]
        dumps += add
        in_file.close()
        #time.sleep(0.5)

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in labels:
                if current[0] in stat:
                    stat[current[0]] += 1
                else:
                    stat[current[0]] = 1
    print('\nStatistics:')
    for i in stat: print('{}: {}'.format(i, stat[i]))
    print('Dataset size: {}'.format(len(dumps)))

    os.chdir(cur_dir)
    # dump = ['0000.jpg', [640,480, [ ['label', xmin, ymin, xmax, ymax],... ] ]]
    return dumps

def parse_one_xml(xml_file):
    with open(xml_file) as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()
        jpg = root.find('filename').text
        imsize = root.find('size')
        w = imsize.find('width').text
        h = imsize.find('height').text
        depth = imsize.find('depth').text
        print('xml: {}; Size: {}'.format(jpg, [w,h,depth]))
        
        boxes = list()
        for obj in root.iter('object'):
            name = obj.find('name').text
            xmlbox = obj.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymin = int(xmlbox.find('ymin').text)
            ymax = int(xmlbox.find('ymax').text)
            
            boxes.append([name, xmin, ymin, xmax, ymax])
            print('{}: {}'.format(name, [xmin, ymin, xmax, ymax]))
        print('Totally {} boxes.'.format(len(boxes)))
        return boxes

if __name__ == '__main__':
    labels=['a','b','d','e','f','g','h','i','j','l','m','n','q','r','t','y',
            'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R',
            'S','T','U','V','W','X','Y','Z',
            '0','1','2','3','4','5','6','7','8','9']
    annotation_dir = 'TextDataset/Annotations'
    annotation_file = 'TextDataset/Annotations/0000.xml'
    #xmls_statistics(annotation_dir, labels)
    parse_one_xml(annotation_file)
