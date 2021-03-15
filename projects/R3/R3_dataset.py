# -*- coding: utf-8 -*-
'''
@author: taofuyu
@description: the dataset func for high_roadside project
@note:
The img list should be like:

    .../path/img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...

eg: .../path/any.jpg 22 34 89 135 1 78 45 564 348 3 ...
'''

import os
import logging
from PIL import Image
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import BoxMode

def check_gt_box(annos, file_name):
    for anno in annos:
        #zero
        if anno["bbox"][0] == 0 and anno["bbox"][1] == 0 and anno["bbox"][2] == 0 and anno["bbox"][3] == 0:
            print("{}, in-valid gt box, check it".format(file_name))
            print(anno["bbox"])
            assert(False)
        #neg
        if anno["bbox"][0] < 0 or anno["bbox"][1] < 0 or anno["bbox"][2] < 0 or anno["bbox"][3] < 0:
            print("{}, in-valid gt box, check it".format(file_name))
            print(anno["bbox"])
            assert(False)
        #big small
        if anno["bbox"][0] > anno["bbox"][2] or anno["bbox"][1] > anno["bbox"][3]:
            print("{}, in-valid gt box, check it".format(file_name))
            print(anno["bbox"])
            assert(False)
    #same box


def R3_dataset_function(img_list):
    '''
    Use img_list_file to generate a stdandard detectron2 data list 
    '''
    logger = logging.getLogger("detectron2")

    #read img list file
    assert(os.path.exists(img_list)), 'assert failed, {} not exists !'.format(img_list)
    with open(img_list, 'r') as f:
        lines = f.readlines()
    num_imgs = len(lines)
    logger.info('total {} imgs to train'.format(num_imgs))

    #cvt img list to detectron2 standard format:
    det2_dataset = []
    img_id = 0
    for line in lines: #one line contains all info about an img
        splited = line.strip('\n').strip(' ').split(' ')
        file_name = splited[0]
        img = Image.open(file_name)
        img_w, img_h = img.size

        img_dict = {}
        annotations = []
        num_boxes = ( len(splited)-1 ) // 5
        if num_boxes == 0:
            continue
        for i in range(num_boxes): #one object
            anno = {}
            
            x_min = float(splited[1 + 5*i])
            y_min = float(splited[2 + 5*i])
            x_max = float(splited[3 + 5*i])
            y_max = float(splited[4 + 5*i])
            label = int(splited[5 + 5*i])

            if x_min == 0 and y_min == 0 and x_max == 0 and y_max == 0:
                continue
            if x_min < 0 or y_min < 0 or x_max < 0 or y_max < 0:
                continue
            if x_min > x_max or y_min > y_max:
                continue
            
            anno['bbox'] = [x_min, y_min, x_max, y_max]
            anno['bbox_mode'] = BoxMode.XYXY_ABS
            anno['category_id'] = label

            annotations.append(anno)
        
        check_gt_box(annotations, file_name)

        img_dict['file_name'] = file_name
        img_dict['width'] = img_w
        img_dict['height'] = img_h
        img_dict['image_id'] = img_id
        img_dict['annotations'] = annotations
        img_id += 1
        
        det2_dataset.append(img_dict)
    
    return det2_dataset

for phase in ['train', 'val']:
    DatasetCatalog.register('R3_dataset_' + phase, lambda phase=phase: R3_dataset_function(\
                            '/data/taofuyu/tao_txt/R3/det2/'+phase+'_R3.txt'))
    MetadataCatalog.get('R3_dataset_' + phase).set(thing_classes=['plate','head','tail','car'], \
                            thing_colors=['red', 'yellow', 'brown', 'green'])
