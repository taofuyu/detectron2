# -*- coding: utf-8 -*-
import os
import cv2
from PIL import Image
#from xml.etree import ElementTree as ET

classname_indexes={'plate':0,'plate_ga':0,'plate_z':0,'palte':0,'headstock':1,'headtock':1, 'tailstock':2,'tailtock':2,'taistock':2,'car':3,'side_window':4,'window':5}

def decode_xinjie_box(xinjie_box, imgWidth, imgHeight):

    X_c = float(xinjie_box[0])
    Y_c = float(xinjie_box[1])
    W_c = float(xinjie_box[2])
    H_c = float(xinjie_box[3])

    minGtX = int((2.0*X_c*(imgWidth-1.0) + 1.0 - W_c*(imgWidth-1)) / 2.0)
    minGtY = int((2.0*Y_c*(imgHeight-1.0) + 1.0 - H_c*(imgHeight-1))/ 2.0)
    maxGtX = int((2.0*X_c*(imgWidth-1.0) - 1.0 + W_c*(imgWidth-1)) / 2.0)
    maxGtY = int((2.0*Y_c*(imgHeight-1.0) - 1.0 + H_c*(imgHeight-1))/ 2.0)

    return [minGtX, minGtY, maxGtX, maxGtY]

def correct_name(name):
    plate_names = ['plate','plate_ga','plate_z','palte']
    if name in plate_names:
        name = 'plate'
        return name
    
    head_names = ['headstock','headtock']
    if name in head_names:
        name = 'headstock'
        return name
    
    tail_names = ['tailstock','tailtock','taistock']
    if name in tail_names:
        name = 'tailstock'
        return name
    
    if name == 'car':
        return 'car'
    print('new error name: ' + name)
    return name

def cvt_from_file(img_list_file, box_file, new_txt_to_create):
    with open(img_list_file, 'r', encoding="utf-8") as f:
        all_img = f.readlines()
    with open(box_file, 'r', encoding="utf-8") as f:
        all_box = f.readlines()
    all_box.sort()
    #cvt to a dict
    boxes = {}
    first_line = True
    tmp_img_name = ""
    box_line = ""
    for box in all_box:
        img_name = box.split(',')[0]
        if first_line:
            tmp_img_name = img_name
            first_line = False
        
        if img_name == tmp_img_name:
            _, cls_id, x_c, y_c, w, h = box.strip('\n').split(',')
            box_line += str(x_c)+' '+str(y_c)+' '+str(w)+' '+str(h)+' '+str(cls_id)+' '
            continue
        else:
            boxes[tmp_img_name] = box_line
            tmp_img_name = img_name

            _, cls_id, x_c, y_c, w, h = box.strip('\n').split(',')
            box_line = str(x_c)+' '+str(y_c)+' '+str(w)+' '+str(h)+' '+str(cls_id)+' '
    boxes[tmp_img_name] = box_line #last img

    new_file = open(new_txt_to_create, "w+", encoding="utf-8")
    all_names = boxes.keys()
    for img in all_img:
        img = img.split(' ')[0]
        img = img.strip('\n').strip(' ').replace('/data/taofuyu/tao_dataset/', "")
        img = '/detectron2/datasets/' + img
        src_img = cv2.imread(img)
        if src_img is None:
            print(img)
            continue
        img_h, img_w = src_img.shape[:2]
        new_file.write(img + ' ')

        img_name = img.split('/')[-1][0:-4]
        if img_name in all_names:
            box_line = boxes[img_name]
            box_line = box_line.strip(" ").split(" ")
        else:
            new_file.write('\n')
            continue
        num_boxes = len(box_line) // 5
        final_boxes = []
        for i in range(num_boxes): #one object
            bbox = []
            x_c = box_line[0 + 5*i]
            y_c = box_line[1 + 5*i]
            w = box_line[2 + 5*i]
            h = box_line[3 + 5*i]
            label = box_line[4 + 5*i]

            x_min, y_min, x_max, y_max = decode_xinjie_box([x_c,y_c,w,h], img_w, img_h)

            bbox.append(x_min)
            bbox.append(y_min)
            bbox.append(x_max)
            bbox.append(y_max)
            bbox.append(label)

            final_boxes.append(bbox)

        for b in final_boxes:
            if not (b[4]=='6' or b[4]=='7'): # dont need roof and cycle
                new_file.write(str(b[0])+' '+str(b[1])+' '+str(b[2])+' '+str(b[3])+' '+str(b[4]+' '))
        new_file.write('\n')
    
    new_file.close()

if __name__ == '__main__':
    img_list_file = '/detectron2/datasets/txt/high_roadside/high_roadside_roof_cycle_soft.txt'
    box_file = '/detectron2/datasets/txt/high_roadside/high_roadside_roof_cycle_box.txt'
    new_txt_to_create = '/detectron2/datasets/txt/high_roadside/train_high_roadisde_imglist.txt'
    cvt_from_file(img_list_file, box_file, new_txt_to_create)
 
    