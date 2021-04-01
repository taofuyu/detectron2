from xml.etree import ElementTree as ET
import json
import io
import os
import numpy as np

def correct_name(name, file):
    if name in ['plate','plate_ga','plate_z','palte','pl','l','t']:
        name = 'plate'
        return name
    
    if name in ['headstock','headtock', 'head', 'h', 'headstockd', 'headstockw']:
        name = 'headstock'
        return name
    
    if name in ['tailstock','tailtock','taistock','tailstockwwwww','tailstockww', 'tailstockw']:
        name = 'tailstock'
        return name
    
    if name in ['car', 'car(difficult)', 'c']:
        return 'car'
    if name in ['person', 'pedestrain']:
        return 'person'
    if name in ['cycle', 'cyclew', 'cycled', 'motorbike']:
        return 'cycle'
    if name == 'roof':
        return 'roof'
    if name == 'ear':
        return 'ear'
    if name in ['side_window', 's']:
        return 'side_window'
    if name in ['window', 'w', 'eewindow', 'windoww']:
        return 'window'
    if name in ['ignore', 'ignored','ingore', 'ingroe']:
        return 'ignore'
    print('error name: ' + name)
    print(file)
    assert(False)

def check_box(xml_file, box):
    if box[0]==0 and box[1]==0 and box[2]==0 and box[3]==0:
        print("{} all labels==0 , check it".format(xml_file))
        assert(False)
    if box[0]<0 or box[1]<0 or box[2]<0 or box[3]<0:
        print("{} label < 0 , check it".format(xml_file))
        assert(False)
    if box[0]>box[2] or box[1]>box[3]:
        print("{} label is wrong, check it".format(xml_file))
        assert(False)
    

#read boxes from txt file and cvt to a dict
def cvt_to_dict_style(box_txt_f):
    '''
    all_names = boxes.keys()
    for img in all_img:
        TODO img name and img w h
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
    '''
    with open(box_txt_f, "r") as f:
        all_box = f.readlines()
    
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

    return boxes

def decode_xinjie_box(path, xinjie_box, imgWidth, imgHeight):
    X_c = float(xinjie_box[0])
    Y_c = float(xinjie_box[1])
    W_c = float(xinjie_box[2])
    H_c = float(xinjie_box[3])

    minGtX = int((2.0*X_c*(imgWidth-1.0) + 1.0 - W_c*(imgWidth-1)) / 2.0)
    minGtY = int((2.0*Y_c*(imgHeight-1.0) + 1.0 - H_c*(imgHeight-1))/ 2.0)
    maxGtX = int((2.0*X_c*(imgWidth-1.0) - 1.0 + W_c*(imgWidth-1)) / 2.0)
    maxGtY = int((2.0*Y_c*(imgHeight-1.0) - 1.0 + H_c*(imgHeight-1))/ 2.0)
    
    if minGtX==0 and minGtY==0 and maxGtX==0 and maxGtY==0:
        print("{} has wrong label".format(path))
        return [-1,-1,-1,-1]
    if minGtX<0 or minGtY<0 or maxGtX<0 or maxGtY<0:
        print("{} has wrong label".format(path))
        return [-1,-1,-1,-1]
    if minGtX>maxGtX or minGtY>maxGtY:
        print("{} has wrong label".format(path))
        return [-1,-1,-1,-1]

    return [minGtX, minGtY, maxGtX, maxGtY]



def write_to_file(f, boxes, dir, img):
    num_box = len(boxes)
    f.write(os.path.join(dir, img))
    for box in boxes:
        to_write = " {} {} {} {} {}".format(box[0], box[1], box[2], box[3], box[4])
        f.write(to_write)
    
    f.write("\n")

def parse_xml(xml_file, class_map):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    width = float(root.find('size').find('width').text)
    height = float(root.find('size').find('height').text)

    xywh_boxes = []
    xyxy_boxes = []
    for each_object in root.findall('object'):
        box = []
        name = each_object.find('name').text
        name = correct_name(name, xml_file)
        if (name=="ignore") or (name not in class_map.keys()):
            continue
        bndbox = each_object.find('bndbox')

        x_min = float(bndbox.find('xmin').text)
        y_min = float(bndbox.find('ymin').text)
        x_max = float(bndbox.find('xmax').text)
        y_max = float(bndbox.find('ymax').text)
        
        x_min=min(width-1.,max(0.,x_min-1.))
        y_min=min(height-1.,max(0.,y_min-1.))
        x_max=min(width-1.,max(0.,x_max-1.))
        y_max=min(height-1.,max(0.,y_max-1.))

        if x_min==0 and y_min==0 and x_max==0 and y_max==0:
            print("{} has wrong label".format(xml_file))
            continue
        if x_min<0 or y_min<0 or x_max<0 or y_max<0:
            print("{} has wrong label".format(xml_file))
            continue
        if x_min>x_max or y_min>y_max:
            print("{} has wrong label".format(xml_file))
            continue

        xyxy_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max), class_map[name]])

        box.append((x_min+x_max)*0.5/(width-1.))
        box.append((y_min+y_max)*0.5/(height-1.))
        box.append((x_max-x_min+1.)/(width-1.))
        box.append((y_max-y_min+1.)/(height-1.))
        box.append(class_map[name])

        xywh_boxes.append(box)
                
    return xywh_boxes, xyxy_boxes
    
def parse_json(json_file, class_map):
    with io.open(json_file, 'r') as f:
        json_data = json.loads(f.read())
    
    width = float(json_data['imageWidth'])
    height = float(json_data['imageHeight'])

    xywh_boxes = []
    xyxy_boxes = []
    for each_object in json_data['shapes']:
        box = []
        name = each_object['label']
        name = correct_name(name, json_file)
        if name=="ignore":
            continue
        bndbox = each_object['points']
        if len(bndbox) != 2:
            print(json_file)
            print("bndbox has not two points")
            continue
        x1 = float(bndbox[0][0])
        y1 = float(bndbox[0][1])
        x2 = float(bndbox[1][0])
        y2 = float(bndbox[1][1])
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)
        
        x_min=min(width-1.,max(0.,x_min-1.))
        y_min=min(height-1.,max(0.,y_min-1.))
        x_max=min(width-1.,max(0.,x_max-1.))
        y_max=min(height-1.,max(0.,y_max-1.))

        if x_min==0 and y_min==0 and x_max==0 and y_max==0:
            print("{} has wrong label".format(json_file))
            continue
        if x_min<0 or y_min<0 or x_max<0 or y_max<0:
            print("{} has wrong label".format(json_file))
            continue
        if x_min>x_max or y_min>y_max:
            print("{} has wrong label".format(json_file))
            continue

        xyxy_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max), class_map[name]])

        box.append((x_min+x_max)*0.5/(width-1.))
        box.append((y_min+y_max)*0.5/(height-1.))
        box.append((x_max-x_min+1.)/(width-1.))
        box.append((y_max-y_min+1.)/(height-1.))
        box.append(class_map[name])

        xywh_boxes.append(box)
    
    return xywh_boxes, xyxy_boxes


def get_wenxin_arm_result(item, width, height, folder_name):
    detections = {"scores":[], "classnames":[], "boxes":[]}
    real_txt = item.replace("plate_test_dataset", folder_name) + ".txt"
    if not os.path.exists(real_txt):
        return detections["boxes"], detections["classnames"], detections["scores"]
    with io.open(real_txt, "r", encoding="gb2312") as f:
        try:
            all_lines = f.readlines()
        except UnicodeDecodeError:
            all_lines = []
    plate_cnt = 0
    for line in all_lines:
        line = line.strip("\n").strip(" ")
        if "plate" in line:
            # plate,0,rect_xywh(495,666,257,151)
            coor = line.replace("plate,"+str(plate_cnt)+",rect_xywh(", "").replace(")", "")
            x_min = float(coor.split(",")[0])
            y_min = float(coor.split(",")[1])
            w = float(coor.split(",")[2])
            h = float(coor.split(",")[3])
            
            #return to abs coor
            left = width * x_min
            top = height * y_min
            right = left + w * width - 1
            bottom = top + h * height -1

            detections["scores"].append(np.array(0.8))
            detections["classnames"].append("plate")
            detections["boxes"].append(np.array([left, top, right, bottom]))
            plate_cnt += 1
        if "ysx_reco_start" in line:
            break
    
    return detections["boxes"], detections["classnames"], detections["scores"]






