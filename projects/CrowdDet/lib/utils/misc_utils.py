import os
import json
import numpy as np

def load_img(image_path):
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return img

def load_json_lines(fpath):
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records

def load_txt_lines(fpath):
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [line.strip('\n') for line in lines]
    return records

def save_json_lines(content,fpath):
    with open(fpath,'w') as fid:
        for db in content:
            line = json.dumps(db)+'\n'
            fid.write(line)

def device_parser(str_device):
    if '-' in str_device:
        device_id = str_device.split('-')
        device_id = [i for i in range(int(device_id[0]), int(device_id[1])+1)]
    else:
        device_id = [int(str_device)]
    return device_id

def ensure_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def xyxy_to_xywh(boxes):
    assert boxes.shape[1]>=4
    boxes[:, 2:4] -= boxes[:,:2]
    return boxes

def xywh_to_xyxy(boxes):
    assert boxes.shape[1]>=4
    boxes[:, 2:4] += boxes[:,:2]
    return boxes

def load_bboxes(dict_input, key_name, key_box, key_score=None, key_tag=None):
    assert key_name in dict_input
    if len(dict_input[key_name]) < 1:
        return np.empty([0, 5])
    else:
        assert key_box in dict_input[key_name][0]
        if key_score:
            assert key_score in dict_input[key_name][0]
        if key_tag:
            assert key_tag in dict_input[key_name][0]
    if key_score:
        if key_tag:
            bboxes = np.vstack([np.hstack((rb[key_box], rb[key_score], rb[key_tag])) for rb in dict_input[key_name]])
        else:
            bboxes = np.vstack([np.hstack((rb[key_box], rb[key_score])) for rb in dict_input[key_name]])
    else:
        if key_tag:
            bboxes = np.vstack([np.hstack((rb[key_box], rb[key_tag])) for rb in dict_input[key_name]])
        else:
            bboxes = np.vstack([rb[key_box] for rb in dict_input[key_name]])
    return bboxes

def load_masks(dict_input, key_name, key_box):
    assert key_name in dict_input
    if len(dict_input[key_name]) < 1:
        return np.empty([0, 28, 28])
    else:
        assert key_box in dict_input[key_name][0]
    masks = np.array([rb[key_box] for rb in dict_input[key_name]])
    return masks

def load_gt(dict_input, key_name, key_box, class_names):
    assert key_name in dict_input
    if len(dict_input[key_name]) < 1:
        return np.empty([0, 5])
    else:
        assert key_box in dict_input[key_name][0]
    bbox = []
    # gtbox{
    # "tag" : "person" or "mask", 
    # "vbox": [x, y, w, h],
    # "fbox": [x, y, w, h],
    # "hbox": [x, y, w, h],
    # "extra" : extra, 
    # "head_attr" : head_attr, 
    # }
    for rb in dict_input[key_name]:
        if rb['tag'] in class_names:
            tag = class_names.index(rb['tag'])
        else:
            tag = -1
        if 'extra' in rb:
            if 'ignore' in rb['extra']:
                if rb['extra']['ignore'] != 0:
                    tag = -1
        bbox.append(np.hstack((rb[key_box], tag)))
    bboxes = np.vstack(bbox).astype(np.float64)
    return bboxes

def load_txt_gt(record):
    bbox = []
    is_empty_img = False
    label = record.split(" ")[1:]
    num_boxes = len(label) // 5
    for i in range(num_boxes):
        x_min = int(label[5*i])
        y_min = int(label[5*i + 1])
        x_max = int(label[5*i + 2])
        y_max = int(label[5*i + 3])
        tag = int(label[5*i + 4])
        bbox.append(np.hstack(([x_min, y_min, x_max-x_min+1, y_max-y_min+1], tag)))
    if len(bbox) == 0:
        is_empty_img = True
        line = "/data/taofuyu/tao_dataset/high_roadside/test/03_20200110_083001_mp4_20200122_151732_1700.jpg 0 1775 130 1887 0 5 1150 690 2000 1 1348 677 1715 1045 1 2173 0 2383 515 7 0 647 753 1175 5 1240 280 1715 555 5 1425 60 2015 280 5 770 377 1348 1120 4 1733 230 1880 545 4 2023 22 2193 300 4 0 187 1363 2077 3 1168 145 1983 1110 3 1418 5 2283 762 3 2156 0 2570 417 3"
        label = line.split(" ")[1:]
        num_boxes = len(label) // 5
        for i in range(num_boxes):
            x_min = int(label[5*i])
            y_min = int(label[5*i + 1])
            x_max = int(label[5*i + 2])
            y_max = int(label[5*i + 3])
            tag = int(label[5*i + 4])
            bbox.append(np.hstack(([x_min, y_min, x_max-x_min+1, y_max-y_min+1], tag)))
    bboxes = np.vstack(bbox).astype(np.float64)
    return bboxes, is_empty_img



def boxes_dump(boxes, is_gt):
    result = []
    boxes = boxes.tolist()
    for box in boxes:
        if is_gt:
            box_dict = {}
            box_dict['box'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            box_dict['tag'] = box[-1]
            result.append(box_dict)
        else:
            box_dict = {}
            box_dict['box'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            box_dict['tag'] = 1
            box_dict['score'] = box[-1]
            result.append(box_dict)
    return result

def clip_boundary(boxes,height,width):
    assert boxes.shape[-1]>=4
    boxes[:,0] = np.minimum(np.maximum(boxes[:,0],0), width - 1)
    boxes[:,1] = np.minimum(np.maximum(boxes[:,1],0), height - 1)
    boxes[:,2] = np.maximum(np.minimum(boxes[:,2],width), 0)
    boxes[:,3] = np.maximum(np.minimum(boxes[:,3],height), 0)
    return boxes

