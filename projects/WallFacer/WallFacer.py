# -*- coding: utf-8 -*-

import os
import cv2
import random
import json
from tools.base_func import correct_name, cvt_to_dict_style, decode_xinjie_box, write_to_file, parse_xml, parse_json, parse_json_sim
from detectron2.data.detection_utils import load_cfg, annotations_to_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from detectron2.structures.boxes import BoxMode

class WallFacer:
    def __init__(self):
        '''
        class_map: label index in my style
        img_path: the folder which contains imgs to generate txt file
        xywh_txt_to_create: abs path to generate a xywh format file
        xyxy_txt_to_create: abs path to generate a xyxy format file
        '''
        print("init WallFacer")
    
    def cvt_xml_json_2_txt(self, class_map, img_path, xywh_txt_to_create, xyxy_txt_to_create):
        '''
        Given an img folder, read each img's xml file or json file, cvt these gt file to txt file format(details in README.md).
        '''
        xywh_file = open(xywh_txt_to_create, "w+")
        xyxy_file = open(xyxy_txt_to_create, "w+")
        #loop each child folder
        for dir, _, files in os.walk(img_path):
            if len(files) == 0:
                continue
            #loop each img
            for img in files:
                if not img[-3:] in ["jpg", "JPG", "bmp", "png"]:
                    continue
                
                img_name = img[0:-4]
                if (" " in img_name) or ("," in img_name):
                    print("{} or {} in image name".format(" ", ","))
                    print(os.path.join(dir,img))
                    assert(False)
                
                #check gt file format
                xml_file = os.path.join(dir, img.replace(img[-3:], "xml"))
                json_file = os.path.join(dir, img.replace(img[-3:], "json"))
                if os.path.exists(xml_file):
                    xywh_boxes, xyxy_boxes = parse_xml(xml_file, class_map)
                elif os.path.exists(json_file):
                    xywh_boxes, xyxy_boxes = parse_json(json_file, class_map)
                    #xywh_boxes, xyxy_boxes = parse_json_sim(json_file, class_map)
                else: #no box in this img
                    xywh_file.write(os.path.join(dir, img) + "\n")
                    xyxy_file.write(os.path.join(dir, img) + "\n")
                    continue
                
                #xywh
                write_to_file(xywh_file, xywh_boxes, dir, img)
                #xyxy
                write_to_file(xyxy_file, xyxy_boxes, dir, img)

        xywh_file.close()
        xyxy_file.close()
    
    def cvt_txt_2_txt(self, class_cvt_map, img_path, caffe_box_file, xywh_txt_to_create, xyxy_txt_to_create):
        '''Given an img folder, read each img's gt from caffe gt txt file, cvt these gt file to txt file format(details in README.md)
        
        class_cvt_map: for different project, caffe gt file has written corresponding label index, convert it to label index mentioned in README.md
        img_path: img folder
        caffe_box_file: box file used when training caffe
        '''
        xywh_file = open(xywh_txt_to_create, "w+")
        xyxy_file = open(xyxy_txt_to_create, "w+")

        boxes = cvt_to_dict_style(caffe_box_file)
        all_names = boxes.keys()

        #loop each folder
        for dir, _, files in os.walk(img_path):
            if len(files) == 0:
                continue
            #loop each img
            for img in files:
                if not img[-3:] in ["jpg", "JPG", "bmp", "png"]:
                    continue
                
                src_img = cv2.imread(os.path.join(dir, img))
                if src_img is None:
                    continue
                img_h, img_w, _ = src_img.shape
                img_name = img[0:-4]
                if (" " in img_name) or ("," in img_name):
                    print("{} or {} in image name".format(" ", ","))
                    assert(False)
                
                if img_name in all_names:
                    box_line = boxes[img_name]
                    box_line = box_line.strip(" ").split(" ")
                else:
                    xywh_file.write(os.path.join(dir, img) + "\n")
                    xyxy_file.write(os.path.join(dir, img) + "\n")
                    continue
                num_boxes = len(box_line) // 5
                xywh_boxes = []
                xyxy_boxes = []
                for i in range(num_boxes): #one object
                    bbox = []
                    x_c = box_line[0 + 5*i]
                    y_c = box_line[1 + 5*i]
                    w = box_line[2 + 5*i]
                    h = box_line[3 + 5*i]
                    label = box_line[4 + 5*i]
                    label = class_cvt_map[label]

                    x_min, y_min, x_max, y_max = decode_xinjie_box(os.path.join(dir, img) ,[x_c,y_c,w,h], img_w, img_h)
                    if x_min == -1:
                        continue
                    xywh_boxes.append([x_c, y_c, w, h, label])
                    xyxy_boxes.append([x_min, y_min, x_max, y_max, label])
                
                #xywh
                write_to_file(xywh_file, xywh_boxes, dir, img)
                #xyxy
                write_to_file(xyxy_file, xyxy_boxes, dir, img)

        xywh_file.close()
        xyxy_file.close()

    def merge_train_imglist(self, config):
        '''merge all the train_imglist to a total list. This is different from generate_train_imglist()
        
        config: config file lies in .../gt/config.yaml
        '''
        merge_config = load_cfg(config)
        for total_list in merge_config["total_train_imglist_to_create"]:
            with open(total_list, "w+") as total_f:
                for each_list in merge_config["merge_list"]:
                    with open(total_list.replace("all_train_imglist", each_list), "r") as each_f:
                        all_l = each_f.readlines()
                    num_imgs = len(all_l)
                    for i in range(int(num_imgs * merge_config["merge_list"][each_list])):
                        total_f.write(all_l[i])

    def get_keys(self, d, value):
        return [k for k,v in d.items() if v == value]

    def generate_train_imglist(self, config, project_class_map, raw_class_map, img_id):
        '''use train_imglist.txt in "gt" folder to generate a train file for caffe or det2
        config: config file to adjust some datasets to generate train files
        project_class_map: class index in this project. 
        '''
        cfg = load_cfg(config)

        if cfg["generate_caffe"]:
            caffe_imglist = open(cfg["caffe_imglist_to_create"], "w+")
            caffe_box = open(cfg["caffe_box_to_create"], "w+")

            has_writen_img = []
            for dataset in cfg["merge_list"]:
                print(dataset)
                img_list_file = cfg["merge_list"][dataset][0].replace("xyxy", "xywh")
                times = cfg["merge_list"][dataset][1]
                add_info = cfg["merge_list"][dataset][2]
                
                final_lines = self.get_final_lines(img_list_file, times)
                
                for img in final_lines:
                    #img info
                    img = img.strip("\n").strip(" ")
                    img_path = img.split(" ")[0]
                    img_file = img_path.split("/")[-1]
                    img_name = img_file[0:-4]

                    #get gt box
                    box_info = img.split(" ")[1:]
                    num_boxes = len(box_info) // 5
                    xywh_boxes = []
                    for i in range(num_boxes): #one object
                        x_c = box_info[0 + 5*i]
                        y_c = box_info[1 + 5*i]
                        w = box_info[2 + 5*i]
                        h = box_info[3 + 5*i]
                        label = box_info[4 + 5*i]
                        if not label in project_class_map.keys():
                            continue
                        
                        label = project_class_map[label]

                        xywh_boxes.append([x_c, y_c, w, h, label])
                    
                    #write to file
                    caffe_imglist.write("{} {}\n".format(img_path, add_info))
                    if not img_name in has_writen_img: #box only need to write once for same img
                        for bbox in xywh_boxes:
                            caffe_box.write("{},{},{},{},{},{}\n".format(img_name, bbox[4], bbox[0], bbox[1], bbox[2], bbox[3]))
                    
                    has_writen_img.append(img_name)
            
            caffe_imglist.close()
            caffe_box.close()

        if cfg["generate_det2"]:
            det2_imglist = open(cfg["det2_imglist_to_create"], "w+")

            for dataset in cfg["merge_list"]:
                print(dataset)
                img_list_file = cfg["merge_list"][dataset][0]
                times = cfg["merge_list"][dataset][1]
            
                final_lines = self.get_final_lines(img_list_file, times)
                for img in final_lines:
                    #img info
                    img = img.strip("\n").strip(" ")
                    img_path = img.split(" ")[0]
                    img_file = img_path.split("/")[-1]
                    img_name = img_file[0:-4]

                    #get gt box
                    box_info = img.split(" ")[1:]
                    num_boxes = len(box_info) // 5
                    xyxy_boxes = []
                    for i in range(num_boxes): #one object
                        x_min = box_info[0 + 5*i]
                        y_min = box_info[1 + 5*i]
                        x_max = box_info[2 + 5*i]
                        y_max = box_info[3 + 5*i]
                        label = box_info[4 + 5*i]
                        if not label in project_class_map.keys():
                            continue
                        
                        label = project_class_map[label]

                        xyxy_boxes.append([x_min, y_min, x_max, y_max, label])
                    
                    #write to file
                    det2_imglist.write("{}".format(img_path))
                    for bbox in xyxy_boxes:
                        det2_imglist.write(" {} {} {} {} {}".format(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]))
                    det2_imglist.write("\n")
            
            det2_imglist.close()

        if cfg["generate_coco"]:
            to_dump_dict = {}
            to_dump_dict["images"] = []
            to_dump_dict["annotations"] = []
            to_dump_dict["categories"] = []
            print("please check category name and order !!!")
            assert(True)
            #order and name must equal train config "classes"
            to_dump_dict["categories"].append({"id":0, "name": "plate"})
            to_dump_dict["categories"].append({"id":1, "name": "headstock"})
            to_dump_dict["categories"].append({"id":2, "name": "tailstock"})
            to_dump_dict["categories"].append({"id":3, "name": "car"})
            to_dump_dict["categories"].append({"id":4, "name": "side_window"})
            to_dump_dict["categories"].append({"id":5, "name": "window"})
            to_dump_dict["categories"].append({"id":6, "name": "roof"})
            to_dump_dict["categories"].append({"id":7, "name": "cycle"})
            #to_dump_dict["categories"].append({"id":8, "name": "cycle"})
            
            for dataset in cfg["merge_list"]:
                print(dataset)
                img_list_file = cfg["merge_list"][dataset][0]
                times = cfg["merge_list"][dataset][1]
            
                final_lines = self.get_final_lines(img_list_file, times)
                for img in final_lines:
                    #img info
                    img = img.strip("\n").strip(" ")
                    img_path = img.split(" ")[0]
                    img_file = img_path.split("/")[-1]
                    img_name = img_file[0:-4]
                    src_img = cv2.imread(img_path)
                    if src_img is None:
                        continue
                    img_h, img_w, _ = src_img.shape

                    to_dump_dict["images"].append({"file_name": img_path, \
                                                   "height": img_h,\
                                                   "width": img_w,\
                                                   "id": img_id})

                    #get gt box
                    box_info = img.split(" ")[1:]
                    num_boxes = len(box_info) // 5
                    for i in range(num_boxes): #one object
                        x_min = float(box_info[0 + 5*i])
                        y_min = float(box_info[1 + 5*i])
                        x_max = float(box_info[2 + 5*i])
                        y_max = float(box_info[3 + 5*i])
                        box_w = x_max-x_min
                        box_h = y_max-y_min
                        label = box_info[4 + 5*i]
                        if not label in project_class_map.keys():
                            continue
                        
                        label = int(project_class_map[label])

                        to_dump_dict["annotations"].append({"iscrowd": 0,\
                                                            "image_id": img_id,\
                                                            "bbox": [x_min, y_min, box_w, box_h],\
                                                            "category_id": label,\
                                                            "id": int(len(to_dump_dict['annotations']) + 1),\
                                                            "area": box_w*box_h})
                    
                    img_id += 1
            
            json.dump(to_dump_dict, open(cfg["coco_json_to_create"], 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def get_final_lines(self, img_list_file, times):
        '''repeat or get part of img_list_file according to times
        '''
        with open(img_list_file, "r") as f:
            all_lines = f.readlines()
            print("{} imgs".format(len(all_lines)))
        
        final_lines = []
        int_times = str(times).split(".")[0]
        float_times = str(times).split(".")[1]

        for i in range(int(int_times)):
            for l in all_lines:
                final_lines.append(l)
        
        float_times = int(len(all_lines) * float("0." + float_times))
        
        random.shuffle(all_lines)
        for i in range(float_times):
            final_lines.append(all_lines[i])
        
        return final_lines

    def draw_to_check_gt(self, imglist, save_path):
        with open(imglist, "r") as f:
            all_lines = f.readlines()
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        color = {"0": "red", "1": "yellow", "2": "brown", "3": "green", "4": "olive", "5": "orange", "6": "purple", "7": "cyan", "8":"blue"}
        for l in all_lines:
            img_path = l.split(" ")[0]
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            image = image[:, :, ::-1]
            visualizer = Visualizer(image)

            box_info = l.strip("\n").split(" ")[1:]
            num_boxes = len(box_info) // 5
            bboxes = []
            for i in range(num_boxes): #one object
                x_min = box_info[0 + 5*i]
                y_min = box_info[1 + 5*i]
                x_max = box_info[2 + 5*i]
                y_max = box_info[3 + 5*i]
                label = box_info[4 + 5*i]

                vis_output = visualizer.draw_box([float(x_min), float(y_min), float(x_max), float(y_max)], alpha=1, edge_color=color[str(label)])

            img_name = img_path.split("/")[-1]
            vis_output.save(os.path.join(save_path, img_name))

if __name__ == "__main__":
    wall_facer = WallFacer()

    ###---------- generate new gt txt file ----------###
    if False:
        class_map = {"plate":0, "headstock":1, "tailstock":2, "car":3, "side_window":4, "window":5, "roof":6, "person":7, "cycle":8}

        img_path = "/data/taofuyu/tao_dataset/high_roadside/test/"
        xywh_txt_to_create = "/data/taofuyu/tao_dataset/high_roadside/gt/xywh/test_imglist.txt"
        xyxy_txt_to_create = "/data/taofuyu/tao_dataset/high_roadside/gt/xyxy/test_imglist.txt"

        wall_facer.cvt_xml_json_2_txt(class_map, img_path, xywh_txt_to_create, xyxy_txt_to_create)

        # caffe_box_file = "/data/taofuyu/tao_txt/monster/monster_all_t4_taostyle_box.txt"
        # #caffe_box_file = "/data/taofuyu/tao_dataset/det_plate/gt/detect_monster_cityscapes_t3_box.txt"
        # cvt_map = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"7", "7":"8"} #{cls idx in project:cls idx in README, ...}
        # wall_facer.cvt_txt_2_txt(cvt_map, img_path, caffe_box_file, xywh_txt_to_create, xyxy_txt_to_create)

    ###---------- merge gt txt file ----------###
    if False:
        config_file = "/data/taofuyu/tao_dataset/road/gt/config.yaml"
        wall_facer.merge_train_imglist(config_file)

    ###---------- generate train file ----------###
    if True:
        config_file = "/data/taofuyu/models/dataset_config/track_det_config.yaml"
        raw_class_map = {"plate":0, "headstock":1, "tailstock":2, "car":3, "side_window":4, "window":5, "roof":6, "person":7, "cycle":8}
        project_class_map = {"0":"0","1":"1","2":"2","3":"3","4":"4","5":"5","6":"6", "8":"7"} #{cls idx in gt file:cls idx in project, ...}
        img_id = 0
        wall_facer.generate_train_imglist(config_file, project_class_map, raw_class_map, img_id)

    ###---------- draw gt ----------###
    if False:
        draw_list = "/data/taofuyu/tao_dataset/high_roadside/gt/xyxy/patch_five_auto_train_imglist.txt"
        save_path = "/data/taofuyu/tao_dataset/high_roadside/draw_gt/"
        wall_facer.draw_to_check_gt(draw_list, save_path)
