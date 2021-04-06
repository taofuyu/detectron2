# -*- coding: utf-8 -*-
'''
main test code
'''

from detectron2.data.detection_utils import load_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from adet.modeling import *
from adet.config import *

from tools.image_reader import ImageReader
from tools.project_model import *
from tools.base_func import get_wenxin_arm_result, parse_xml
import cv2
import os
import io
import json
import shutil
import torch
import sys
import numpy as np
sys.path.append("/data/taofuyu/repos/detectron2/projects/")
from R3 import R3_dataset_function
from HighRoadside.highroadside_model import *
from detectron2.engine.defaults import DefaultPredictor

class WallBreaker:
    '''compute some metrics for given images
    '''
    def __init__(self, cfg):
        self.class_to_test = cfg["test_class"]
        self.conf_thr = cfg["conf_thr"]
        self.iou_thr = cfg["iou_thr"]
        self.draw_img_path = cfg["draw_img_path"]
        self.save_img_path = cfg["save_img_path"]
        self.offset = 1.0
        self.color = cfg["color"]

        self.class_map = {}
        for cls in self.class_to_test:
            self.class_map[cls] = cls

        if not os.path.exists(self.save_img_path):
            os.makedirs(self.save_img_path)

    def compute_det_metrics(self, img_list, result_on_dataset):
        #init
        total_num = {}
        total_iou = {}
        FP = {}
        predict = {}
        total_pre = {}
        total_score = {}
        for each_cls in self.class_to_test:
            total_num[each_cls] = 0
            total_iou[each_cls] = 0
            FP[each_cls] = 0
            predict[each_cls] = 0
            total_pre[each_cls] = 0
            total_score[each_cls] = 0
        
        for i,img in enumerate(img_list):
            #img info
            img_path = img_list[i]
            det_result = result_on_dataset[img][0:-1] #[[x,y,x,y], 'plate', 0.6]
            img_shape = result_on_dataset[img][-1]
            height, width, _  = img_shape
            
            #get gt
            all_class_gt = self.get_det_gt(img_path, height, width)
            for each_cls in self.class_to_test:
                #start computing
                total_num[each_cls] += len(all_class_gt[each_cls]["box_gt"])

                count_fp=[]
                for j in range(len(det_result[0])): #each detection box
                    if not float(det_result[2][j]) >= self.conf_thr:
                        continue
                    score = det_result[2][j]
                    category = det_result[1][j]
                    pre_box = det_result[0][j]

                    iou = 0
                    if not category==each_cls:
                        continue
                    total_pre[each_cls] += 1
                    total_score[each_cls] += score

                    #for this det box, if there is no gt, it is a FP
                    if len(all_class_gt[each_cls]["count_gt"])==0 or sum(all_class_gt[each_cls]["count_gt"])==len(all_class_gt[each_cls]["count_gt"]):
                        count_fp.append(1)
                    else:
                        has_matched = False
                        #loop each gt box to find a match, if no match, it is a FP
                        for tt in range(len(all_class_gt[each_cls]["count_gt"])):
                            if all_class_gt[each_cls]["count_gt"][tt] == 0:
                                iou = self.IOU(pre_box, all_class_gt[each_cls]["box_gt"][tt])
                                cover = self.COVER(pre_box, all_class_gt[each_cls]["box_gt"][tt])
                                if iou >= self.iou_thr or (iou > 0.3 and cover > 0.9):
                                    all_class_gt[each_cls]["count_gt"][tt] = 1 #flag it, means TP
                                    has_matched = True
                                    total_iou[each_cls] = total_iou[each_cls] + iou
                                    break
                        if not has_matched:
                            count_fp.append(1)

                predict[each_cls] += sum(all_class_gt[each_cls]["count_gt"])
                FP[each_cls] += sum(count_fp)

        #return each class's metircs
        avg_iou = {}
        avg_score = {}
        recall = {}
        error = {}
        for each_cls in self.class_to_test:
            avg_iou[each_cls] = float(total_iou[each_cls]) / float(max(1, predict[each_cls]))
            avg_score[each_cls] = float(total_score[each_cls]) / float(max(1, total_pre[each_cls]))
            recall[each_cls] = float(predict[each_cls]) / float(max(1, total_num[each_cls]))
            error[each_cls] = float(FP[each_cls]) / float(max(1, total_pre[each_cls]))
        
        return total_num, avg_iou, avg_score, recall, error

    def compute_cls_metrics(self, class_map, dataset, result_on_dataset):
        gt_category = dataset.strip("\n").strip(" ").split("/")[-1]
        total_img = len(result_on_dataset)
        right_cnt = 0
        for result in result_on_dataset:
            pred_category = class_map[str(result_on_dataset[result])]
            if pred_category == gt_category:
                right_cnt += 1
        return right_cnt, float(right_cnt) / total_img

    def compute_classification_classmap(self, cfg):
        class_name_file = cfg["class_name_file"]
        class_map = {}
        with open(class_name_file, "r") as f:
            all_lines = f.readlines()
            for i,line in enumerate(all_lines):
                line = line.strip("\n").strip(" ")
                class_map[str(i)] = line
        
        return class_map


    def get_det_gt(self, img_path, height, width):
        #init
        all_class_gt = {} # {"plate":{"box_gt":[], "count_gt":[]},   "head":{"box_gt":[], "count_gt":[]}}
        for each_cls in self.class_to_test:
            all_class_gt[each_cls] = {"box_gt":[], "count_gt":[]}

        img_format = img_path[-3:]
        #load xml
        if os.path.exists(img_path.replace(img_format, "xml")):#read xml and return
            _, xyxy_boxes = parse_xml(img_path.replace(img_format, "xml"), self.class_map)
            for box in xyxy_boxes:
                each_cls = box[-1]
                all_class_gt[each_cls]["box_gt"].append(box[0:-1])
                all_class_gt[each_cls]["count_gt"].append(0)
            
            return all_class_gt

        #load json
        anno = img_path.replace(img_format, 'json')
        if not os.path.exists(anno):
            return all_class_gt
        with io.open(anno, 'r', encoding='utf8') as f:
            data = json.loads(f.read())
        
        #load gt from json for each class
        for each_cls in self.class_to_test:
            for i in range(data[each_cls.replace("stock", "") + '_cnt']): #because json file write "head"
                root = data[each_cls.replace("stock", "") + "_" + str(i)]
                xmin = root['x_min']
                ymin = root['y_min']
                xmax = root['x_max']
                ymax = root['y_max']

                w = xmax - xmin
                h = ymax - ymin

                xmin = (0, xmin)[xmin > 0]
                xmax = (width-1, xmax)[xmax < width]
                ymin = (0, ymin)[ymin > 0]
                ymax = (height-1, ymax)[ymax < height]

                all_class_gt[each_cls]["box_gt"].append([xmin, ymin, xmax, ymax])
                all_class_gt[each_cls]["count_gt"].append(0)

        return all_class_gt

    def draw_img(self, img_path, boxes, classnames, scores):
        if len(boxes) == 0:
            img_name = img_path.split("/")[-1]
            shutil.copy(img_path, os.path.join(self.save_img_path, img_name))
            return
        src_img = cv2.imread(img_path)
        if src_img is None:
            return
        
        src_img = src_img[:, :, ::-1]
        img_h, img_w, _ = src_img.shape
        
        vis = Visualizer(src_img)
        for i in range(len(boxes)):
            x_min = float(boxes[i][0])
            y_min = float(boxes[i][1])
            x_max = float(boxes[i][2])
            y_max = float(boxes[i][3])
            box = [x_min, y_min, x_max, y_max]

            vis_output = vis.draw_box(box_coord=box, alpha=1, edge_color=self.color[classnames[i]])
            text = "{},{:.4f}".format(classnames[i], scores[i]) 
            vis_output = vis.draw_text(text=text, position=[x_min, max(0, y_min-5)], font_size=10, color=self.color[classnames[i]])

        img_name = img_path.split("/")[-1]
        vis_output.save(os.path.join(self.save_img_path, img_name))

    def save_to_txt(self, img_path, result_file, boxes, classnames, scores):
        if len(boxes) == 0:
            return
        
        mapp = {"plate":"0", "headstock":"1", "tailstock":"2", "car":"3"}
        to_write = img_path
        for i in range(len(boxes)):
            x_min = int(boxes[i][0])
            y_min = int(boxes[i][1])
            x_max = int(boxes[i][2])
            y_max = int(boxes[i][3])
            label = mapp[classnames[i]]
            to_write += " {} {} {} {} {}".format(x_min, y_min, x_max, y_max, label)
        
        to_write += "\n"

        result_file.write(to_write)

    def save_to_excel(self):
        pass

    def draw_video(self):
        pass

    def IOU(self, preds, gt):
        overlap = self.OVERLAP(preds,gt)
        if overlap <= 0.:
            return 0.
        iou = overlap / ((preds[2]-preds[0]+self.offset)*(preds[3]-preds[1]+self.offset)+(gt[2]-gt[0]+self.offset)*(gt[3]-gt[1]+self.offset)-overlap)
        return iou

    def COVER(self, preds, gt):
        overlap = self.OVERLAP(preds,gt)
        if overlap <= 0.:
            return 0.
        cover = overlap / ((gt[2]-gt[0]+self.offset)*(gt[3]-gt[1]+self.offset))
        return cover
    
    def OVERLAP(self, preds, gt):
        overlap=max(0.,min(preds[2],gt[2])-max(preds[0],gt[0])+self.offset)*max(0.,min(preds[3],gt[3])-max(preds[1],gt[1])+self.offset)
        return overlap
    
    def setup_cfg(self, test_cfg):
        '''
        det2 model config
        '''
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.set_new_allowed(True)# to add new key in yaml
        cfg.merge_from_file(test_cfg["cfg_path"])
        cfg["MODEL"]["WEIGHTS"] = test_cfg["pth_path"]
        print("using det2 weight: {}".format(cfg["MODEL"]["WEIGHTS"]))
        cfg["MODEL"]["SOLOV2"]["SCORE_THR"] = test_cfg["conf_thr"]
        cfg["INPUT"]["MIN_SIZE_TEST"] = test_cfg["model_input_sz"]["input_h"]

        cfg.freeze()
        return cfg

    def create_model(self, cfg):
        if cfg["caffe"]:
            if cfg["mission"] == "SSDdetection":
                model = DetectionModel(cfg)
            elif cfg["mission"] == "FCOSdetection":
                model = FCOSDetectionModel(cfg)
            elif cfg["mission"] == "classification":
                model = ClassificationModel(cfg)
        elif cfg["Pytorch"]:
            model_cfg = self.setup_cfg(cfg)

            model = DefaultPredictor(model_cfg)
            #self.trace_model(model, cfg)
        return model

    def parse_pred(self, predictions):
        boxes = []
        classnames = []
        scores = []

        if "instances" in predictions:
            instances = predictions["instances"].to(torch.device("cpu"))
            mask = instances.scores > 0.5
            instances.pred_boxes.tensor = instances.pred_boxes.tensor[mask]
            instances.pred_classes = instances.pred_classes[mask]
            instances.scores = instances.scores[mask]

            boxes = instances.pred_boxes.tensor.tolist()
            classnames = instances.pred_classes.tolist()
            scores = instances.scores.tolist()
        
        class_map = {}
        for i,cls in enumerate(cfg["project_class"]):
            class_map[i] = cls
        
        classnames = [class_map[label] for label in classnames]

        return boxes, classnames, scores

    def trace_model(self, model, cfg):
        dummy_input = torch.rand(1, cfg["model_input_sz"]["input_c"], cfg["model_input_sz"]["input_h"], cfg["model_input_sz"]["input_w"])
        smodel = torch.jit.trace(model, dummy_input)
        smodel.save("/data/taofuyu/fcos_x5.script.pth")

if __name__ == "__main__":
    #load cfg
    test_cfg = "/data/taofuyu/models/dataset_config/test_H1M.yaml"
    cfg = load_cfg(test_cfg)
    wall_breaker = WallBreaker(cfg)

    #load model
    model = wall_breaker.create_model(cfg)

    #read img
    image_reader = ImageReader(cfg)
    if cfg["print_metric"]:
        child_dataset_list = image_reader.create_dataset_list()
        if cfg["mission"] == "classification":
            class_map = wall_breaker.compute_classification_classmap(cfg)
            total_num = 0
            total_right = 0
        for dataset in child_dataset_list:
            img_list = image_reader.create_img_list(dataset)

            #detection result
            result_on_dataset = {}
            for img in img_list:
                src_img = image_reader.read_img(img)
                src_img_shape = src_img.shape
                if not cfg["arm_result"]:
                    if cfg["caffe"]:
                        src_img = image_reader.preprocess(src_img)
                        if cfg["mission"] in ["SSDdetection", "FCOSdetection"]:
                            boxes, classnames, scores = model.run(src_img, src_img_shape)
                        elif cfg["mission"] == "classification":
                            max_class_idx = model.run(src_img, src_img_shape)
                    elif cfg["Pytorch"]:
                        predictions = model(src_img)
                        boxes, classnames, scores = wall_breaker.parse_pred(predictions)
                else:
                    boxes, classnames, scores = get_wenxin_arm_result(img, src_img_shape[1], src_img_shape[0], "210313_R3_Detect_quant")#R3 is right
                if cfg["mission"] == "SSDdetection":
                    result_on_dataset[img] = [boxes, classnames, scores, src_img_shape]
                elif cfg["mission"] == "classification":
                    result_on_dataset[img] = max_class_idx

            #compute
            if cfg["mission"] == "SSDdetection":
                total_num, avg_iou, avg_score, recall, error = wall_breaker.compute_det_metrics(img_list, result_on_dataset)
                for each_cls in cfg["test_class"]:
                    print('{}: Class: {} Total: {} Avg_IoU: {:.4f} Avg_score: {:.4f} Recall: {:.4f} Error: {:.4f}'.format(dataset, each_cls, \
                                            total_num[each_cls], avg_iou[each_cls], avg_score[each_cls], recall[each_cls], error[each_cls]))
            elif cfg["mission"] == "classification":
                total_num += len(img_list)
                right, acc = wall_breaker.compute_cls_metrics(class_map, dataset, result_on_dataset)
                total_right += right
                print("{} total num: {}, acc: {:.4f}".format(dataset, len(img_list), acc))

        if cfg["mission"] == "classification":
            print("total num: {}, acc: {:.4f}".format(total_num, float(total_right)/total_num))
        
    if cfg["draw_result"]:
        img_list = image_reader.create_img_list(cfg["draw_img_path"])
        for img in img_list:
            src_img = image_reader.read_img(img)
            src_img_shape = src_img.shape

            if cfg["caffe"]:
                src_img = image_reader.preprocess(src_img)
                boxes, classnames, scores = model.run(src_img, src_img_shape)
            elif cfg["Pytorch"]:
                predictions = model(src_img)
                boxes, classnames, scores = wall_breaker.parse_pred(predictions)

            wall_breaker.draw_img(img, boxes, classnames, scores)
            #wall_breaker.save_to_txt(img, result_file, boxes, classnames, scores)
    
    if cfg["draw_video"]:
        pass




            

