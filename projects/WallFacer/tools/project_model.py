'''
create a base model(no matter caffe or pytorch), use for classification or detection or regression or something...
to make sure that env is right, this project should run in my docker.
'''
import os
#os.environ['GLOG_minloglevel']='2'
import sys
caffe_root='/data/taofuyu/cafferaw_det/'
sys.path.append(caffe_root+'python')
import caffe
import numpy as np
import cv2
import time
from .extract_box import BoxExtractor


__all__ = ["BaseModel", "DetectionModel"]

class BaseModel:
    def __init__(self, cfg):
        self.deploy = cfg["deploy_path"]
        self.weight = cfg["weight_path"]
        self.conf_thr = cfg["conf_thr"]
        self.iou_thr = cfg["iou_thr"]
        self.project_class = cfg["project_class"]
        self.input_w = cfg["model_input_sz"]["input_w"]
        self.input_h = cfg["model_input_sz"]["input_h"]
        self.input_c = cfg["model_input_sz"]["input_c"]

    def run(self, img, src_img_shape):
        net_out = self.forward(img)
        result = self.post_process(net_out, src_img_shape)

        return result
    
    def forward(self, img):
        print("please implement a sub-class method")
        return None
    
    def post_process(self, net_out, src_img_shape):
        print("please implement a sub-class method")
        return None


class DetectionModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        caffe.set_mode_gpu()

        self.gpu_id = cfg["gpu_id"]
        self.net = caffe.Net(self.deploy, self.weight, caffe.TEST)
        self.ssd_add_bg = cfg["ssd_add_bg"]
        self.use_ssd = cfg["use_ssd"]

        #reshape input layer
        self.net.blobs['data'].reshape(1, self.input_c, self.input_h, self.input_w)
        self.net.reshape()
        self.net.forward()
        self.in_data = np.empty((1, self.input_c, self.input_h, self.input_w)).astype('float')
        
        print('{} initialized'.format(self.deploy))

    def run(self, img, src_img_shape):
        net_out = self.forward(img)
        boxes, classnames, scores = self.post_process(net_out, src_img_shape)

        return boxes, classnames, scores
    
    def forward(self, img):
        #load img data
        for c in range(self.input_c):
            self.in_data[0, c, :, :] = img[:, :, c]
        
        #pass data to input blob
        caffe.set_device(self.gpu_id)
        self.net.blobs['data'].data[...] = self.in_data[0:1,:,:,:]
        
        #get net result
        net_out = self.net.forward()

        return net_out
    
    def post_process(self, net_out, src_img_shape):
        box_extractor = BoxExtractor(0., self.conf_thr, self.use_ssd, self.ssd_add_bg)

        box_data = net_out["final_nms_box"][0,...].reshape((1,-1))
        
        boxes, classnames, scores = box_extractor.extractImageBoxes(box_data, self.project_class, src_img_shape)

        return boxes, classnames, scores