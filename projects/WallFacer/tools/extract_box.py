# -*- coding: utf-8 -*-
'''
modified from darkpaladin
decode net output to normal box.
'''

import numpy as np

class BoxExtractor(object):
    def __init__(self, padding_rate=0., score_thresh=0.5, use_ssd=False, ssd_add_bg=False):
        #[0, class, score, x, y, w, h]
        self.pos_start = 2
        self.box_len = 6
        self.fixed_num = 100
        self.class_start = 0
        self.conf_index = 1
        self.score_index = 1
        self.prob_index = 1

        self.score_thresh = score_thresh
        self.use_ssd = use_ssd
        self.ssd_add_bg = ssd_add_bg
        assert(padding_rate >= 0.)
        self.padding_rate = padding_rate

        if use_ssd:
            self.pos_start = 3
            self.box_len = 7
            self.fixed_num = -1
            self.class_start = 1
            self.conf_index = 2
            self.score_index = 2
            self.prob_index = 2

    def _extractBoxes(self, box_data, classname_map):
        class_bias=0
        if self.use_ssd:
            assert box_data.shape[1]%self.box_len==0
            if self.ssd_add_bg:
                class_bias = -1
            self.fixed_num = int(box_data.shape[1]/self.box_len)
            box_data = box_data.copy().reshape((-1,self.box_len))
            box_data[:,self.pos_start:self.pos_start+4]=\
                np.concatenate((\
                    (box_data[:,self.pos_start]+box_data[:,self.pos_start+2]).reshape((-1,1))*0.5,\
                    (box_data[:,self.pos_start+1]+box_data[:,self.pos_start+3]).reshape((-1,1))*0.5,\
                    (box_data[:,self.pos_start+2]-box_data[:,self.pos_start]).reshape((-1,1)),\
                    (box_data[:,self.pos_start+3]-box_data[:,self.pos_start+1]).reshape((-1,1))),axis=1)
            box_data = box_data.reshape((1,-1))

        assert(box_data.shape[0]==1 and box_data.shape[1]==self.fixed_num*self.box_len)
        #boxes=np.zeros((self.fixed_num,5),np.float)
        boxes=np.zeros((self.fixed_num,4),np.float)
        classnames=[]
        scores=np.zeros(self.fixed_num)
        out_count=0
        end_flag=False
        while out_count<self.fixed_num and (not end_flag):
            if self.use_ssd:
                end_flag=box_data[0,out_count*self.box_len+self.class_start]<0

            if not end_flag:
                boxes[out_count,0]=(max(0.0,box_data[0,out_count*self.box_len+self.pos_start]-box_data[0,out_count*self.box_len+self.pos_start+2]*0.5*(self.padding_rate+1.))+min(1.0,box_data[0,out_count*self.box_len+self.pos_start]+box_data[0,out_count*self.box_len+self.pos_start+2]*0.5*(self.padding_rate+1.)))*0.5
                boxes[out_count,1]=(max(0.0,box_data[0,out_count*self.box_len+self.pos_start+1]-box_data[0,out_count*self.box_len+self.pos_start+3]*0.5*(self.padding_rate+1.))+min(1.0,box_data[0,out_count*self.box_len+self.pos_start+1]+box_data[0,out_count*self.box_len+self.pos_start+3]*0.5*(self.padding_rate+1.)))*0.5
                boxes[out_count,2]=min(1.0,box_data[0,out_count*self.box_len+self.pos_start]+box_data[0,out_count*self.box_len+self.pos_start+2]*0.5*(self.padding_rate+1.))-max(0.0,box_data[0,out_count*self.box_len+self.pos_start]-box_data[0,out_count*self.box_len+self.pos_start+2]*0.5*(self.padding_rate+1.))
                boxes[out_count,3]=min(1.0,box_data[0,out_count*self.box_len+self.pos_start+1]+box_data[0,out_count*self.box_len+self.pos_start+3]*0.5*(self.padding_rate+1.))-max(0.0,box_data[0,out_count*self.box_len+self.pos_start+1]-box_data[0,out_count*self.box_len+self.pos_start+3]*0.5*(self.padding_rate+1.))
                #boxes[out_count,4]=box_data[0,out_count*self.box_len+self.conf_index]
                classnames.append(classname_map[int(box_data[0,out_count*self.box_len+self.class_start])+class_bias])
                scores[out_count]=box_data[0,out_count*self.box_len+self.score_index]
                out_count+=1
        boxes=boxes[0:out_count,:]
        scores=scores[0:out_count]
        score_mask=scores>=self.score_thresh
        boxes=boxes[score_mask,:]
        classnames=[classnames[i] for i in range(len(classnames)) if score_mask[i]>0]
        scores=scores[score_mask]
        return boxes,classnames,scores

    def extractImageBoxes(self,box_data,classname_map,img_shape):
        raw_boxes,classnames,scores=self._extractBoxes(box_data,classname_map)
        boxes=raw_boxes.copy()
        raw_width=img_shape[1]
        raw_height=img_shape[0]
        vec_zero=np.zeros(boxes.shape[0])
        vec_one=np.ones(boxes.shape[0])
        boxes[:,0]=(raw_boxes[:,0]-raw_boxes[:,2]*0.5)*(raw_width-1)
        boxes[:,1]=(raw_boxes[:,1]-raw_boxes[:,3]*0.5)*(raw_height-1)
        boxes[:,2]=(raw_boxes[:,0]+raw_boxes[:,2]*0.5)*(raw_width-1)
        boxes[:,3]=(raw_boxes[:,1]+raw_boxes[:,3]*0.5)*(raw_height-1)
        return boxes,classnames,scores

