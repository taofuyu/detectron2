
import os
import cv2


class ImageReader:
    def __init__(self, cfg):
        print("init ImageReader")
        self.mean = cfg["mean"]
        self.std = cfg["std"]
        self.input_w = cfg["model_input_sz"]["input_w"]
        self.input_h = cfg["model_input_sz"]["input_h"]
        self.input_c = cfg["model_input_sz"]["input_c"]
        self.test_img_path = cfg["test_img_path"]
        self.mask_osd = cfg["mask_osd"]

    def load_img_and_boxGT(self, img_path, gt_path):
        '''
        '''
        img = self.read_img(img_path)
        gt = self.load_boxGT(gt_path)

        return img, gt
    
    def load_img_and_classGT(self, img_path, gt_path):
        pass

    def load_img_and_imgGT(self, img_path, gt_path):
        pass

    def read_img(self, full_path):
        img = cv2.imread(full_path)
        if img is None:
            print("{} is empty".format(full_path))
            assert(False)
        
        return img

    def load_boxGT(self):
        pass

    def create_dataset_list(self):
        dataset_list = []

        for test_dataset in self.test_img_path:
            for child_path in self.test_img_path[test_dataset]:
                dataset_list.append(child_path)
        
        return dataset_list

    def create_img_list(self, dataset):
        img_list = []

        for dir, _, files in os.walk(dataset):
            if len(files) == 0:
                continue
            for img in files:
                if not img[-3:] in ["jpg", "JPG", "bmp", "png"]:
                    continue
                img_list.append(os.path.join(dir, img))
        
        return img_list

    def preprocess(self, img):
        '''resize and normolize the img
        '''
        #mask osd
        if self.mask_osd:
            img = self.mask_img_osd(img)

        #resize if w or h not equals between img and model_input_sz
        if (not img.shape[0]==self.input_h) or (not img.shape[1]==self.input_w):
            new_img = cv2.resize(img, (self.input_w, self.input_h))
            if self.input_c == 1:
                new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        else:
            if self.input_c == 1:
                new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                new_img = img.copy()

        #normolize
        new_img = new_img.reshape((new_img.shape[0], new_img.shape[1], self.input_c))
        new_img = new_img.astype('float')
        for c in range(self.input_c):
            new_img[:,:,c] = (new_img[:,:,c] - self.mean[c]) / self.std[c]

        assert new_img.shape[2] == self.input_c, 'img channel != model input channel'

        return new_img

    def mask_img_osd(self, img):
        rects = []
        img_h, img_w, _ = img.shape

        if img_w == 1280:
            rects.append([30,30,300,150])
            rects.append([30,110,170,30])
            rects.append([900,30,400,60])
            rects.append([900,650,300,60])
        elif img_w == 1920:
            rects.append([50,50,600,100])
            rects.append([70,130,600,100])
            rects.append([1300,40,600,100])
            rects.append([1400,1000,600,100])

        for rect in rects:
            x_min, y_min, w, h = rect
            x_max = x_min + w - 1
            y_max = y_min + h - 1
            img[y_min:y_max, x_min:x_max] = [0,0,0]

        return img