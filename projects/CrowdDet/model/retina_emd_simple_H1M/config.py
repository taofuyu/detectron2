import os
import sys

import numpy as np

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_dir = '/data/taofuyu/repos/detectron2/projects/CrowdDet'
add_path(os.path.join(root_dir))
add_path(os.path.join(root_dir, 'lib'))

class H1M_data:
    class_names = ['plate', 'headstock', 'tailstock', 'car', 'side_window', 'window', 'roof', 'cycle']
    num_classes = len(class_names)
    root_folder = '/data/taofuyu/tao_dataset/high_roadside'
    image_folder = '/data/taofuyu/tao_dataset/high_roadside'
    train_source = os.path.join('/data/taofuyu/tao_txt/high_roadside/det2/train_high_roadisde_imglist_only_h1m.txt')
    eval_source = os.path.join('/data/taofuyu/tao_txt/high_roadside/det2/val_high_roadisde_imglist.txt')

class Config:
    output_dir = ''
    model_dir = os.path.join('model_dump')
    eval_dir = os.path.join('eval_dump')
    init_weights = ''

    # ----------data config---------- #
    image_mean = np.array([0,0,0])
    image_std = np.array([255, 255, 255])
    train_image_short_size = 324
    train_image_max_size = 576
    eval_resize = True
    eval_image_short_size = 324
    eval_image_max_size = 576
    seed_dataprovider = 3
    train_source = H1M_data.train_source
    eval_source = H1M_data.eval_source
    image_folder = H1M_data.image_folder
    class_names = H1M_data.class_names
    num_classes = H1M_data.num_classes
    class_names2id = dict(list(zip(class_names, list(range(num_classes)))))
    gt_boxes_name = 'fbox'

    # ----------train config---------- #
    backbone_freeze_at = 0
    train_batch_per_gpu = 8
    momentum = 0.9
    weight_decay = 1e-4
    base_lr = 0.005
    focal_loss_alpha = 0.25
    focal_loss_gamma = 2

    warm_iter = 800
    max_epoch = 50
    lr_decay = [33, 43]

    with open(train_source, "r") as f:
        all_imgs = f.readlines()
    nr_images_epoch = len(all_imgs)
    del all_imgs
    
    log_dump_interval = 20

    # ----------test config---------- #
    test_layer_topk = 1000
    test_nms = 0.5
    test_nms_method = 'set_nms'
    visulize_threshold = 0.5
    pred_cls_threshold = 0.5

    # ----------dataset config---------- #
    nr_box_dim = 5
    max_boxes_of_image = 500

    # --------anchor generator config-------- #
    anchor_base_size = 128 # the minimize anchor size in the bigest feature map.
    anchor_base_scale = [2, 2*(2**(1/2)), 4]
    anchor_aspect_ratios = [1, 2]
    num_cell_anchors = len(anchor_aspect_ratios) * len(anchor_base_scale)

    # ----------binding&training config---------- #
    smooth_l1_beta = 0.1
    negative_thresh = 0.4
    positive_thresh = 0.5
    allow_low_quality = True

config = Config()
