import json
import os

if __name__ == "__main__":
    raw_files = ["/data/taofuyu/tao_dataset/high_roadside/gt/xyxy/patch_one_train_imglist",\
                "/data/taofuyu/tao_dataset/high_roadside/gt/xyxy/patch_three_train_imglist",\
                "/data/taofuyu/tao_dataset/high_roadside/gt/xyxy/patch_four_train_imglist",\
                "/data/taofuyu/tao_dataset/high_roadside/gt/xyxy/patch_haikang_train_imglist",\
                "/data/taofuyu/tao_dataset/high_roadside/gt/xyxy/patch_songli_train_imglist",\
                "/data/taofuyu/tao_dataset/high_roadside/gt/xyxy/patch_vz_roof_train_imglist"]
    
    dst_file = "/data/taofuyu/tao_dataset/high_roadside/annotation_train.odgt"

    for raw in raw_files:
        with open(raw, "r") as f:
            all_lines = f.readlines()
        
        for line in all_lines:
            line = line.strip("\n").strip(" ")
            img_path = line.split(" ")[0]
            label = line.split(" ")[1:]
            