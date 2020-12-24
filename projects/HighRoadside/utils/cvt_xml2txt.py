import os
from xml.etree import ElementTree as ET

classname_indexes={'plate':0,'plate_ga':0,'plate_z':0,'palte':0,'headstock':1,'headtock':1, 'tailstock':2,'tailtock':2,'taistock':2,'car':3,'side_window':4,'window':5}

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

def cvt_from_folder(xml_file_path, txt_file_path, train_txt_file):
    all_files = os.listdir(xml_file_path)
    all_files.sort()
    txt_file = open(txt_file_path, 'a')
    #train_file = open(train_txt_file, 'a')
    for each_file in all_files:
        if each_file.split('.')[-1] == 'xml':
            xml_filename = each_file.split('.')[0]
            if os.path.exists(xml_file_path + xml_filename + '.jpg'):
                #train_file.write('/data/taofuyu/tao_dataset/roadside/bread_car/'+xml_filename + '.jpg'+' '+'\n')
                pass
            else:
                continue
            xml_file_path_and_name = os.path.join(xml_file_path, each_file)
            tree = ET.parse(xml_file_path_and_name)
            root = tree.getroot()
            width = float(root.find('size').find('width').text)
            height = float(root.find('size').find('height').text)
            src_area = width*height
            boxes = []
            
            for each_object in root.findall('object'):
                box = []
                name = each_object.find('name').text
                name = correct_name(name)
                bndbox = each_object.find('bndbox')
                if not name in classname_indexes:
                    print(xml_file_path,name)
                    continue
                if classname_indexes[name] is not None:
                    x_min = int(bndbox.find('xmin').text)
                    y_min = int(bndbox.find('ymin').text)
                    x_max = int(bndbox.find('xmax').text)
                    y_max = int(bndbox.find('ymax').text)

                    box.append(x_min)
                    box.append(y_min)
                    box.append(x_max)
                    box.append(y_max)
                    box.append(classname_indexes[name])

                    boxes.append(box)
            
            txt_file.write('/detectron2/datasets/high_roadside/patch_haikang/'+xml_filename+'.jpg ')
            for box in boxes:
                txt_file.write(str(box[0])+' '+str(box[1])+' '+str(box[2])+' '+str(box[3])+' '+str(box[4])+' ')
            txt_file.write('\n')
           

    txt_file.close()
    #train_file.close()

if __name__ == '__main__':
    xml_file_path = '/media/t/64006e9c-fffc-427f-b780-00c20066dffa/map_server/mapserver23_taofuyu/tao_dataset/high_roadside/patch_haikang/'
    train_txt_file = '/media/tao/64006e9c-fffc-427f-b780-00c20066dffa/map_server/mapserver23_taofuyu/tao_dataset/roadside/'
    txt_file_path = '/media/t/64006e9c-fffc-427f-b780-00c20066dffa/map_server/mapserver23_taofuyu/tao_txt/det2/high_roadside/train_high_roadisde_imglist.txt'
    cvt_from_folder(xml_file_path, txt_file_path, train_txt_file)
    cvt_from_file()
 
    