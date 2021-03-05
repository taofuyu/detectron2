## WallFacer Project
此工程用于训练的准备工作、测试工作。
数据筛选和数据标注阶段称为WallBuilder、训练数据文件准备阶段称为WallFacer、测试阶段称为WallBreaker。

### 主要目的
* 现有的所有数据，统一标签格式、顺序、存储位置
* 有新的任务时，可以根据需求从所有数据中迅速组织出所需要的数据
* 从大量新数据中，自动挑选出所需要的数据
* 自动标注新数据
* 不同框架之间的标签格式转换
* 新任务快速搭建测试

### 标签规定
darkpaladin数据标签顺序：
```
0        1        2        3        4             5        6          7          8          9        10        11        12         13         
person   car      cycle    window   side_window   plate    headstock  tailstock  lightarea  light    bulb
人       车身      自行车    前后车窗  侧边车窗        车牌      车头       车尾        交通灯牌     交通灯   交通灯泡    人头       人头戴头盔  模型标记的人头
```
taofuyu标签顺序：
```
0        1            2          3        4             5        6       7        8
plate    headstock    tailstock  car      side_window   window   roof    person   cycle
车牌      车头         车尾        车身      侧边车窗       前后车窗   车顶    人       非机动车 
```

### 现有数据集
#### 统一GT格式
1. 在每个数据集的文件夹中，有一个"gt"文件夹
2. "gt"文件中，有"xywh"文件夹、"xyxy"文件夹。文件夹中标签按照上文"taofuyu标签顺序"规定。有一个config.yaml文件，用于配置该数据集的各个子数据集如何进行组合。
3. "xywh"文件夹，有每个子数据集的train_imglist.txt;以及总的train_imglist。gt组织格式:
full/path/img_name x y w h label_id x y w h label_id ... x y w h label_id\n
\n
full/path/img_name x y w h label_id x y w h label_id ... x y w h label_id\n
绝对路径/图片名 归一化的中心点x 归一化的中心点y 归一化的w 归一化的h 类别标签
/data/taofuyu/.../1.jpg 0.34245432 0.68597784 0.43267846 0.4782336 3 0.473846278 0.32432 0.4323423 0.4873647 2
4. "xyxy"文件夹，有每个子数据集的train_imglist.txt;以及总的train_imglist。gt组织格式:
full/path/img_name x y x y label_id x y x y label_id ... x y x y label_id\n
\n
full/path/img_name x y x y label_id x y x y label_id ... x y x y label_id\n
绝对路径/图片名 左上角x 左上角y 右下角x 右下角y 类别标签
/data/taofuyu/.../1.jpg 489 285 540 305 0 1968 273 2028 294 5
5. "tools"文件夹中，包含将xml、json标注转换为上述txt格式的脚本、将上述txt转换为caffe和det2所需格式的脚本、将不同txt进行各种组合的脚本

#### H1L（更新时间：2021.2.5）
1. 路径: 20:/data1/taofuyu/tao_dataset/roadside/
2. 人工标记标签: plate  headstock  tailstock  car  person  cycle
3. 自动标记标签: none 
4. 训练集: 
```
patch_one(早期手持相机拍摄,9267张)
patch_three(早期手持相机拍摄,16183张)
patch_four(实验田录制视频,挑选进出场过程,9711张)
patch_five(汽车之家挑选的面包车、SUV,2185张)

patch_real_false(现场返回的一些出问题视频所拆图片,1648张)
patch_bread_car(实验田的一些面包车,1512张)

picked_cycle(实验田的一些非机动车,730张)
patch_real_cycle(测试部攒的几千张现场非机动车,3776张)

patch_cycle(晓燕外采非机动车,16297张)
patch_xiaoyan(晓燕外采遮挡图片,5826张)

patch_xinghan(星汉爬的一波数据,27269张)
patch_cq(重庆实验田录制视频，挑选进出场过程,1227张)
patch_cq_second(重庆实验田录制视频，挑选进出场过程,ISP有问题,11427张)
patch_dianjiang(重庆垫江爬取的图片，带"labeled"后缀的文件夹,专门挑选面包车SUV车尾、非机动车、行人,19480张)

patch_nanwei(南威项目图片，主要是大量非机动车, 8090张)
上述共13万4628张
pic_lxh(星汉持续搜集的新现场图片，待挑选、待标记)
```
5. 测试集:
```
test(随机挑选的2089张图片，偏少，可以补充了)
test_cycle(测试部的419张非机动车测试图片)
test_wrong(偶尔返回的现场问题图，无标签)
```
#### H1M（更新时间：2021.2.5）
1. 路径: 20:/data1/taofuyu/tao_dataset/high_roadside/
2. 人工标记标签: plate  headstock  tailstock  car  side_window  window  roof  person  cycle
3. 自动标记标签: none
4. 训练集: 
```
patch_one(项目早期实验田图片,1714张)
patch_three(项目早期实验田图片,2613张)
patch_four(客户提供的其他厂商相机图片,3841张)
patch_songli(松立提供的其他厂商相机图片,8317张)
patch_haikang(海康相机图片,25266张)
patch_vz_roof(我们的实验田图片，702张)
patch_vz_ear(我们的实验田图片，标记了后视镜,702张)
上述共4万2453张
high_vz_等待标记_雪天(待标记的雪天图片)
```
5. 测试集:
```
test(随机挑选的1818张图片，偏少，可以补充了)
test_wrong(画出来看看情况的图片)
```
#### 出入口（更新时间：2021.2.6）
1. 路径: 20:/data1/taofuyu/tao_dataset/det_plate/
2. 人工标记标签: plate  headstock
3. 自动标记标签: tailstock  car  side_window  window
4. 训练集: 
```
image/Audi (奥迪车，943张)
image/FinDouble (双层黄牌,4756张)
image/NewPlate (使领馆车牌,单层武警,单排黄牌车,双层武警车牌,双排军车牌,双排黄牌,教练车牌,新能源车牌)
image/PlusHead (普通蓝牌,29265张)
image/PlusHeadGreenTrain (GAN造的新能源,14257张)
image/PlusHeadHKTrain (港澳车牌,3924张)
image/YHF_lmdb_trans (普通蓝牌,29223张)

aodi_new4000 (奥迪车,4479张)
lexus_spider_picked (爬虫爬的雷克萨斯，133张)
similar_audi (类似奥迪的有点条纹的车，430张copy8次)
wuxi_special (从无锡场景中筛选出的特殊车牌，但其中单层和双层黄牌共3万张，数量太多，容易淹没其他类别)

new_RM (2020年RM相机从全国各省搜集的90万张触发图，数量虽多，但绝大多数是蓝牌，label全自动标注)
new_RX (2020年RX相机从全国各省搜集的100多万张触发图，按触发时的识别结果分了文件夹，不一定准确。绝大多数是蓝牌，还没自动标记。这些文件夹中，数字开头的和字母开头的，大多是特殊车牌，值得标记)

parking_img/headstock (车正面, 5975张)
parking_img/tailstock (车背面, 16022张)
parking_img/hardsamples (现场问题, 5023张)
parking_img/2020_collect/1-35 (2020年搜集的车位引导图, 14万1629张)

深圳珠海车牌港澳采集 (未标记，基本都是粤字蓝牌)
```
5. 测试集:
```
plate_test_dataset 精心整理的出入口测试集，详情见该文件夹的readme
```

#### 无锡（更新时间：2021.2.6）
1. 路径: 20:/data1/taofuyu/tao_dataset/wuxi/
2. 人工标记标签: plate  headstock  tailstock  car  side_window  window  roof  person  cycle
3. 自动标记标签: 无
4. 训练集: 
```
masked_wuxi_full_p6(3342张)
masked_wuxi_full_p7(2136张)
masked_wuxi_full_p8(12039张)
masked_wuxi_p1_oldchange(540张)
wuxi_lightbulb_p2_oldchange(2741张)
wuxi_lightbulb_p3(14228张)
上述共35026张
```
5. 测试集:
```
写在了对应的val.txt中
```
#### road（更新时间：2021.2.6）
1. 路径: 20:/data1/taofuyu/tao_dataset/road/
2. 人工标记标签: plate  car   roof
3. 自动标记标签: headstock  tailstock  side_window  window  person  cycle
5. 训练集: 
```
data_1.5k(1524张)
data_2.5k(2401张)
data_R(3367张)
data_R_A(5768张)
上述共13060张，视角类似于高位同侧
```
6. 测试集:
```
null
```
#### MINI（更新时间：2021.2.6）
1. 路径: 20:/data1/taofuyu/tao_dataset/MINI/
2. 人工标记标签: plate
3. 自动标记标签: none
5. 训练集: 
```
Karl文件夹，多样车牌数据，但很多被用到了720测试集当中，因此直接使用det_plate中的多样车牌。
车头栅格图片 (new_RM数据集中，经过车型车款模型粗筛、人工细筛之后的车头栅格图片，28843张)
双层挂车牌 (ysx训练识别模型用到的图，2583张)
双层军车牌 (ysx训练识别模型用到的图，1352张)
双层普通车牌 (即双层黄牌，ysx训练识别模型用到的图，13867张)
双层武警车牌 (ysx训练识别模型用到的图，447张)
```
6. 测试集:
```
double_test_imglist.txt
```
#### DETRAC（更新时间：）
#### CityScapes（更新时间：）
#### COCO（更新时间：）
#### 南威（更新时间：2021.2.6）

### 项目情况
1. H1L
模型系列: T
预测类别: 0plate  1head  2tail  3car  4person  5cycle
场景特点: 鱼眼镜头，大角度车牌，车容易被行人、非机动车大面积遮挡，目标车辆少(二车位)，训练集中特殊车辆少，误检要少
训练构成: 更新时间`2021.2.6`，1batch=(66*H1L数据 + 12*其他数据 + 10*南威数据)*2卡
其中，H1L数据: 1*patch_one + 1*patch_three + 1*patch_four + 1*patch_real_false + 10*patch_bread_car + 3*patch_five + 3*picked_cycle + 1*patch_cq
+2*patch_xiaoyan + 1*patch_cq_second + 1*patch_xinghan + 1*patch_cycle
其中，其他数据: 2975张CityScapes + 2134张COCO + 8906张road + 2896张DETRAC + 51947张出入口多样车牌
其中，南威数据: 1*patch_real_cycle + 1*patch_nanwei + 1*patch_dianjiang

2. H1M
模型系列: H
预测类别: 0plate  1head  2tail  3car  4side_window   5window  6roof  7cycle
场景特点: 有同侧、异侧、垂直三种视角，车辆之间遮挡严重，远端四车位极端遮挡(只剩车顶)，场景内目标数量众多，误检要少
训练构成: 更新时间`2021.3.4`，1batch=(14*H1M数据 + 24*其他数据)*3卡
其中，H1M数据: 见H98_coonfig.yaml
其中，其他数据: 见H98_coonfig.yaml

3. R3/R4/RX
模型系列: X
预测类别: 0plate  1head  2tail  3car
场景特点: 速度要快，plate recall必须特别高，车头误检要少，目标车辆一般就一辆，车身字符和人身字符容易误检，多样车牌要求高
训练构成: 更新时间`2021.2.6`，1batch=(57*出入口数据 + 9*new_RM + 2*wuxi_special)*6卡
其中，出入口数据: 2*FinDouble + 1*NewPlate/教练车牌 + 3*NewPlate/使领馆车牌 + 2*NewPlate/单层武警 + 3*NewPlate/双层黄牌 + 10*NewPlate/单层黄牌 + 10*NewPlate/双层武警 + 10*NewPlate/双层军牌 + 1*PlusHeadHKTrain + 1*PlusHeadGreenTrain + 1*YHF_lmdb_trans
其中，new_RM: 从new_RM中随机挑选的42万张图。不全部使用是因为基本都是蓝牌，且标注是自动标注的。
其中，wuxi_special: 在wuxi_special中，蓝牌小汽车只随机选3000张，单层黄牌只随机选4000张，警车车牌只随机选2600张，双排黄牌只选1000张，避免淹没其他类型。

4. RM
模型系列: newCpro
预测类别: 0plate  1head
场景特点: 同R3，但是得是IVE结构的模型，保检出，疯狂误检，靠miniSSD和后续逻辑再做处理
训练构成: 更新时间`2021.2.6`，1batch=(64*出入口数据)*?卡
其中，出入口数据: 5*Audi + 2*FinDouble + 1*NewPlate/教练车牌 + 3*NewPlate/使领馆车牌 + 2*NewPlate/单层武警 + 3*NewPlate/双层黄牌 + 10*NewPlate/单层黄牌 + 10*NewPlate/双层武警 + 10*NewPlate/双层军牌 + 1*PlusHeadHKTrain + 1*PlusHeadGreenTrain + 1*YHF_lmdb_trans

5. miniSSD
模型系列: MINI
预测类别: 0single_plate  1double_plate
场景特点: 输入是四拼图，拼图是车牌外扩、误检、纯黑构成，检出要求高(漏了的话，SSD检出也白检)，模型特别小，图像增强在线模拟拼图
训练构成: 更新时间`2021.2.6`，1batch=(160*Karl数据 + 60*双层数据 + 36*栅格数据)*1卡
其中，Karl数据: MINI数据集里的Karl集，即xiaomeng在21上的miniSSD训练数据(主要是出入口多样车牌)
其中，双层数据: 6*双层军车牌 + 1*双层挂车牌 + 1*双层普通车牌 + 11*双层武警车牌
其中，栅格数据: 1*车头栅格图片 + 1*lexus_spider_picked

6. F1
模型系列: W
预测类别: 0plate  1head or tail  2car
场景特点: 停车场视角容易导致车牌很小，车辆固定不动、因此固定的误检会导致重复上报，
训练构成:

7. F2
模型系列: F
预测类别: 0plate  1head or tail  2car
场景特点: 在F1的基础上，只抠车位部分的patch用于检测
训练构成: 更新时间`2021.2.6`，1batch=(120*停车场数据 + 10*出入口数据)*1卡
其中，停车场数据: 1*det_plate/parking_img/headstock + 1*det_plate/parking_img/tailstock + 1*det_plate/parking_img/hardsamples + 1*det_plate/parking_img/2020_collect
其中，出入口数据: 1*FinDouble + 1*NewPlate + 1*PlusHeadHKTrain + 1*PlusHeadGreenTrain + 1*YHF_lmdb_trans

8. RM汉字识别

9. 无锡行人分类

10. 车辆底盘回归

### 测试指标

### WallFacer使用说明
1. 对于新拿到的标注xml或json文件，用cvt_xml_json_2_txt()将其转换为统一的txt文件(gt文件夹中)
2. 对于没有xml或json标注文件，而是caffe训练文件中带有gt的，用cvt_txt_2_txt()将其转换为统一的txt文件(gt文件夹中)
3. merge_train_imglist()把所有的原始train_imglist合并成一个文件all_train_imglist.txt
4. generate_train_imglist()根据给定的config，指定各个gt文件夹中写好的子数据集文件，生成训练所需的格式的文件
5. draw_to_check_gt()根据train_imglist画gt，看标注情况
6. 流程是: 
有了新的标注数据，放到对应的父数据集文件夹中，作为一个子数据集；
使用cvt_xml_json_2_txt()在"父数据集文件夹/gt/"中生成gt txt文件；
draw_to_check_gt()检查标注情况；
把该子数据集添加到项目的dataset config中；
generate_train_imglist()重新生成训练文件；

### WallBreaker使用说明

### Tips
1. 老的出入口数据和MINI数据可以提供多样车牌训练来源
2. 南威数据可以提供一些非机动车训练来源
3. 数据集中覆盖的普通车辆已经足够，开启新项目时，有基本的普通数据后，重点提前搜集corner case
4. corner case： 大巴车、卡车、货车、工程车、公交车、夜间、遮挡、铁皮三轮车、很大(近)或很小(远)的目标，以及这些车型身上的各种组件