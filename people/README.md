## 文件夹文件简介

**\plot** : 训练完之后进行检测时所做的数据统计

**.\people.py**：训练和预测时用到的脚本

**.\try.py**：进行代码书写时的代码片段(可忽略)



## 使用训练好的模型进行预测

对于图片:

```bash
python3 people.py splash --weights=/path/to/mask_rcnn/mask_rcnn_people.h5 --image=<file name or URL>
```

对于视频. 要求 OpenCV 3.2+:

```bash
python3 people.py splash --weights=/path/to/mask_rcnn/mask_rcnn_people.h5 --video=<file name or URL>
```

## 训练模型

使用预训练模型进行训练

```bash
python3 people.py train --dateset=/path/to/people/dataset --weights=/path/to/mask_rcnn/mask_rcnn_coco.h5
```

继续训练之前停止训练的模型

```bash
python3 people.py train --dataset=/path/to/people/dataset --weights=last
```

## One more thing

​	people.py根据balloon.py样例代码进行修改，并对其中的可视化模块进行替换，对车厢内人数进行分层，方便画图时对拥挤状况进行评估。