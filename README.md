## U-GAT-IT - Unofficial Paddle Implementation
This project is an unofficial Paddle implementation of U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation. It is adapted from  [znxlwm/UGATIT-pytorch](https://github.com/znxlwm/UGATIT-pytorch).

本项目是U-GAT-IT: 对官方实现[znxlwm/UGATIT-pytorch](https://github.com/znxlwm/UGATIT-pytorch)进行修改为PaddlePaddle框架.

本人能力有限，复现效果一般，本项目为百度顶会论文复现营的结课作业。
https://aistudio.baidu.com/aistudio/education/group/info/1340


## 准备数据
将selfie2anime数据集放到dataset目录下, 这里将数据集文件夹名称命名为selfie2anime

## 训练模型

这里我们训练UGATIT的Light版本的模型, Light版本的模型占用的显存比Full版本的少. 如果想训练Full版本, 可以去掉参数`--light True`.

```bash
python main.py ---dataset selfie2anime
```

生成的结果会保存在目录`results/<数据集名称>/img`下.

## 测试模型
```bash
python main.py  --dataset selfie2anime --phase test
```
程序会加载迭代为10000的保存点, 并进行测试. 如果使用selfie2anime数据集.

结果会保存在目录`results/<数据集名称>/test`下.



## 代码结构

- main.py

入口代码, 负责声明各种命令参数

- networks.py

  U-GAT-IT模型的搭建

- UGATIT.py

  U-GAT-IT模型的训练

- dataset.py

  读取数据集

- transforms.py 

  图像的相关方法，与torchvision中的transforms相似

- results/selfie2anime/model
 
 模型保存地址
  
- results/selfie2anime/test

 测试输出的图片


## 训练日志
log.txt记录训练过程
数据集: selfie2anime




## 论文引用

```
@inproceedings{
Kim2020U-GAT-IT:,
title={U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation},
author={Junho Kim and Minjae Kim and Hyeonwoo Kang and Kwang Hee Lee},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJlZ5ySKPH}
}
```
