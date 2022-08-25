# DeeplabV3Plus

本仓库实现DeeplabV3plus的算法。

## 项目目录
```
├─data：存放类别文件名
│------VOCdevkit.names
│
├─doc：存放文档
│------Deeplab.md
│      
├─model：存放训练好的模型
|
├─nets：网络实现代码
|
├─pytorch：Pytorch实现部分
|
├─utils：基础工具方法
|
├─config.py：配置文件
|
├─evaluate.py：模型评估代码
|
├─export.py：模型导出代码
|
├─inference.py：推理代码
|
├─splitDataset.py：数据集切分代码
|
└─train.py：训练代码
```

## 算法训练

DeepV3实现有两个版本，一个是基于Tensorflow2实现，另一个是基于Pytorch实现的。


## 验证评估


## 模型导出


## 参考

1. [Deeplab系列算法](./doc/Deeplab.md)
2. [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)
3. [keras-deeplab-v3-plus](https://github.com/bonlime/keras-deeplab-v3-plus)