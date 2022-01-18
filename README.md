# DeeplabV3Plus
Deeplab笔记以及DeeplabV3Plus实现

## 背景
语义分割是许多视觉理解系统重要组成部分。主要有以下的应用场景：医学图像分析，无人驾驶，地物分类等。最早的语义分割算法是基于阈值化、直方图、区域划分、聚类等方法，而基于深度学习的分割方法主要分为以下几类:

- Fully convolutional networks
- Convolutional models with graphical models
- Encoder-decoder based models
- Multi-scaledand pyramid network based models
- R-CNN based models(for instance models)
- Dilated convolutional models and DeepLab family
- Recurrent network based models
- Attention-based models
- Generative models and adversarial training
- Convolutional models with active contour 
![在这里插入图片描述](https://img-blog.csdnimg.cn/913a01fa8273433e8ede3cc4094bb002.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_20,color_FFFFFF,t_70,g_se,x_16)
本文主要主要介绍Deeplab系列算法，Deeplab算法里面主要介绍Deeplabv3plus算法，后续会将语义分割算法综述的论文翻译一遍（见参考1）。

## Deeplab Family
Dilated convolution(扩张/空洞 卷积)如下图所示。$y_i=\sum^{K}_{k=1}x[i+rk]w[k]$，其中r是膨胀率，即卷积核里权重之间的间距。
![在这里插入图片描述](https://img-blog.csdnimg.cn/53f9182d4a02439596b03cd1ec96d28a.png)
### DeeplabV1
[Semantic Image Segmentation With Deep Convolution Nets and Fully Connected CRFS](https://arxiv.org/pdf/1412.7062v3.pdf)
 
 Deeplabv1主要结合了深度卷积神经网络（DCNNS）和概率图模型（CRFs）的方法。由于DCNNs的高级特征的平移不等性，在重复的池化和下采样导致DCNNs在语义分割任务精准度不够。针对信号下采样或池化降低分辨率，Deeplab采用空洞卷积算法扩展感受野，获取更多的语义信息。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/6188f95c05c642c689a932cea63086ed.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_20,color_FFFFFF,t_70,g_se,x_16)
Deeplabv1做了以下的修改：

- VGG16的全连接层转为卷积
- 最后两个最大池化层去掉下采样
- 后续卷积层改为空洞卷积
![在这里插入图片描述](https://img-blog.csdnimg.cn/92559835055e45bfa44b688272715c0f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/f98be230b2c34a389b29ec9045159dd8.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_20,color_FFFFFF,t_70,g_se,x_16)

### DeeplabV2

[DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/pdf/1606.00915.pdf)
![在这里插入图片描述](https://img-blog.csdnimg.cn/3bdfdf2c7b27416fa11dfea6769e86d6.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_20,color_FFFFFF,t_70,g_se,x_16)
相比DeeplabV1 Deeplabv2的主要改进是：

- 提出了ASPP，使用多个不同的采样率采样得到多尺度分割对象获得更好的分割效果。
![在这里插入图片描述](https://img-blog.csdnimg.cn/695011afbdb2463b88f7573112a7085d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_17,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/e61cf40c415b4729ba834329d870fcd4.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_16,color_FFFFFF,t_70,g_se,x_16)

-  Backbone使用Restnet


### DeeplabV3
[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf)
![在这里插入图片描述](https://img-blog.csdnimg.cn/1efa3cb45ac141fe9fba9d92f3affa5f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_14,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/99f0bbda5b054e318a31f84884f46e15.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_15,color_FFFFFF,t_70,g_se,x_16)

![在这里插入图片描述](https://img-blog.csdnimg.cn/73f437a25006411591a42bc495b6a97f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/fe6190e026224bfc85cc475488691ba5.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/dfc602e5b29a4f3e9fd2bad3400f17b3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_20,color_FFFFFF,t_70,g_se,x_16)


### DeeplabV3+
[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)

![在这里插入图片描述](https://img-blog.csdnimg.cn/ad1852bd92d2433aa0be7dfbf3b8b619.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_19,color_FFFFFF,t_70,g_se,x_16)

![在这里插入图片描述](https://img-blog.csdnimg.cn/47e0bf7360404973b0da03ac4bac114c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/b0f491d15b58415a98659fbe68b21a45.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/d921b74657d44a40b0f156a38cf1f15b.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/1c5b4c0a806a4dd0928e6aae3815c57c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQ2hlbmpsX3JlYWw=,size_20,color_FFFFFF,t_70,g_se,x_16)


## 参考
1. [语义分割综述](https://arxiv.org/abs/2001.05566)
2. [DeeplabV3+ Tensorflow2.0实现](https://github.com/RyanCCC/DeeplabV3Plus/tree/main)有用的话请给我一个star，非常感谢！
