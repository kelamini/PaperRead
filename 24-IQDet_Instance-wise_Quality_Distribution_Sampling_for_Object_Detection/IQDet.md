# IQDet: Instance-wise Quality Distribution Sampling for Object Detection

## 前期阅读

### 粗读

**主要解决的问题**：

**提出或应用的新方法**：提出了一种具有 instance-wise 抽样策略的密集目标检测器——IQDet。首先提取每个 ground-truth 的区域特征来估计 instance-wise  quality 分布。（根据空间维度上的混合模型，该分布具有更好的噪声鲁棒性，并适应每个实例的 semantic pattern。）基于这种分布，提出了一种 quality 抽样策略，该策略以概率的方式自动选择训练样本，并以更多的 quality 样本进行训练。

**最终达到的结果**：引用原文的表述：“Extensive experiments on MS COCO show that our method  steadily improves baseline by nearly 2.4 AP without bells  and whistles. Moreover, our best model achieves 51.6 AP,  outperforming all existing state-of-the-art one-stage detectors and it is completely cost-free in inference time.”

### 精读

## 后期阅读

中心思想：

论文写作架构：

对该论文的评价：