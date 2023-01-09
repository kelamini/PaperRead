# 深度学习在视觉 SLAM 中应用综述

## 前期阅读

### 粗读

**主要解决的问题**：总结了传统 SLAM 与基于深度学习的SLAM 的特点 、 性质, 重点介绍和总结了深度学习在视觉里程计、 回环检测中的研究成果 , 展望了基于深度学习的视觉 SLAM 的研究发展方向。

**提出或应用的新方法**：

**最终达到的结果**：

### 精读

 一个完整的 SLAM 框架由以下 4 个方面组成: 前端跟踪 、 后端优化、 回环检测 、 地图重建：

- 前端跟踪即视觉里程计负责初步估计相机帧间位姿状态及地图点的位置; 

- 后端优化负责接收视觉里程计前端测量的位姿信息并计算最大后验概率估计; 

- 回环检测负责判断机器人是否回到了原来的位置, 并进行回环闭合修正估计误差; 

- 地图重建负责根据相机位姿和图像 , 构建与任务要求相适应的地图。

传统的视觉 SLAM 方案分为特征点法和直接法两类：

- 特征点法从每帧图片中提取稳定的特征点, 通过这些特征点具有不变性的描述子完成相邻帧的匹配; 然后通过对极几何较为鲁棒地恢复相机的姿态和地图点坐标, 最后通过最小化投影误差完成相机位姿和地图结构的微调, 每帧所提取的特征点通过聚类等操作进行回环检测或重定位。
- 直接法不再提取特征点, 直接通过光度误差来恢复相机的姿态和地图结构, 不用计算关键点和描述子。

 传统视觉 SLAM 方法有以下几个方面的问题还没有较为完备的解决方案：

1.  在光照条件恶劣或光照变化较大等不利条件下 , 算法的鲁棒性还不是很高；
2.  在相机运动较大的情况, 传统算法容易出现 “ 跟丢” 的情况；
3. 传统算法不能识别前景物体 , 即对场景中运动的物体只能当作 “ 坏点” 来处理, 没有较好的解决方案。

采用深度学习方式处理 SLAM 问题, 有以下几个研究层面的优势：

1. 基于深度学习的 SLAM 方案对光照有较好的不变性 , 能够在光照条件较为恶劣的条件下工作；
2. 基于深度学习的 SLAM 方案能够识别并提取环境中移动的物体，可以进行动态环境下的 SLAM 建模；
3. 通过深度学习的方式可以提取高层语义信息, 为语义 SLAM 的构建以及场景语义信息的理解及使用提供了更大的帮助；
4. 采用深度学习的方式更有利于信息及知识的压缩保存, 更有益于机器人知识库的构建；
5. 基于深度学习的 SLAM 方案更符合人类认知及环境交互的规律, 有更大的研究及发展的潜力。

根据训练方法和数据集标签化程度的不同，将基于深度学习的视觉里程计方法分为监督学习，无监督学习，半监督学习三类分别进行讨论。

监督学习方法的典型代表：

- **PoseNet**：KENDALL A,GRIMES M,CIPOLLA R. Posenet: A convolutional network for real-time 6-dof camera relocal-ization[ C] ∥Computer Vision ( ICCV) ,2015 IEEE In-ternational Conference. New York: IEEE,2015: 2938-2946.
- **DeepVO**：WANG S,CLARK R,WEN H,et al. Deepvo: Towards end-to-end visual odometry with deep recurrent convolutional neural networks[ C] ∥2017 IEEE International Conference. New York: IEEE, 2017: 2043-2050.
- **VINet**：CLARK R,WANG S,WEN H,et al. VINet: Visual-inertial odometry as a sequence-to-sequence learning problem[C]∥ Thirty-First AAAI Conference on Artificial Intelligence. Washington D. C. : AIAA, 2017.
- **Deep EndoVO**：TURAN M, ALMALIOGLU Y, ARAUJO H, et al. Deep endovo: A recurrent convolutional neural network ( rcnn) based visual odometry approach for endoscopic capsule robots[ J] . Neurocomputing, 2018, 275: 1861-1870.





## 后期阅读

**中心思想**：

**论文写作架构**：

**对该论文的评价**：