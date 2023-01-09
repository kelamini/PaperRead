





## An ordinary object detector is composed of several parts（一个标准的目标检测器包含以下部分）:

- **Input**（输入）: 
	- Image, 
	- Patches, 
	- Image Pyramid

- **Backbones**（骨干）: 
	- VGG16
	- ResNet-50
	- SpineNet
	- EfficientNet-B0/B7
	- CSPResNeXt50
	- CSPDarknet53

- **Neck**（颈部）:
	- Additional blocks（附加块）: 
		- SPP
		- ASPP
		- RFB
		- SAM
	- Path-aggregation blocks（路径聚合块）: 
		- FPN
		- PAN
		- NAS-FPN
		- Fully-connected FPN
		- BiFPN
		- ASFF
		- SFAM

- **Heads**（头部）:
	- Dense Prediction (one-stage)（密集预测）:
		- RPN
		- SSD
		- YOLO
		- RetinaNet (anchor based)
		- CornerNet
		- CenterNet
		- MatrixNet
		- FCOS (anchor free)
	- Sparse Prediction (two-stage)（稀疏预测）:
		- Faster R-CNN
		- R-FCN
		- Mask R-CNN (anchor based) 
		- RepPoints (anchor free)





## For improving the object detection training, a CNN usually uses the following（为了提高目标检测训练的速度和精度，一个 CNN 网络通常使用下列方法）:

- **Activations**（激活函数）: 
	- ReLU
	- leaky-ReLU
	- parametric-ReLU
	- ReLU6
	- SELU
	- Swish
	- Mish
- **Bounding box regression loss**（Bounding box 回归损失）: 
	- MSE
	- IoU
	- GIoU
	- CIoU
	- DIoU
- **Data augmentation**（数据增强）: 
	- CutOut
	- MixUp
	- CutMix
- **Regularization method**（正则化方法）: 
	- DropOut
	- DropPath
	- Spatial DropOut
	- DropBlock
- **Normalization of the network activations by their mean and variance**（通过均值和方差标准化网络激活）: 
	- Batch Normalization (BN)
	- Cross-GPU Batch Normalization (CGBN or SyncBN)
	- Filter Response Normalization (FRN)
	- Cross-Iteration Batch Normalization (CBN)
- **Skip-connections**（跳连）: 
	- Residual connections
	- Weighted residual connections (WRC)
	- Multi-input weighted residual connections
	- Cross stage partial connections (CSP)





## Additional improvements（进一步改进）:

- Mosaic data augmentation —— Mosaic 数据增强
- Self-adversarial-training（SAT） —— 自我对抗训练
- Cross mini-Batch Normalization（CmBN）—— 交叉小批量标准化
- select optimal hyper-parameters while applying genetic algorithms —— 应用遗传算法选取最优超参数





## YOLOv4:

- YOLOv4 consists of（整体架构）:
	- **Backbone**: CSPDarknet53
	- **Neck**: SPP and PAN
	- **Head**: YOLOv3

- YOLOv4 uses（细节）:
	- Bag of Freebies (BoF) for backbone: 
		- CutMix and Mosaic data augmentation
		- DropBlock regularization
		- Class label smoothing
	
	- Bag of Specials (BoS) for backbone: 
	    - Mish activation
	    - Cross-stage partial connections (CSP)
	    - Multi-input weighted residual connections (MiWRC)
	
	- Bag of Freebies (BoF) for detector: 
	    - CIoU-loss
	    - CmBN
	    - DropBlock regularization
	    - Mosaic data augmentation
	    - Self-Adversarial Training
	    - Eliminate grid sensitivity
	    - Using multiple anchors for a single ground truth
	    - Cosine annealing scheduler
	    - Optimal hyper-parameters
	    - Random training shapes
	
	- Bag of Specials (BoS) for detector: 
	    - Mish activation
	    - SPP-block
	    - SAM-block
	    - PAN path-aggregation block
	    - DIoU-NMS