# Mean Shift: A Ribust Approach Toward Feature Space Analysis



## 1. Introduction

LOW-LEVEL computer vision tasks are misleadingly difficult.
低级计算机视觉任务具有误导性的困难。

Incorrect results can be easily obtained since the employed techniques often rely upon the user correctly guessing the values for the tuning parameters.
由于所采用的技术通常依赖于用户正确猜测调整参数的值，因此很容易获得不正确的结果。

To improve performance ,the execution of low-level tasks should be task driven ,ie, supported by independent high-level information.
为了提高性能，低层任务的执行应该是任务驱动的，即由独立的高层信息支持。

This approach ,however ,requires that ,first ,the lowlevel stage provides a reliable enough representation of the input and that the feature extraction process be controlled only by very few tuning parameters corresponding to intuitive measures in the input domain.
然而，这种方法要求首先，低级阶段提供足够可靠的输入表示，并且特征提取过程仅由与输入域中的直观测量相对应的极少数调整参数控制。

Feature space-based analysis of images is a paradigm which can achieve the above-stated goals.
基于特征空间的图像分析是一种可以实现上述目标的范式。

A feature space is a mapping of the input obtained through the processing of the data in small subsets at a time.
特征空间是通过一次处理小子集中的数据获得的输入的映射。

For each subset ,a parametric representation of the feature of interest is obtained and the result is mapped into a point in the multidimensional space of the parameter.
对于每个子集，获得感兴趣特征的参数表示，并将结果映射到参数的多维空间中的一个点。

After the entire input is processed ,significant features correspond to denser regions in the feature space ,ie,to clusters ,and the goal of the analysis is the delineation of these clusters.
在处理完整个输入之后，重要的特征对应于特征空间中更密集的区域，即聚类，分析的目标是描绘这些聚类。

The nature of the feature space is application dependent.
特征空间的性质取决于应用程序。

The subsets employed in the mapping can range from individual pixels, as in the color space representation of an image, to a set of quasi-randomly chosen data points, as in the probabilistic Hough transform.
映射中使用的子集可以从单个像素（如图像的颜色空间表示）到一组准随机选择的数据点（如概率霍夫变换）。 

Both the advantage and the disadvantage of the feature space paradigm arise from the global nature of the derived representation of the input. 
特征空间范式的优点和缺点都源于输入的派生表示的全局性质。 

On one hand ,all the evidence for the presence of a significant feature is pooled together, providing excellent tolerance to a noise level which may render local decisions unreliable.
一方面，所有存在显着特征的证据都汇集在一起，对可能导致局部决策不可靠的噪声水平提供了极好的容忍度。 

On the other hand ,features with lesser support in the feature space may not be detected in spite of being salient for the task to be executed.
另一方面，尽管对于要执行的任务很重要，但在特征空间中支持较少的特征可能不会被检测到。

This disadvantage, however ,can be largely avoided by either augmenting the feature space with additional (spatial) parameters from the input domain or by robust postprocessing of the input domain guided by the results of the feature space analysis.
然而，通过使用来自输入域的附加（空间）参数来增加特征空间，或者在特征空间分析结果的指导下对输入域进行稳健的后处理，可以在很大程度上避免这个缺点。

Analysis of the feature space is application independent.
特征空间的分析与应用程序无关。

While there are a plethora of published clustering techniques ,most of them are not adequate to analyze feature spaces derived from real data.
虽然有大量已发表的聚类技术，但其中大多数不足以分析源自真实数据的特征空间。

Methods which rely upon a priori knowledge of the number of clusters present (including those which use optimization of a global criterion to find this number) ,as well as methods which implicitly assume the same shape (most often elliptical) for all the clusters in the space ,are not able to handle the complexity of a real feature space. For a recent survey of such methods ,see [29 ,Section 8].
依赖于存在的集群数量的先验知识的方法（包括那些使用全局标准优化来找到这个数量的方法），以及隐含地假设所有集群都具有相同形状（通常是椭圆形）的方法空间，无法处理真实特征空间的复杂性。 有关此类方法的最新调查，请参阅 [29，第 8 节]。

In Fig. 1 ,a typical example is shown. The color image in Fig. 1a is mapped into the three-dimensional L\*u\*v\* color space (to be discussed in Section 4). There is a continuous transition between the clusters arising from the dominant colors and a decomposition of the space into elliptical tiles will introduce severe artifacts. Enforcing a Gaussian mixture model over such data is doomed to fail ,e.g. ,[49] , and even the use of a robust approach with contaminated Gaussian densities [67] cannot be satisfactory for such complex cases. Note also that the mixture models require the number of clusters as a parameter ,which raises its own challenges. For example ,the method described in [45] proposes several different ways to determine this number.
图 1 显示了一个典型的例子。 图 1a 中的彩色图像被映射到三维 L\*u\*v\* 颜色空间（将在第 4 节中讨论）。 由主色产生的簇之间存在连续的过渡，并且将空间分解为椭圆形瓷砖将引入严重的伪影。 对此类数据执行高斯混合模型注定要失败，例如 ，[49]，甚至使用具有污染高斯密度的稳健方法 [67] 对于这种复杂的情况都不能令人满意。 还要注意，混合模型需要集群的数量作为参数，这带来了自己的挑战。 例如，[45] 中描述的方法提出了几种不同的方法来确定这个数字。

Arbitrarily structured feature spaces can be analyzed only by nonparametric methods since these methods do not have embedded assumptions.
任意结构的特征空间只能通过非参数方法进行分析，因为这些方法没有嵌入假设。

Numerous nonparametric clustering methods were described in the literature and they can be classified into two large classes: hierarchical clustering and density estimation.
文献中描述了许多非参数聚类方法，它们可以分为两大类：层次聚类和密度估计。

Hierarchical clustering techniques either aggregate or divide the data based on some proximity measure.
层次聚类技术是基于某种邻近性度量对数据进行聚合或分割。 

See [28, Section 3.2] for a survey of hierarchical clustering methods. 
参见[28，第3.2节]了解层次聚类方法的概况。 

The hierarchical methods tend to be computationally expensive and the definition of a meaningful stopping criterion for the fusion (or division) of the data is not straightforward.
分层方法的计算成本往往很高，而且为数据的融合（或分割）定义有意义的停止准则也不是很简单。

The rationale behind the density estimation-based nonparametric clustering approach is that the feature space can be regarded as the empirical probability density function (p.d.f.) of the represented parameter. 
基于密度估计的非参数聚类方法的基本原理是，特征空间可以视为表征参数的经验概率密度函数(p.d.f)。

Dense regions in the feature space thus correspond to local maxima of the p.d.f., that is ,to the modes of the unknown density. 
因此，特征空间中的密集区域对应于 p.d.f 的局部极大值，即未知密度的模态。

Once the location of a mode is determined ,the cluster associated with it is delineated based on the local structure of the feature space [25], [60], [63].
一旦确定了模式的位置，则根据特征空间的局部结构 [25]、[60]、[63] 来描绘与其相关联的簇。

Our approach to mode detection and clustering is based on the mean shift procedure, proposed in 1975 by Fukunaga and Hostetler [21] and largely forgotten until Cheng's paper [7] rekindled interest in it.
我们的模式检测和聚类方法基于 Fukunaga 和 Hostetler [21] 于 1975 年提出的均值偏移过程，直到 Cheng 的论文 [7] 重新引起了人们对它的兴趣之前，它基本上被遗忘了。

In spite of its excellent qualities ,the mean shift procedure does not seem to be known in statistical literature.
尽管均值偏移过程具有出色的品质，但在统计文献中似乎并不为人所知。

While the book [54 ,Section 6.2.2] discusses [21] ,the advantages of employing a mean shift type procedure in density estimation were only recently rediscovered [8].
虽然这本书 [54，第 6.2.2 节] 讨论了 [21]，但在密度估计中采用 mean shift 类型程序的优势直到最近才被重新发现 [8]。

As will be proven in the sequel ,a computational module based on the mean shift procedure is an extremely versatile tool for feature space analysis and can provide reliable solutions for many vision tasks. In Section 2 ,the mean shift procedure is defined and its properties are analyzed. In Section 3 ,the procedure is used as the computational module for robust feature space analysis and implementational issues are discussed. In Section 4 ,the feature space analysis technique is applied to two low-level vision tasks: discontinuity preserving filtering and image segmentation. Both algorithms can have as input either gray level or color images and the only parameter to be tuned by the user is the resolution of the analysis. The applicability of the mean shift procedure is not restricted to the presented examples. In Section 5 ,other applications are mentioned and the procedure is put into a more general context.
正如后续将证明的那样，基于均值偏移过程的计算模块是用于特征空间分析的极其通用的工具，可以为许多视觉任务提供可靠的解决方案。 在第 2 节中，定义了均值偏移过程并分析了它的性质。 在第 3 节中，该过程被用作鲁棒特征空间分析的计算模块，并讨论了实现问题。 在第 4 节中，特征空间分析技术应用于两个低级视觉任务：不连续性保持过滤和图像分割。 两种算法都可以将灰度或彩色图像作为输入，用户需要调整的唯一参数是分析的分辨率。 均值偏移过程的适用性不限于所提供的示例。 在第 5 节中，提到了其他应用程序，并将该过程置于更一般的上下文中。







## 3. Robust Analysis of Feature Spaces

Multimodality and arbitrarily shaped clusters are the defining properties of a real feature space.
多模态和任意形状的簇是一个真实特征空间的定义性质。

The quality of the mean shift procedure to move toward the mode (peak) of the hill on which it was initiated makes it the ideal computational module to analyze such spaces.
均值漂移过程向小山的模式(峰值)移动的特性使其成为分析此类空间的理想计算模块。

To detect all the significant modes, the basic algorithm given in Section 2.3 should be run multiple times (evolving in principle in parallel) with initializations that cover the entire feature space.
为了检测所有的显著模式，在2.3节中给出的基本算法应该运行多次(原则上是并行演化的)，并覆盖整个特征空间进行初始化。

Before the analysis is performed ,two important (and somewhat related) issues should be addressed: the metric of the feature space and the shape of the kernel.
在执行分析之前，应该解决两个重要（并且有些相关）的问题：特征空间的度量和内核的形状。 

The mapping from the input domain into a feature space often associates a non-Euclidean metric to the space.
从输入域到特征空间的映射通常将非欧几里德度量与空间相关联。 

The problem of color representation will be discussed in Section 4, but the employed parameterization has to be carefully examined even in a simple case like the Hough space of lines, eg, [48], [61].
颜色表示的问题将在第 4 节中讨论，但是即使在像线的霍夫空间这样的简单情况下，也必须仔细检查所采用的参数化，例如 ，[48]，[61]。

The presence of a Mahalanobis metric can be accommodated by an adequate choice of the bandwidth matrix (2).
可以通过足够的带宽矩阵（2）来容纳Mahalanobis度量的存在。

In practice, however, it is preferable to have assured that the metric of the feature space is Euclidean and, thus, the bandwidth matrix is controlled by a single parameter, $H=h^2I$.
然而，在实践中，最好确保特征空间的度量是欧几里得，因此，带宽矩阵由单个参数控制，$H=h^2I$。

To be able to use the same kernel size for all the mean shift procedures in the feature space, the necessary condition is that local density variations near a significant mode are not as large as the entire support of a significant mode somewhere else.
为了能够在特征空间中对所有的 mean shift 过程使用相同的核大小，必要的条件是，在一个显著模态附近的局部密度变化不像在其他地方对一个显著模态的整个支持那么大。

The starting points of the mean shift procedures should be chosen to have the entire feature space (except the very sparse regions) tessellated by the kernels (windows).
mean shift 过程的起始点应该选择使整个特征空间（除了非常稀疏的区域）由内核（窗口）嵌合。

Regular tessellations are not required.
不需要规则的嵌合。

As the windows evolve toward the modes, almost all the data points are visited and, thus, all the information captured in the feature space is exploited.
随着窗口向模式演变，几乎所有的数据点都被访问，因此，所有在特征空间中捕获的信息都被利用。

Note that the convergence to a given mode may yield slightly different locations due to the threshold that terminates the iterations.
请注意，由于终止迭代的阈值，收敛到给定模式可能会产生稍微不同的位置。

Similarly, on flat plateaus ,the value of the gradient is close to zero and the mean shift procedure could stop.
同样，在平坦的高原上，梯度值接近于零，mean shift 过程可能会停止。

These artifacts are easy to eliminate through postprocessing.
这些伪影很容易通过后处理消除。

Mode candidates at a distance less than the kernel bandwidth are fused, the one corresponding to the highest density being chosen.
在距离小于核带宽的候选模式进行融合，选择对应于最高密度的模式。

The global structure of the feature space can be confirmed by measuring the significance of the valleys defined along a cut through the density in the direction determined by two modes.
特征空间的整体结构可以通过测量沿两模态确定的方向穿过密度的切线定义的谷的重要性来确定。

The delineation of the clusters is a natural outcome of the mode seeking process.
集群的划分是模式寻找过程的自然结果。

After convergence, the basin of attraction of a mode, ie, the data points visited by all the mean shift procedures converging to that mode, automatically delineates a cluster of arbitrary shape.
收敛后，某一模态的吸引域，即收敛到该模态的所有 mean shift 过程所访问的数据点，自动地描绘出任意形状的簇。

Close to the boundaries, where a data point could have been visited by several diverging procedures, majority logic can be employed.
在接近边界的地方，一个数据点可以被几个不同的程序访问，多数逻辑（投票机制）可以被使用。

It is important to notice that, in computer vision, most often we are not dealing with an abstract clustering problem.
值得注意的是，在计算机视觉中，我们通常不是在处理抽象的聚类问题。

The input domain almost always provides an independent test for the validity of local decisions in the feature space.
输入域几乎总是为特征空间中局部决策的有效性提供一个独立的检验。

That is, while it is less likely that one can recover from a severe clustering error, allocation of a few uncertain data points can be reliably supported by input domain information.
也就是说，虽然从严重的聚类错误中恢复的可能性较小，但输入域信息可以可靠地支持一些不确定数据点的分配。

The multimodal feature space analysis technique was discussed in detail in [12].
多模态特征空间分析技术在[12]中有详细讨论。

It was shown experimentally, that for a synthetic, bimodal normal distribution, the technique achieves a classification error similar to the optimal Bayesian classifier.
 实验表明，对于一个合成的双峰正态分布，该技术实现了类似于最优贝叶斯分类器的分类误差。

The behavior of this feature space analysis technique is illustrated in Fig. 2.
这种特征空间分析技术的行为如图 2 所示。

A two-dimensional data set of 110,400 points (Fig. 2a) is decomposed into seven clusters represented with different colors in Fig. 2b.
一个 110,400 个点的二维数据集（图 2a）被分解为图 2b 中用不同颜色表示的七个簇。

A number of 159 mean shift procedures with uniform kernel were employed.
采用了 159 种具有一致内核的 mean shift 方法。

Their trajectories are shown in Fig. 2c, overlapped over the density estimate computed with the Epanechnikov kernel.
它们的轨迹如图 2c 所示，与 Epanechnikov 核计算的密度估计重叠。

The pruning of the mode candidates produced seven peaks.
对候选模态的修剪产生了7个峰。

Observe that some of the trajectories are prematurely stopped by local plateaus.
观察到一些轨迹被局部高原过早地停止了。

### 3.1 Bandwidth Selection

The influence of the bandwidth parameter h was assessed empirically in [12] through a simple image segmentation task.
带宽参数 h 的影响在 [12] 中通过一个简单的图像分割任务进行了经验评估。

In a more rigorous approach ,however ,four different techniques for bandwidth selection can be considered.
然而，在更严格的方法中，可以考虑四种不同的带宽选择技术。

- The first one has a statistical motivation.第一个具有统计动机。
- The optimal bandwidth associated with the kernel density estimator (6) is defined as the bandwidth that achieves the best compromise between the bias and variance of the estimator, over all $x \in R^d$, ie, minimizes AMISE.与核密度估计器 (6) 相关的最佳带宽被定义为在所有 $x \in R^d$ 上实现估计器的偏差和方差之间最佳折衷的带宽，即最小化 AMISE。
- In the multivariate case, the resulting bandwidth formula [54, p. 85], [62, p. 99] is of little practical use, since it depends on the Laplacian of the unknown density being estimated, and its performance is not well understood [62, p. 108].在多元情况下，得到的带宽公式 [54, p. 85]，[62，页. 99] 没有什么实际用途，因为它依赖于被估计的未知密度的拉普拉斯算子，而且它的性能还不是很清楚 [62, p. 108]。 
- For the univariate case, a reliable method for bandwidth selection is the plug-in rule [53], which was proven to be superior to leastsquares cross-validation and biased cross-validation [42], [55, p. 46].对于单变量情况，一种可靠的带宽选择方法是插件规则[53]，它被证明优于最小二乘交叉验证和有偏交叉验证[42]，[55，p. 46]。
- Its only assumption is the smoothness of the underlying density.它唯一的假设是底层密度的平滑度。
- The second bandwidth selection technique is related to the stability of the decomposition.第二个带宽选择技术与分解的稳定性有关。
- The bandwidth is taken as the center of the largest operating range over which the same number of clusters are obtained for the given data [20, p. 541].带宽被视为最大操作范围的中心，在该范围内为给定数据获得相同数量的集群。[20, p. 541]。
- For the third technique ,the best bandwidth maximizes an objective function that expresses the quality of the decomposition (ie, the index of cluster validity).对于第三种技术，最佳带宽最大化表示分解质量的目标函数（即集群有效性指数）。
- The objective function typically compares the inter- versus intra-cluster variability [30], [28] or evaluates the isolation and connectivity of the delineated clusters [43].目标函数通常比较集群间和集群内的可变性 [30]、[28] 或评估所描绘集群的隔离性和连通性 [43]。
- Finally, since in most of the cases the decomposition is task dependent, top-down information provided by the user or by an upper-level module can be used to control the kernel bandwidth.最后，由于在大多数情况下，分解是依赖于任务的，因此用户或上层模块提供的自顶向下信息可用于控制内核带宽。

We present in [15] ,a detailed analysis of the bandwidth selection problem.
我们在 [15] 中详细分析了带宽选择问题。

To solve the difficulties generated by the narrow peaks and the tails of the underlying density ,two locally adaptive solutions are proposed.
为解决底层密度的窄峰和窄尾带来的困难，提出了两种局部自适应的解决方案。

One is nonparametric, being based on a newly defined adaptive mean shift procedure, which exploits the plug-in rule and the sample point density estimator.
。一种是非参数的，基于新定义的自适应 mean shift 过程，该过程利用插件规则和样本点密度估计器。

The other is semiparametric, imposing a local structure on the data to extract reliable scale information.
另一种是半参数，对数据施加局部结构以提取可靠的尺度信息。

We show that the local bandwidth should maximize the magnitude of the normalized mean shift vector.
我们表明，局部带宽应该使归一化 mean shift 向量的幅度最大化。

The adaptation of the bandwidth provides superior results when compared to the fixed bandwidth procedure. For more details ,see [15].
与固定带宽程序相比，带宽的调整提供了更好的结果。 有关详细信息，请参阅 [15]。

### 3.2 Implementation Issues

An efficient computation of the mean shift procedure first requires the resampling of the input data with a regular grid.
mean shift 过程的有效计算首先需要使用规则网格对输入数据进行重采样。

This is a standard technique in the context of density estimation which leads to a binned estimator [62, Appendix D].
这是密度估计背景下的一种标准技术，它导致了一个分级估计器[62，附录 D]。 

The procedure is similar to defining a histogram where linear interpolation is used to compute the weights associated with the grid points.
该过程类似于定义直方图，其中线性插值用于计算与网格点相关的权重。

Further reduction in the computation time is achieved by employing algorithms for multidimensional range searching [52, p. 373] used to find the data points falling in the neighborhood of a given kernel.
通过采用多维范围搜索算法可以进一步减少计算时间 [52, p. 373] 用于查找落在给定内核附近的数据点。

For the efficient Euclidean distance computation, we used the improved absolute error inequality criterion, derived in [39].
对于有效的欧几里得距离计算，我们使用了改进的绝对误差不等式准则，该准则源自 [39]。

## 4. Application

The feature space analysis technique introduced in the previous section is application independent and, thus, can be used to develop vision algorithms for a wide variety of tasks.
上一节介绍的特征空间分析技术独立于应用程序，因此可用于开发各种任务的视觉算法。

Two somewhat related applications are discussed in the sequel: discontinuity preserving smoothing and image segmentation.
下面讨论两个有些相关的应用：不连续性保持平滑和图像分割。

The versatility of the feature space analysis enables the design of algorithms in which the user controls performance through a single parameter, the resolution of the analysis (ie, bandwidth of the kernel).
特征空间分析的通用性使用户能够通过单个参数，即分析的分辨率（即内核的带宽）来控制算法的性能。

Since the control parameter has clear physical meaning, the new algorithms can be easily integrated into systems performing more complex tasks.
由于控制参数具有明确的物理意义，新算法可以很容易地集成到执行更复杂任务的系统中。

Furthermore, both gray level and color images are processed with the same algorithm, in the former case, the feature space containing two degenerate dimensions that have no effect on the mean shift procedure.
此外，灰度图像和彩色图像使用相同的算法进行处理，在灰度图像中，特征空间包含两个退化维，对 mean shift 过程没有影响。

Before proceeding to develop the new algorithms, the issue of the employed color space has to be settled.
在着手开发新算法之前，必须解决所使用的色彩空间问题。

To obtain a meaningful segmentation, perceived color differences should correspond to Euclidean distances in the color space chosen to represent the features (pixels).
为了获得有意义的分割，感知到的颜色差异应该对应于用于代表特征（像素）的颜色空间中的欧氏距离。

An Euclidean metric, however, is not guaranteed for a color space [65, Sections 6.5.2, 8.4].
但是，对于颜色空间 [65，第 6.5.2，8.4]，不能保证欧几里德度量。

The spaces L\*u\*v\* and L\*a\*b\* were especially designed to best approximate perceptually uniform color spaces.
空间 L\*u\*v\* 和 L\*a\*b\* 专门设计用于最接近感知均匀的颜色空间。

In both cases, L\*, the lightness (relative brightness) coordinate, is defined the same way, the two spaces differ only through the chromaticity coordinates.
在这两种情况下，亮度（相对亮度）坐标 L\* 的定义是相同的，两个空间的区别只是色度坐标不同。

The dependence of all three coordinates on the traditional RGB color values is nonlinear. See [46, Section 3.5] for a readily accessible source for the conversion formulae.
所有三个坐标对传统 RGB 颜色值的依赖是非线性的。请参阅[46，第3.5节]，以获得易于获取的换算公式来源。

The metric of perceptually uniform color spaces is discussed in the context of feature representation for image segmentation in [16].
在[16]中图像分割的特征表示的上下文中讨论了感知均匀颜色空间的度量。

In practice, there is no clear advantage between using L\*u\*v\* or L\*a\*b\*; in the proposed algorithms ,we employed L\*u\*v\* motivated by a linear mapping property [65, p.166].
在实践中，使用 L\*u\*v\* 或 L\*a\*b\* 之间没有明显的优势； 在提出的算法中，我们采用了 L\*u\*v\*，其动机是线性映射属性 [65, p.166]。

Our first image segmentation algorithm was a straightforward application of the feature space analysis technique to an L\*u\*v\* representation of the color image [11].
我们的第一个图像分割算法是将特征空间分析技术直接应用于彩色图像的 L\*u\*v\* 表示 [11]。

The modularity of the segmentation algorithm enabled its integration by other groups to a large variety of applications like image retrieval [1], face tracking [6], object-based video coding for MPEG-4 [22], shape detection and recognition [33], and texture analysis [47], to mention only a few.
分割算法的模块化使其能够被其他组集成到各种各样的应用中，如图像检索[1]，人脸跟踪[6]，基于对象的视频编码MPEG-4[22]，形状检测和识别[33]，纹理分析[47]，这只是举几个例子。

However, since the feature space analysis can be applied unchanged to moderately higher dimensional spaces (see Section 5), we subsequently also incorporated the spatial coordinates of a pixel into its feature space representation.
然而，由于特征空间分析可以应用于适度高维的空间(见第5节)，我们随后也将像素的空间坐标纳入其特征空间表示。

This joint domain representation is employed in the two algorithms described here.
此处描述的两种算法都采用了这种联合域表示。

An image is typically represented as a two-dimensional lattice of p-dimensional vectors (pixels) ,where $p=1$ in the gray-level case ,three for color images, and $p>3$ in the multispectral case.
图像通常表示为 $p$ 维向量（像素）的二维晶格，其中在灰度情况下$p=1$，在彩色情况下为 $p=3$，在多光谱情况下为 $p>3$。

The space of the lattice is known as the spatial domain, while the gray level, color, or spectral information is represented in the range domain.
晶格的空间称为空间域，而灰度、颜色或光谱信息则在范围域中表示。

For both domains, Euclidean metric is assumed.
对于这两个域，都假定欧几里德度量。

When the location and range vectors are concatenated in the joint spatial-range domain of dimension $d=p+2$, their different nature has to be compensated by proper normalization.
当在 $d=p+2$ 的联合 空间-范围 域中对位置向量和距离向量进行级联时，必须通过适当的归一化来补偿它们的不同性质。

Thus, the multivariate kernel is defined as the product of two radially symmetric kernels and the Euclidean metric allows a single bandwidth parameter for each domain: 
因此，多元核被定义为两个径向对称核的乘积，而欧氏度量允许每个域只有一个带宽参数：

$$K_{h_s,h_r}(\mathbf{x})=\frac{C}{h^2_s h^p_r}k({\left \| \frac{\mathbf{x}^s}{h_s} \right \|}^2) k({\left \| \frac{\mathbf{x}^r}{h_r} \right \|}^2) \tag{35}$$

where $\mathbf{x}^s$ is the spatial part, $\mathbf{x}^r$ is the range part of a feature vector, $k(x)$ the common profile used in both two domains, $h_s$ and $h_r$ the employed kernel bandwidths, and $C$ the corresponding normalization constant.
其中 $\mathbf{x}^s$ 是空间部分，$\mathbf{x}^r$ 是特征向量的范围部分，$k(x)$ 是两个域中使用的通用配置函数，$h_s $ 和 $h_r$ 是使用的内核带宽，$C$ 是相应的归一化常数。

In practice, an Epanechnikov or a (truncated) normal kernel always provides satisfactory performance, so the user only has to set the bandwidth parameter $\mathbf{h}=(h_s,h_r)$, which, by controlling the size of the kernel, determines the resolution of the mode detection.
在实践中，Epanechnikov 或（截断的）普通内核总是提供令人满意的性能，因此用户只需设置带宽参数 $\mathbf{h}=(h_s,h_r)$，通过控制内核的大小 , 决定模式检测的分辨率。





### 4.2 Image Segmentation

Image segmentation, decomposition of a gray level or color image into homogeneous tiles, is arguably the most important low-level vision task.
图像分割，将灰度或彩色图像分解为同质块，可以说是最重要的低级视觉任务。

Homogeneity is usually defined as similarity in pixel values, ie, a piecewise constant model is enforced over the image.
同质性通常定义为像素值的相似性，即在图像上强制执行分段常数模型。

From the diversity of image segmentation methods proposed in the literature, we will mention only some whose basic processing relies on the joint domain.
从文献中提出的图像分割方法的多样性来看，我们只会提到一些基本处理依赖于联合域的方法。

In each case, a vector field is defined over the sampling lattice of the image.
在每种情况下，都在图像的采样点阵上定义了一个矢量场。

The attraction force field defined in [57] is computed at each pixel as a vector sum of pairwise affinities between the current pixel and all other pixels, with similarity measured in both spatial and range domains.
[57] 中定义的引力场在每个像素处计算为当前像素与所有其他像素之间的成对亲和力的矢量和，在空间域和范围域中测量相似度。

The region boundaries are then identified as loci where the force vectors diverge.
然后将区域边界识别为力矢量发散的位置。

It is interesting to note that, for a given pixel ,the magnitude and orientation of the force field are similar to those of the joint domain mean shift vector computed at that pixel and projected into the spatial domain.
值得注意的是，对于给定的像素，力场的大小和方向与在该像素处计算并投影到空间域的联合域 mean shift 向量的大小和方向相似。

However, in contrast to [57], the mean shift procedure moves in the direction of this vector, away from the boundaries.
然而，与 [57] 相比，mean shift 过程沿该向量的方向移动，远离边界。

The edge flow in [34] is obtained at each location for a given set of directions as the magnitude of the gradient of a smoothed image.
[34] 中的边缘流是在给定一组方向的每个位置处获得的，作为平滑图像的梯度大小。

The boundaries are detected at image locations which encounter two opposite directions of flow.
在遇到两个相反流动方向的图像位置检测边界。

The quantization of the edge flow direction ,however ,may introduce artifacts.
然而，边缘流动方向的量化可能会引入伪影。

Recall that the direction of the mean shift is dictated solely by the data.
回想一下，均值偏移的方向仅由数据决定。

The mean shift procedure-based image segmentation is a straightforward extension of the discontinuity preserving smoothing algorithm.
基于均值偏移过程的图像分割是不连续性保持平滑算法的直接扩展。

Each pixel is associated with a significant mode of the joint domain density located in its neighborhood, after nearby modes were pruned as in the generic feature space analysis technique (Section 3).
在像通用特征空间分析技术（第3节）那样对附近的模态进行修剪后，每个像素都与位于其邻域的联合域密度的显著模态相关联。

### 4.2.1 Mean Shift Segmentation

Let $\mathbf{x}_i$ and $\mathbf{z}_i$, $i=1,...,n$, be the d-dimensional input and filtered image pixels in the joint spatial-range domain and $L_i$ the label of the $i$-th pixel in the segmented image.
令 $\mathbf{x}_i$ 和 $\mathbf{z}_i$, $i=1,...,n$ 是联合空间范围域中的 d 维输入和滤波图像像素，$ L_i$ 为分割图像中第 $i$ 个像素的标签。

1. Run the mean shift filtering procedure for the image and store all the information about the d-dimensional convergence point in $\mathbf{z}_i$, ie, $\mathbf{z}_i=\mathbf{y}_{i,c}$.对图像进行 mean shift 滤波，将 d 维收敛点的所有信息存储在 $\mathbf{z}_i$ 中，即 $\mathbf{z}_i=\mathbf{y}_{i,c}$ 中。

2. Delineate in the joint domain the clusters $\{\mathbf{C}_p\}_{p=1...m}$ by grouping together all $\mathbf{z}_i$ which are closer than $h_s$ in the spatial domain and $h_r$ in the range domain, ie, concatenate the basins of attraction of the corresponding convergence points.在联合域中描述集群 $\{\mathbf{C}_p\}_{p=1…M}$ 通过将空间域上比 $h_s$ 和范围域上比 $h_r$ 更接近的  $\mathbf{z}_i$ 组合在一起，即将相一致的收敛点引力场连接起来。

3. For each $i=1,...,n$, assign $L_i = \{p | \mathbf{z}_i \in \mathbf{C}_p\}$.对于每个 $i=1,... ,n$，分配 $L_i = \{p | \mathbf{z}_i \in \mathbf{C}_p\}$。

4. Optional: Eliminate spatial regions containing less than $M$ pixels.可选：消除包含小于$M$像素的空间区域。

The cluster delineation step can be refined according to a priori information and, thus, physics-based segmentation algorithms, eg, [2], [35], can be incorporated. 
可以根据先验信息细化集群描绘步骤，因此可以结合基于物理的分割算法，例如[2]、[35]。

Since this process is performed on region adjacency graphs, hierarchical techniques like [36] can provide significant speed-up.
由于这个过程是在区域邻接图上执行的，像[36]这样的分层技术可以提供显著的加速。

The effect of the cluster delineation step is shown in Fig. 4d.
聚类描绘步骤的效果如图 4d 所示。

Note the fusion into larger homogeneous regions of the result of filtering shown in Fig. 4c.
请注意图 4c 中显示的过滤结果融合到更大的均匀区域中。

The segmentation step does not add a significant overhead to the filtering process.
分割步骤不会给过滤过程增加很大的开销。

The region representation used by the mean shift segmentation is similar to the blob representation employed in [64].
mean shift 分割使用的区域表示类似于 [64] 中使用的 blob 表示。

However, while the blob has a parametric description (multivariate Gaussians in both spatial and color domain), the partition generated by the mean shift is characterized by a nonparametric model.
然而，虽然 blob 具有参数描述（空间和颜色域中的多元高斯），但由 mean shift 生成的分区由非参数模型表征。

An image region is defined by all the pixels associated with the same mode in the joint domain.
一个图像区域由联合域中与同一模式相关联的所有像素定义。

In [43], a nonparametric clustering method is described in which, after kernel density estimation with a small bandwidth, the clusters are delineated through concatenation of the detected modes' neighborhoods.
在[43]中，描述了一种非参数聚类方法，其中，在用小带宽进行核密度估计之后，通过连接检测到的模式的邻域来描绘聚类。

The merging process is based on two intuitive measures capturing the variations in the local density.
合并过程基于两种直观的测量方法，捕捉了局部密度的变化。

Being a hierarchical clustering technique, the method is computationally expensive; it takes several minutes in MATLAB to analyze a 2,000 pixel subsample of the feature space.
作为一种层次聚类技术，该方法计算量大；在MATLAB中分析2000像素的特征空间子样本需要几分钟的时间。

The method is not recommended to be used in the joint domain since the measures employed in the merging process become ineffective.
由于合并过程中采用的措施变得无效，因此不建议在联合域中使用该方法。

Comparing the results for arbitrarily shaped synthetic data [43, Fig. 6] with a similarly challenging example processed with the mean shift method [12, Fig. 1] shows that the use of a hierarchical approach can be successfully avoided in the nonparametric clustering paradigm.
将任意形状的合成数据 [43，图6] 与用 mean shift 方法处理的同样具有挑战性的例子 [12，图1] 的结果进行比较，表明在非参数聚类范式中可以成功地避免使用分层方法。

All the segmentation experiments were performed using uniform kernels.
所有的分割实验都是使用统一的内核进行的。

The improvement due to joint space analysis can be seen in Fig. 6 where the $256 \times 256$ gray-level image $MIT$ was processed with $(h_s, h_r, M)=(8, 7, 20)$.
由于联合空间分析的改进可以在图 6 中看到，其中 $256 \times 256$ 灰度图像 $MIT$ 是用 $(h_s, h_r, M)=(8, 7, 20)$ 处理的。

A number of $225$ homogeneous regions were identified in fractions of a second, most of them delineating semantically meaningful regions like walls, sky, steps, inscription on the building, etc.
在几分之一秒内确定了 225 个同质区域，其中大多数描绘了具有语义意义的区域，如墙壁、天空、台阶、建筑物上的铭文等。

Compare the results with the segmentation obtained by one-dimensional clustering of the gray-level values in [11, Fig. 4] or by using a Gibbs random fieldsbased approach [40, Fig. 7].
将结果与通过 [11，图 4] 中的灰度值的一维聚类或使用基于吉布斯随机场的方法 [40，图 7] 获得的分割结果进行比较。

The joint domain segmentation of the color $256 \times 256$ room image presented in Fig. 7 is also satisfactory.
图 7 中呈现的彩色 $256 \times 256$ 房间图像的联合域分割也令人满意。

Compare this result with the segmentation presented in [38, Figs. 3e and 5c] obtained by recursive thresholding.
将此结果与 [38，图 3e和5c]通过递归阈值获得。

In both these examples, one can notice that regions in which a small gradient of illumination exists (like the sky in theMITor the carpet in the roomimage) were delineated as a single region.
在这两个示例中，人们都可以注意到存在小梯度照明的区域（如 MIT 中的天空或房间图像中的地毯）被描绘为单个区域。

Thus, the joint domain mean shift-based segmentation succeeds in overcoming the inherent limitations of methods based only on gray-level or color clustering which typically oversegment small gradient regions.
因此，基于联合域均值偏移的分割成功地克服了仅基于灰度或颜色聚类的方法的固有局限性，这些方法通常会过度分割小梯度区域。

