
#### pytorch是什么:
PyTorch是一个开源的深度学习框架，主要用于开发和训练各种机器学习模型。GPU加速张量计算，通过动态计算图实现高效的模型构建与调试。自动微分实现梯度计算，支持分布式训练，拥有丰富的预训练模型库，如torchvision，torchtext，Torchaudio，所以适用于计算机视觉，nlp，音频方面

#### pytorch，tensorflow，paddle区别：
1. **PyTorch**：
    - 动态计算图(Eager Mode)
    - 提供了丰富的预训练模型库，如TorchVision、TorchText等。
    - 广泛应用于学术研究中，因其灵活性而受到研究人员的青睐。
    - 支持Python优先的编程体验
    - 可以分布式，torch.distributed包，支持单机多卡（数据并行）、多机多卡的分布式训练功能。支持多种后端，如NCCL（针对NVIDIA GPU优化）、Gloo（适用于CPU和GPU环境）
2. **TensorFlow**：
    - 最初以静态计算图为主要特征，后来也加入了Eager Execution模式，因为静态，所以部署稳定
    - 拥有强大的分布式训练支持
    - 跨平台部署，通过TensorFlow Serving或TensorFlow Lite进行部署移动端和嵌入式系统部署，以及tensorflow.js支持web端应用开发
    - 被广泛用于工业界的大规模应用开发：易部署，分布式，静态计算图，
    - 提供了多种高级API，比如Keras，简化了模型构建过程。
3. **PaddlePaddle**：
    - 百度开源的深度学习平台，提供了全面的中文文档和支持，非常适合中文社区使用。
    - 包含多个模块化的组件，包括PaddleCV、PaddleNLP等，针对不同应用场景提供解决方案。
    - 强调易用性和效率，提供了简单直观的API接口，降低了深度学习入门门槛,支持动态图模式
    - 提供了专门针对大规模数据处理和高性能计算优化的功能,,支持多种硬件加速器，包括NVIDIA GPU、AMD GPU、FPGA等
    - 提供了一系列面向企业用户的工具和服务，如模型部署服务Paddle Serving

#### 数据量不够：
- [[数据增强]]
- 微调：使用已经在一个大数据集上预训练好的模型作为起点，并针对你的任务进行微调
- 合成新数据：可以使用GANs（生成对抗网络）来生成与现有数据相似但又不完全相同的新图像。
- 正则化：采用L2正则化、Dropout等技术可以防止模型过拟合少量的数据
- 外部数据集：寻找并利用公开的外部数据集来扩充你的训练数据
- 减少模型复杂度：设计一个较简单、参数较少的模型可能更为合适,复杂的模型容易在小数据集上过拟合

#### 解决过拟合
- 训练数据不足：数据增强
- 解决迭代次数过多：早停法
- dropout:随机去除一些神经元，将输出设置为0，保持其他部分不变
- [[正则化]]

####  模型调优方法
- 超参数优化：
- 正则化
- 数据增强
- 早停法（Early Stopping）
- 批量归一化（Batch Normalization）：对每一层输入的数据进行归一化处理，帮助加速训练过程并减少对初始化的依赖。
- 迁移学习与微调：利用预训练模型的知识作为起点，并针对特定任务进行微调，特别是在数据量有限的情况下非常有效。
- 集成学习
- 调整网络架构
- 学习率调整策略：采用动态调整学习率的方法（如学习率衰减、周期性学习率变化等）可以帮助模型更快收敛且达到更好的效果。

#### 礼帽与黑帽
腐蚀：自身被背景腐蚀，缩小了
![[Pasted image 20250304105818.png]]

膨胀，自身变大了
![[Pasted image 20250304105951.png]]

开运算：先腐蚀后膨胀，消除小物体，在纤细点处分离物体，并且在平滑较大物体边界的同时不明显改变其面积。
`opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)`

闭运算：先膨胀后腐蚀，用来填充小孔洞、连接邻近物体以及平滑物体边界而不明显改变其面积。
`closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)`

礼帽：原图像减去开运算的结果，可以提取出比周围更亮的斑点，适合于突出显示图像中那些比其周围环境更明亮的区域，腐蚀后消失的内容
`tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)`

黑帽：原图像减去其闭运算的结果。此操作可用于检测图像中比周围更暗的斑点。膨胀后消失的内容

#### 边缘检测算法
##### 边缘检测
- sobel算子：通过在x方向和y方向分别使用两个3x3的卷积核来近似计算图像的梯度，从而确定边缘。特点：简单易实现，对噪声有一定的抑制作用。
```
import cv2
import numpy as np

img = cv2.imread('image.jpg', 0)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x方向
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y方向
```
- canny边缘检测：是一个多阶段的边缘检测算法，包括噪声减少、梯度计算、非极大值抑制和双阈值筛选等步骤，以产生更精确的边缘。特点：相比其他方法能提供更好的边缘检测结果，但相对复杂一些。 
```
edges = cv2.Canny(img, 100, 200)
```
- laplacian算子：基于二阶导数的边缘检测方法，它利用拉普拉斯变换来查找图像中的零交叉点，以此来确定边缘位置。特点：对噪声敏感，通常需要先进行平滑处理（如高斯模糊）。
```
laplacian = cv2.Laplacian(img, cv2.CV_64F)
```


#### 聚类算法
- k-means聚类：
	- 步骤：
	1.在数据集中随机选取k个中心点
	2.分别计算每个数据点到k个中心点的距离，根据距离对该数据点进行分类。
	3.计算同类数据点的中点作为待更新的该类中心点位置。
	4.更新中心点，重复步骤二，若每个数据点与其所属类中心点的距离之和不再变化，则算法结束。
- k-meadns++:因为k-means初始随机选取k中心点，所以导致收敛很慢
	- 步骤：
		- 随机选择一个样本点为聚类中心
		- 计算其他点到聚类中心距离，选择第二个聚类中心，根据距离越大，概率越大
		- 直到选择完K个聚类中心，再使用k-means算法
#### 数据集图片有倾斜，光亮不好
-  对于倾斜的图片：
	- **霍夫变换**：应用霍夫变换来检测直线，并计算这些直线的角度，以此来确定图像需要旋转的角度。
	- **仿射变换**：根据计算出的角度，使用仿射变换对图像进行旋转校正
- 对于光亮不好的图片：
	- **直方图均衡化**：通过调整图像的对比度来改进亮度分布。这在增强图像细节方面特别有效。
	- **伽马校正**：这是一种用于控制图像亮度和对比度的方法，可以通过调整伽马值来改变图像的亮度。
	- **自适应阈值处理**：对于二值化处理特别有用，能够根据不同局部区域的亮度自动选择最佳阈值。
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像并转换为灰度图
image = cv2.imread('your_image_path.jpg', 0)

# 应用自适应阈值处理
# adaptiveThreshold函数参数说明：
# src：源图像（需为灰度图）
# maxValue：分配给符合阈值条件的像素的最大值
# adaptiveMethod：自适应方法，可以选择ADAPTIVE_THRESH_MEAN_C或ADAPTIVE_THRESH_GAUSSIAN_C
# thresholdType：阈值类型，通常使用THRESH_BINARY或THRESH_BINARY_INV
# blockSize：用于计算阈值的邻域大小，必须是奇数且大于1
# C：从计算的均值或加权均值中减去的常数
adaptive_threshold_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                 cv2.THRESH_BINARY, 11, 2)

# 显示原始图像和处理后的图像
plt.figure(figsize=(10, 7))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(adaptive_threshold_image, cmap='gray')
plt.title('Adaptive Thresholded Image'), plt.xticks([]), plt.yticks([])
plt.show()
```