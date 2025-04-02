- 原理：YOLO将输入图像分成S×S的网格。每个网格按照设定的长宽比(设定的长宽，在YOLO中是采用聚类的方法产生的，而在YOLOV5中具体为采用k-means聚类+遗传算法计算得到)，生成一系列锚框，根据objectness标签和NMS进行筛选，得到最终锚框，再根据训练数据，进行微调得到预测框
![[Pasted image 20250328185258.png]]
![[Pasted image 20250328185226.png]]
![[Pasted image 20250328185331.png]]
锚框的初始长宽（pw，ph）
yolo训练就是在预测tx，ty，th，tw的值，由于知道真实框的bx，by，bh，bw，训练过程中，通过与实际值的比较，调校tx，ty，th，tw的值
锚框选择的三点：
	- objectness标签：锚框是否包含物体，通过锚框和实际框的交并比，选出n个锚框，设置为1，其他设置为0，并且超过交并比阈值的也设置-1
	- location标签：对应的预测框中心位置(在当前网格的长度比上当前网格的长度)和长宽(和图片长宽的比值)，
	- label:具体类别
NMS基本原理：
	- 根据置信度分数对所有候选边界框进行排序。
	- 选择置信度分数最高的边界框作为最终预测框。
	- 计算其他边界框与当前预测框的交并比（IoU），如果 IoU 超过某个阈值（例如 0.5），则将这些边界框剔除。
	- 重复上述过程，直到所有边界框都被处理完毕。
通过
- 结构：![[Pasted image 20250328195828.png]]
- 结果
	- 边界框参数：
		- x,y:边界框中心点坐标
		- w,h:边界框宽度和高度
	- 置信度：物体的可信度，Confidence=Pr(Object)⋅IoU，Pr(Object)有物体为1
	- 物体类别


### YOLOv1
- **发布日期**：2016年6月
- **作者**：Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
- **论文**："[You Only Look Once: Unified, Real-Time Object Detection](https://link.zhihu.com/?target=https%3A//www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)"
- **主要优化点**：
- 将目标检测任务转化为单次前向传播问题，显著提升检测速度
- 能够以45 FPS的速度处理图像，有一个更快的版本可以达到155 FPS
- 限制：在小物体检测上的精度较差，且定位误差较高

### YOLOv2 (YOLO9000)
- **发布日期**：2017年12月
- **作者**：Joseph Redmon, Ali Farhadi
- **论文**："[YOLO9000: Better, Faster, Stronger](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1612.08242)"
- **主要优化点**：
- 能够检测9000种类别物体
- 多尺度图片训练增强模型鲁棒性
- 使用 K-means 聚类算法根据训练集中的真实框来确定最佳的先验框尺寸，而不是手动设定。
- 引入先验框改进对小物体的检测能力
- 在每个卷积层后添加了批量归一化（Batch Normalization），这有助于稳定训练过程、加速收敛
- 采用Darknet-19 网络，因为更少参数，提高效率，[[passthrough layer]]增强模型对小目标的检测能力

### YOLOv3
- **发布日期**：2018年4月
- **作者**：Joseph Redmon, Ali Farhadi
- **论文**："[YOLOv3: An Incremental Improvement](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1804.02767)"
- **主要优化点**：
- 引入Darknet-53作为主干网络，结合残差网络提高检测精度
- 多尺度预测改善对小物体的检测
- 取消软分类器，使用独立的二元分类器提高性能

### YOLOv4
- **发布日期**：2020年4月
- **作者**：Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao
- **论文**："[YOLOv4: Optimal Speed and Accuracy of Object Detection](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2004.10934)"
- **主要优化点**：
- 提出Bag of Freebies和Bag of Specials优化策略，提高模型精度
- CSPDarknet53更高效的主干网络，提升网络推理速度和精度
- 引入CIoU损失函数提高边界框回归性能

### YOLOv5
- **发布日期**：2020年6月
- **作者**：Glenn Jocher
- 无论文发表，开源地址：[https://github.com/ultralytics/yolov5](https://link.zhihu.com/?target=https%3A//github.com/ultralytics/yolov5)
- **主要优化点**：
- YOLOv5转向Pytorch框架，便于开发者使用和扩展
- 自适应的anchor box学习机制提高检测效率
- 提供多种尺寸的预训练模型满足不同场景需求

### YOLOv6
- **发布日期**：2022年6月
- **作者**：美团技术团队
- **论文：** “[YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2209.02976)”
- **主要优化点**：
- 针对行业应用优化，尤其注重推理速度
- 引入EfficientRep带来更高效的网络架构
- 优化模型部署性能，适合工业环境中的大规模应用

### YOLOv7
- **发布日期**：2022年7月
- **作者**：Wong Kin-Yiu, Alexey Bochkovskiy, Chien-Yao Wang
- **论文**："[YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"](https://link.zhihu.com/?target=https%3A//openaccess.thecvf.com/content/CVPR2023/papers/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.pdf)
- **主要优化点**：
- 在COCO数据集上达到新的速度与精度平衡
- 跨尺度特征融合提高对不同尺度物体的检测能力
- 改进训练过程中的标签分配方式提高训练效率

### YOLOv8
- **发布日期**：2023年1月
- **作者**：Ultralytics团队
- 无论文发表，开源地址：[[https://github.com/ultralytics/ultralytics](https://link.zhihu.com/?target=https%3A//github.com/ultralytics/ultralytics)]
- **主要优化点**：
- 提供可定制的模块化设计方便用户根据需求进行扩展
- 内置多种训练和超参数优化策略简化模型调优过程
- 集成检测、分割和跟踪功能

### YOLOv9
- **发布日期**：2024年2月
- **作者/贡献者**：WongKinYiu等
- **论文：**[YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2402.13616)
- **主要优化点**：
- 可编程梯度信息(PGI)+广义高效层聚合网络(GELAN)。
- 与YOLOv8相比，其出色的设计使深度模型的参数数量减少了49%，计算量减少了43%，但在MS COCO数据集上仍有0.6%的AP改进。

### YOLOv10
- **发布日期**：2024年5月
- **作者**：清华大学
- **论文**：[YOLOv10: Real-Time End-to-End Object Detection](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2405.14458)
- **主要优化点**：
- 实时端到端的对象检测，主要在速度和性能方面的提升

### YOLOv11
- **发布日期**：2024年9月
- **作者**：Ultralytics团队
- 无论文发表，开源地址：[https://github.com/ultralytics/ultralytics](https://link.zhihu.com/?target=https%3A//github.com/ultralytics/ultralytics)
- **主要优化点**：
- YOLOv11继承自YOLOv8，在YOLOv8基础上进行了改进，使同等精度下参数量降低20%，在速度和准确性方面具有无与伦比的性能。
- 其流线型设计使其适用于各种应用，并可轻松适应从边缘设备到云 API 等不同硬件平台。
- 使其成为各种物体检测与跟踪、实例分割、图像分类和姿态估计任务的绝佳选择。