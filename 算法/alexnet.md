#### 结构

![[Pasted image 20250316230542.png]]
- 模块一和模块二:卷积-激活函数（ReLU）-lrn-降采样（池化）-标准化
- 模块三和模块四:卷积-激活函数（ReLU）
- 模块五和模块六:卷积-激活函数（ReLU）-降采样（池化）
- 模块七和模块八：全连接层-relu-drop
- 模块八：输出层和softmax进行分类

特点:
- lrn：图片的同一区域的不同通道处进行归一化，只是针对单一图片，已经被bn取代
- 重叠池化：
	- 池化时步长小于池化核长，会保存更多信息
	- 引入重叠区域，使得处理图像形变、提高模型鲁棒性的场景中效果显著
	- 模型从略微不同的视角看待输入数据，增加了模型内部的多样性，减少过拟合
- relu:
	- 因为x小于0时,梯度为0，所以加速训练
	- 因为x大于0时，梯度不会为0，所以缓解梯度消失