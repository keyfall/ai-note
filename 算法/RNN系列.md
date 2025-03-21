#### 循环神经网络-RNN
![[Pasted image 20250222101701.png]]
 原理： 
 &emsp;每个时间步的隐藏状态会根据输入和前一个时间步的隐藏状态进行更新，从而实现信息的传递和保存。这使得RNN可以捕捉到上下文关系，并对时序数据进行建模。 

结构：
- 输入和输出：输入可以是任意长度的序列数据，如文本、语音等。输出可以是每个时间步的预测结果，也可以是最后一个时间步的隐藏状态。
- 隐藏状态更新：隐藏状态会根据当前时间步的输入和前一个时间步的隐藏状态进行计算，并被传递给下一个时间步。隐藏状态的更新可以使用简单的线性变换和激活函数，也可以使用更复杂的门控机制。
- 参数共享：在每个时间步上，RNN使用相同的权重和偏置进行计算。 

优点：
 - 处理序列数据
 - 参数共享：减少了模型的复杂度，也使得模型对输入序列长度的变化更加鲁棒。 

缺点：
- [[梯度消失或爆炸]]：由于反向传播过程中需要对长时间步上的误差进行累乘，这可能导致梯度变得非常小（梯度消失），使得早期层的学习变得极其缓慢，或者相反，导致梯度异常大（梯度爆炸），使训练过程不稳定。
- 序列长度过长不易捕捉关系：这个问题部分被LSTM和GRU所缓解，它们通过引入门控机制更好地管理了信息流。
- 训练成本：训练时需要更多的计算资源和时间。这是因为这些模型通常包含更多的参数，并且需要较长的时间序列来进行有效的训练。
- 并行化：RNN是按顺序处理输入数据，因此很难实现并行化加速训练过程

应用：
- 自然语言处理：机器翻译，文本生成，情感分析
- 语音识别
- 时间序列预测：天气预测，金融预测，股票预测

#### 长短期记忆人工神经网络-LSTM
![[Pasted image 20250222224140.png]]![[Pasted image 20250222224149.png]]
原理： 
 &emsp;和rnn相似，只是多了一些设定条件，控制输入信息的保留和舍弃，使得网络更好的处理长期依赖关系

结构：
- 遗忘门：确定细胞状态中哪些信息应该被遗忘或丢弃。它也包含一个sigmoid激活函数，根据输入数据和前一个时间步的隐藏状态来输出一个介于0和1之间的值。![[Pasted image 20250222224715.png]]
- 输入门：控制着新输入信息对当前单元状态的影响程度。它包括一个sigmoid激活函数，用于生成一个介于0和1之间的值，表示每个单元状态中的哪些值应该被更新。首先，Sigmoid层决定了哪些部分的状态需要更新，这意味着只有那些被认为重要的状态才会得到更新的机会。然后，Tanh层提供了具体的数值，这些数值表示了可能的新信息。两者相乘的结果决定了最终哪些信息会被加入到细胞状态中。![[Pasted image 20250225063720.png]]![[Pasted image 20250222225044.png]]
- 输出门: 控制着当前时刻的隐藏状态以及下一个时刻的细胞状态如何影响最终的输出。它包含一个sigmoid激活函数来确定输出状态的哪些部分将被激活。![[Pasted image 20250222225049.png]]

除了RNN的优缺点外，LSTM只解决梯度消失问题，因为公式被微分后，使用了sigmoid，导数值为0-1


#### 门控循环神经网络-GRU
![[Pasted image 20250225094156.png]]

原理： 
 &emsp;和lstm类似，把3个门换成了2个门
结构：
- 更新门：控制前一时刻的状态信息被带入到当前状态中的程度，更新门的值越大说明前一时刻的状态信息带入越多，作用类似于LSTM中的遗忘门和输入门的结合，它决定了要忘记哪些旧的信息以及添加哪些新信息。
![[Pasted image 20250225094430.png]]
- 重置门：忽略之前的状态信息，重置门越小，前一状态的信息被写入的越少
![[Pasted image 20250225094313.png]]
- 候选隐状态：将重置门$r_t$与上一时间步的隐藏状态$h_{t-1}$进行计算，得到在时间步t的候选隐状态![\widetilde{H_{t}}](https://latex.csdn.net/eq?%5Cwidetilde%7BH_%7Bt%7D%7D)，在这里使用的是tanh非线性激活函数来确保候选隐状态中的值保持在区间（-1,1)中。$r_t$与$h_{t-1}$相乘可以减少以往状态的影响。每当重置门$r_t$中的项接近与1时，GRU就能恢复成一个普通的循环神经网络。对于重置门$r_t$中所有接近0的项，候选隐状态是以$X_{t}$作为输入的多层感知机的结果。因此，任何预先存在的隐状态都会被重置为默认值。![[Pasted image 20250225102422.png]]
- 隐藏状态：把上一个隐藏状态和新隐藏状态进行相加，每当更新门接近1时，模型就倾向只保留旧状态。 此时，来自$X_{t}$的信息基本上被忽略， 从而有效地跳过了依赖链条中的时间步t。 相反，当$Z_{t}$接近0时,新的隐状态就会接近候选隐状态。 ![[Pasted image 20250225103449.png]]

优缺点和lstm相似

#### LSTM和GRU区别
- STM对新产生的状态可以通过输出门(output gate)进行调节，而GRU对输出无任何调节。
- GRU的优点是这是个更加简单的模型，所以更容易创建一个更大的网络，而且它只有两个门，在计算性能上也运行得更快，然后它可以扩大模型的规模。
- LSTM更加强大和灵活，因为它有三个门而不是两个。