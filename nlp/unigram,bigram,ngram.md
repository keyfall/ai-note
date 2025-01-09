语言模型根据复杂程度划分
unigram表示词和词之间是相互独立的，和其他词没有关系，因此不会牵扯上下文
![[Pasted image 20240528093517.png]]

bigram是1st order马尔科夫假设，考虑前面的单词的概率
![[Pasted image 20240528093703.png]]

n-gram就是多马尔科夫假设
![[Pasted image 20240528093849.png]]


实操
unigram
![[Pasted image 20240528094308.png]]

bigram
![[Pasted image 20240528094336.png]]

可以发现如果有1个概率为0，那么就可能整个句子概率为0，所以可以对每个概率进行平滑操作(对所有计数值加1，保证不会为0)

平滑操作：
 - add one smoothing(+1平滑)
 - add k smoothing(+k平滑)
 - interpolation
 - 

+1平滑：
对有所计数值+1，分母+v，分母加v是为了最后的最大值不会超过1
![[Pasted image 20240528104634.png]]
![[Pasted image 20240528104652.png]]

+k平滑
和+1平滑类似，不过不是+1，是加k，分母加的是kv
目的是找到似的perplexity最小的k值
两种方法：
 - k值从1开始试
 - 使用一个基本的语言模型，然后k作为位置参数，获取perlexity最小时k的值
![[Pasted image 20240528111706.png]]

interpolation
p(a|b),b词出现的情况下，a词出现的概率
这个值为0，可能因为b词出现后a词没出现
也可能因为a词根本没出现过，b词没出现(这种情况unigram可以找到不同)
因此把unigram,bigram,ngram进行加权平均，ngram的权重最大，unigram最小
![[Pasted image 20240528114323.png]]

good turning smoothing
会在历史概率基础上考虑当前未出现的内容的情况
N1的意思是出现一次物种数量的概率
![[Pasted image 20240531165848.png]]
![[Pasted image 20240531170955.png]]
缺点是到“出现次数”很多的时候可能会出现数量为0
因为计算概率时会使用“出现次数+1”的数量值
所以数量值可能为0，因此算的当前概率可能为0
可以使用次数和数量计算一个回归公式，预测出后一个可能出现数量为0的次数
再计算当前次数的概率
![[Pasted image 20240531171627.png]]

评估语言模型指标：perplexity(困惑度)
找到每个概率值，然后进行log平均(x),其实x值越大，perplexity越小，说面该模型越好
![[Pasted image 20240528103035.png]]

