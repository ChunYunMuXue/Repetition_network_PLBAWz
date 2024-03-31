$M = \frac{N + 2  \times padding - kernel_size}{stride} + 1$

## Classfition

### AlexNet 

- 把网络层不同模块写在一个模块里打包
- 把模型，训练，测试分开使用
- 初始化模型参数
  
### VGG

- 堆叠两个 $3 \times 3$ 卷积核代替 $5 \times 5$ 的卷积核，堆叠三个 $3 \times 3$ 卷积核代替 $7 \times 7$ 的，具有相同感受野
- 感受野：决定某一层输出中单个元素的输入层区域大小
- $F_i = (F_{i + 1} - 1) \times Stride + Kszie$,第一项是去除相同的部分
- 格式规范化不同网络参数，使用参数导入网络
- * 把 list 拆分成单个元素
- 看起来VGG的计算量比AlexNet大不少，我的电脑跑的时候声音有点大（
  - upd：传上去github的时候发现他的参数文件高达500m
- 但是好像训练量少 (epoch < 10) 的时候不如AlexNet的准确度提升高，但提升比较稳定
- 但是他网络层数多，潜力大，我猜测可以在训练次数多的时候高于AlexNe (怕我的电脑顶不住就不试验更大的epoch了


### GoogLeNet

- 引入了Inception结构  
- 使用 1 x 1 的卷积核进行降维 以及 映射处理
- 添加两个辅助分类器帮助训练
- 丢弃了全连接层，用平均池化层，减少了参数量
- 写到现在第一个有宽度的网络
- 然后大部分的重复的比较复杂的部件可以另写成一个class，比如可以单独实现一个Inception的类
- nn.MaxPool2d(3,stride = 2,ceil_mode = True),当ceil_mode = true时，将保存不足为kernel_size大小的数据保存，自动补足NAN至kernel_size大小；
- GoogLeNet确实好复杂,但是参数量应该比VGG小，这归功于1x1降维功能，跑起来也是这样，风扇声小了很多（
- 然后正确率提升比较迅速，比 VGG 好，但是上下波动也比较大(猜测是步骤较多，训练还不够，前期容易波动)，最大正确率为0.794，比相同训练量的VGG高
- 训练epoch为30,后续尝试一下更多的层数，看起来他训练的需要计算量比VGG小很多

### ResNet ResNext

- 超深的网络结构
- 提出了 residual 结构
- 使用 Batch Normalization 加速训练
- 层数过多导致梯度消失 / 梯度爆炸，退化问题
- 虚线残差结构 右侧加了一个 1x1 kernel 保证大小一样
- batch normalization
- 迁移学习
- ResNext 有了block 有了类似于GoogLeNet的宽度上的改变
- 导入原本的预训练数据迁移学习
- 由于原本就有预训练数据支撑，所以跑的很厉害，正确率比前面的所有的都高很多 同时参数也比前面的少
- 学会了使用os相关技术完成批量预测

### Mobile Net
- 专注于移动端 / 嵌入式设备的轻量CNN网络，相比传统卷积网络，在准确率小幅下降的前提下大量减少了模型参数和运算量，所以原来不一定要卷准确度，让模型变得更通用也是一个方向
- v1 
  - 增加超参数 $\alpha$ $\beta$
  - Depthwise Convolution:没有层和层之间的关系了 DW + PW
  - ![alt text](D:\article and  study\study\SELF_STUDYING\rebuild_network\rebuild_notes.assets\image.png)]
  - 其实有点类似于拆维的思想
- v2
  - Inverted Residuals倒残差结构
    - 1x1 卷积升维
    - 3x3 卷积
    - 1x1 卷积降维
    - 使用了ReLU6()函数
    - 倒残差结构
  - Linear Bottlenecks
    - ReLU激活函数对低维的信息造成大量损失
    - 最后那个地方用线性的替换掉了原本的ReLu
- 发现我的vscode的python居然没办法直接点进torchvision,发现要下pylance插件
- 学会了怎么直接把官方的pth拿过来用，而且最后一层的参数不导入，而是自己训练
- 第一个epoch训练速度有点慢 然后正确度大概在83左右，确实不如上面ResNet,但是他确实跑的特别块

- v3

