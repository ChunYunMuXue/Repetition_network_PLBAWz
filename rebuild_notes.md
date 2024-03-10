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
- 但是好像训练量少 (epoch < 10) 的时候不如AlexNet的准确度提升高，但提升比较稳定
- 但是他网络层数多，潜力大，我猜测可以在训练次数多的时候高于AlexNe (怕我的电脑顶不住就不试验更大的epoch了