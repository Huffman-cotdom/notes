# Pre-Norm 和 Post-Norm

> 苏剑林博客：https://kexue.fm/archives/9009
>
> 知乎：https://zhuanlan.zhihu.com/p/474988236

Layer Normalization 的位置对于 Transformer 模型训练也非常重要的。

现在的 LLMs 不断加深模型，但是堆叠太多的层会因为梯度消失/梯度爆炸的原因导致模型难以训练。

传统的 LN 是在残差之后，做完 Add 之后再进行归一化，这种方式叫做 Post-Norm。

Pre-nrom 是把 LN 放在残差之前：

![Pre-Norm 和 Post-Norm](assents/截屏2023-04-23%2016.23.25.png)

## 理解

post-norm 在残差之后做归一化，对参数正则化的效果更强，进而模型的鲁棒性也会更好

pre-norm 相对于 post-norm，因为有一部分参数直接加在了后面，不需要对这部分参数进行正则化，正好可以防止模型的梯度爆炸或者梯度消失

@苏剑林

对比实验：Post Norm 的结构迁移性能更加好，也就是说在 Pre training 中，Pre Norm 和 Post Norm 都能做到大致相同的结果，但是 Post Norm 的 Finetune 效果明显更好。

完全相同的训练设置下 Pre Norm 的效果要优于 Post Norm，这只能显示出 Pre Norm 更容易训练，因为 Post Norm 要达到自己的最优效果，不能用跟 Pre Norm 一样的训练配置（比如 Pre Norm 可以不加 Warmup 但 Post Norm 通常要加）。

一个 L 层的 Pre Norm 模型，其实际等效层数不如 L 层的 Post Norm 模型，而层数少了导致效果变差了。

Pre_Norm 迭代：

$$
\begin{equation}\begin{aligned} 
\boldsymbol{x}_{t+1} =&\,\boldsymbol{x}_t + F_t(\text{Norm}(\boldsymbol{x}_t)) \\ 
=&\, \boldsymbol{x}_{t-1} + F_{t-1}(\text{Norm}(\boldsymbol{x}_{t-1})) +  F_t(\text{Norm}(\boldsymbol{x}_t)) \\ 
=&\, \cdots \\ 
=&\, \boldsymbol{x}_0 + F_0 (\text{Norm}(\boldsymbol{x}_0)) + \cdots + F_{t-1}(\text{Norm}(\boldsymbol{x}_{t-1})) +  F_t(\text{Norm}(\boldsymbol{x}_t)) 
\end{aligned}\end{equation}
$$

其中每一项都是同一量级的，那么有$\boldsymbol{x}_{t+1}=\mathscr{O}(t+1)$，也就是说当$t$较大时第$t+1$层跟第$t$层的差别，两者的相对差别是很小的，因此

$$\begin{equation}\begin{aligned}  &\,F_t(\text{Norm}(\boldsymbol{x}_t)) + F_{t+1}(\text{Norm}(\boldsymbol{x}_{t+1})) \\  \approx&\,F_t(\text{Norm}(\boldsymbol{x}_t)) + F_{t+1}(\text{Norm}(\boldsymbol{x}_t)) \\  =&\, (F_t\oplus F_{t+1})(\text{Norm}(\boldsymbol{x}_t))  \end{aligned}\end{equation}$$

这个意思是说，当 $t$ 比较大时，$x_t,x_{t+1}$ 相差较小，所以$F_{t+1}(\text{Norm}(\boldsymbol{x}_{t+1}))$与$F_{t+1}(\text{Norm}(\boldsymbol{x}_t))$很接近，**因此原本一个$t$层的模型与$t+1$层和，近似等效于一个更宽的$t$层模型**，所以在 Pre Norm 中多层叠加的结果更多是增加宽度而不是深度，层数越多，这个层就越“虚”。

说白了，Pre Norm 结构无形地增加了模型的宽度而降低了模型的深度，而我们知道深度通常比宽度更重要，所以是无形之中的降低深度导致最终效果变差了。而 Post Norm 刚刚相反，在[《浅谈 Transformer 的初始化、参数化与标准化》](https://kexue.fm/archives/8620)中就分析过，它每 Norm 一次就削弱一次恒等分支的权重，所以 Post Norm 反而是更突出残差分支的，因此 Post Norm 中的层数更加“足秤”，一旦训练好之后效果更优。

## 为什么 Layer Normalization 要加在 F 的前面，而不是 F 的后面呢？
![](assents/Pasted%20image%2020230512002333.png)
因为做完 Layer Normalization 之后的数据不能和平常的数据加在一起，如果这样做的话**残差中从上一层出来的信息会占很大比重**，这显然并不合理。