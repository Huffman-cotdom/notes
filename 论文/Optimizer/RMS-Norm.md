> RMS Norm 论文：[RMS-Norm](./assents/RMS-Norm.pdf)
> 
> GitHub 地址：https://github.com/bzhangGo/rmsnorm
> 
> ChatDoc 地址：https://chatdoc.com/chatdoc/#/chat/e1696ba4-9890-486e-9002-5251692b465b

# 摘要

LayerNorm 帮助稳定训练并提高模型收敛性，因为它能处理输入和权重矩阵的重新居中（re-centering）和重新缩放（re-scaling）（its capability in handling re-centering and re-scaling of both inputs and weight matrix.）

RMS Norm 相比传统 LayerNorm 计算更加简单高效，在几个任务的广泛实验中，使用不同的网络架构，证明 RMSNorm 实现与 LayerNorm 相当的性能，但可以在不同模型上降低运行时间 7％〜64％

论文还提出了部分 RMSNorm，或 pRMSNorm，其中 RMS 是从$p$％的和输入估计的，不会破坏上述属性

@苏建林 https://zhuanlan.zhihu.com/p/400925524
center 操作，类似于全连接层的 bias 项，储存到的是关于数据的一种先验分布信息，而把这种先验分布信息直接储存在模型中，反而可能会导致模型的迁移能力下降。所以 T5 不仅去掉了 Layer Normalization 的 center 操作，它把每一层的 bias 项也都去掉了。

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

# 介绍

LayerNorm 加速模型收敛，通过对一层中神经元动态进行平均值和方差统计来使深度神经网络的训练稳定。

此外，与 BatchNorm 相比，LayerNorm 的解耦能力使其在使用 RNN 处理可变长度序列时具有优越性。

LayerNorm 的引入会增加计算开销。对于更大模型来说这点非常重要。因此，由于每个训练步骤的计算成本增加，减少了网络的效率增益（以训练步骤数量表示的更快更稳定的训练）。
![训练步数](assents/Pasted%20image%2020230515234444.png)
LayerNorm 被广泛认为对稳定的贡献之一是其平移不变特性：当输入或权重矩阵被某些噪声偏移时，经过 LayerNorm 后的输入之和仍然保持不变。作者认为，这种均值归一化并不能减少隐藏状态或模型梯度的方差，并假设它对 LayerNorm 的成功没有太大影响。

# 背景

## LN 的作用

### 没有 LN 时，梯度不稳定，模型延迟收敛

没有 LayerNorm 时，给定一个输入向量 $x∈R^m$，一个前馈网络通过线性转换和非线性激活将其投影到一个输出向量 $y∈R^n$ 中，如下所示：
$$$a_i=\sum_{j=1}^m w_{i j} x_j, \quad y_i=f\left(a_i+b_i\right)$$
其中，$w_i$ 是第 $i$ 个输出神经元的权重向量，$b_i$ 是偏置标量，通常由 0 来初始化，$f(·)$ 是逐元素非线性函数。$a ∈ R^y$ 表示神经元的加权输入，也是归一化的目标。

这个网络可能会遇到内部协变量转移问题（*internal covariate shift*）。当前一层被更新时，层的输入分布会发生改变。这可能会对参数梯度的稳定性产生负面影响，延迟模型的收敛。

## 使用 LN

为了减少这种变化，LayerNorm 对输入进行求和后将其归一化，以固定它们的平均值和方差，如下所示：

$$$\bar{a}_i=\frac{a_i-\mu}{\sigma} g_i, \quad y_i=f\left(\bar{a}_i+b_i\right)$$
其中$\bar ai$是向量$\bar a∈R^n$中的第 i 个值，它作为第$i$层激活的归一化替代物。$g∈R^n$是用于重新调整标准化的总和输入的增益参数，在开始时设置为 1。$µ$ 和$σ^2$分别是从原始总和输入$a$估计的均值和方差。
$$\mu=\frac{1}{n} \sum_{i=1}^n a_i, \quad \sigma=\sqrt{\frac{1}{n} \sum_{i=1}^n\left(a_i-\mu\right)^2}$$
因此，LN 强制神经元的范数与输入和权重矩阵解耦。

LN 成功的一个著名解释是它的重新中心化（re-centering）和重新缩放（re-scaling）不变性特性。前者使得模型对输入和权重的位移噪声不敏感，后者在输入和权重随机缩放时保持输出表示不变。在本文中，我们假设重新缩放不变性是 LN 成功的原因，而不是重新中心化不变性。

# RMS Norm

RMSNorm，它仅关注重新缩放不变性，并根据均方根（RMS）统计量简单地正则化总输入。

$$\bar{a}_i=\frac{a_i}{\operatorname{RMS}(\mathbf{a})} g_i, \quad where \operatorname{RMS}(\mathbf{a})=\sqrt{\frac{1}{n} \sum_{i=1}^n a_i^2}$$
RMSNorm 简化了 LN，牺牲了平均标准化带来的不变性，完全去除了 LN 中的均值统计量。当输入的总和的均值为零时，RMSNorm 与 LayerNorm 完全相同。

RMS 度量了输入的平方均值，在 RMSNorm 中，它将总和输入强制转换成一个标准化的单位球，这样输出的分布不受输入和权重分布的缩放影响，有利于层激活的稳定性。

## Invariance Analysis 不变性分析

>进行缩放输入和权重矩阵对层输出影响

与 LayerNorm 的主要区别在于，RMSNorm 没有重新居中（re-scaling），因此在变量移位方面不显示类似的线性特性。它不对所有重新居中（re-scaling）操作不变。

## Gradient Analysis 梯度分析

在一般情况下，RMSNorm 增强的神经网络是通过标准的随机梯度下降方法进行训练的，模型梯度的鲁棒性对参数的更新和模型的收敛非常关键。（归一化方法的成功并非来自于对层输入的稳定性增强，而是由于优化平滑度增加）

RMS Norm 梯度$∂L /∂W$对输入缩放不变，但保持与权重矩阵缩放的负相关性。减少梯度$∂L /∂W$对输入缩放的敏感性可以确保平滑性并提高学习的稳定性。另一方面，负相关作为隐式学习率适配器，并动态控制梯度的范数，避免了大范数的权重矩阵，并提高了模型的收敛性。

# pRMSNorm

RMSNorm 的重新缩放不变性属性归因于 RMS 的线性性质。考虑到一个层中的神经元通常具有独立等分布结构，我们认为可以在子集神经元上估计 RMS 而不是全部神经元。我们提出了部分 RMSNorm（pRMSNorm）。给定未归一化的输入$a$，pRMSNorm 从$a$的前$p$％元素中推断出 RMS 统计量：$\overline{\operatorname{RMS}}(\mathbf{a})=\sqrt{\frac{1}{k} \sum_{i=1}^k a_i^2}$，其中 $k=\lceil n \cdot p\rceil$ 表示用于 RMS 估计的元素数。RMS 仍然保持线性性质，这表明 pRMSNorm 具有与 RMSNorm 相同的不变性属性。

RMS 是一种有偏估计，通常不准确。虽然理论上 pRMSNorm 近似于 RMSNorm，但我们观察到在小 m 的情况下梯度不稳定性。然而，在实践中，使用部分比率为 6.25％的 pRMSNorm 模型可以成功地达到令人满意的收敛性。

# 实验

RMS Norm 保持模型效果的同时，模型训练时间减少很多

![实验结果](assents/Pasted%20image%2020230516005721.png)![](assents/Pasted%20image%2020230516005930.png)
![](assents/Pasted%20image%2020230516010146.png)
![](assents/Pasted%20image%2020230516010215.png)![](assents/Pasted%20image%2020230516010234.png)