> 旋转式位置编码（Rotary Position Embedding，RoPE）
> 博客：https://kexue.fm/archives/8265
> 知乎：https://zhuanlan.zhihu.com/p/415020704

假设函数 $f(⋅,f)$ 表示添加绝对位置信息的函数（即 RoPE）， $g(q,k,m-n)$ 表示位置敏感的内积运算，那么 RoPE 满足：
$$f(\boldsymbol{q}, m)^{\top} f(\boldsymbol{k}, n)=g(\boldsymbol{q}, \boldsymbol{k}, m-n)$$
假设 $f(⋅,f)$ 的形式为 $f(x,t)=R_tx$ ，那么有
$$\begin{aligned} g(\boldsymbol{q}, \boldsymbol{k}, m-n) & =f(\boldsymbol{q}, m)^{\top} f(\boldsymbol{k}, n) \\ & =\left(R_m \boldsymbol{q}\right)^{\top}\left(R_n \boldsymbol{k}\right) \\ & =\boldsymbol{q}^{\top} R_m^{\top} R_n \boldsymbol{k}\end{aligned}$$
考虑到公式一，那么有$f(\boldsymbol{q}, m)^{\top} f(\boldsymbol{k}, n)=f(\boldsymbol{q}, m+k)^{\top} f(\boldsymbol{k}, n+k)$，即有：
$${R}^{\top}_mR_n={R}^{\top}_{m+k}R_{n+k}=W_{m-n}$$
在二维情况下：
$$\begin{equation} \boldsymbol{f}(\boldsymbol{q}, m) =\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix} \begin{pmatrix}q_0 \\ q_1\end{pmatrix}\end{equation}$$
那么有：
$$\begin{align} R_m^\top R_n &=\left(\begin{array}{cc} \cos \alpha_m & -\sin \alpha_m\\ \sin \alpha_m & \cos \alpha_m\\ \end{array}\right)^\top \left(\begin{array}{cc} \cos \alpha_n & -\sin \alpha_n\\ \sin \alpha_n & \cos \alpha_n\\ \end{array}\right)\\ &=\left(\begin{array}{cc} \cos \alpha_m & \sin \alpha_m\\ -\sin \alpha_m & \cos \alpha_m\\ \end{array}\right)\left(\begin{array}{cc} \cos \alpha_n & -\sin \alpha_n\\ \sin \alpha_n & \cos \alpha_n\\ \end{array}\right)\\ &=\left(\begin{array}{cc} \cos \alpha_m\cos \alpha_n + \sin \alpha_m \sin \alpha_n & - \cos \alpha_m \sin \alpha_n + \sin \alpha_m\cos \alpha_n\\ - \sin \alpha_m\cos \alpha_n+\cos \alpha_m \sin \alpha_n & \sin \alpha_m \sin \alpha_n+\cos \alpha_m\cos \alpha_n \\ \end{array}\right)\\ \end{align}$$
根据三角恒等变换公式：
$$\begin{align} \sin (\alpha+\beta)&=\sin \alpha \cos\beta + \cos \alpha \sin\beta\\ \cos (\alpha+\beta)&=\cos \alpha \cos\beta - \sin \alpha \sin\beta \end{align}$$
可以化简得到
$$R_m^\top R_n=\left(\begin{array}{cc} \cos (\alpha_n-\alpha_m) & -\sin (\alpha_n-\alpha_m)\\ \sin (\alpha_n-\alpha_m) & \cos (\alpha_n-\alpha_m)\\ \end{array}\right)=R_{n-m}$$
于是任意偶数维的 RoPE，我们都可以表示为二维情形的拼接，即
$$\begin{equation}\scriptsize{\underbrace{\begin{pmatrix} \cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ \sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\ 0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ 0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \\ \end{pmatrix}}_{\boldsymbol{\mathcal{R}}_m} \begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1}\end{pmatrix}}\end{equation}$$
也就是说，给位置为$m$的向量$q$乘上矩阵$R_m$、位置为$n$的向量$k$乘上矩阵$R_n$，用变换后的$Q,K$序列做 Attention，那么 Attention 就自动包含相对位置信息了，因为成立恒等式：
$$\begin{equation}(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) =  \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n \boldsymbol{k} = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}\end{equation}$$
值得指出的是，$R_m$是一个正交矩阵，它不会改变向量的模长，因此通常来说它不会改变原模型的稳定性。

由于$R_m$的稀疏性，所以直接用矩阵乘法来实现会很浪费算力，推荐通过下述方式来实现 RoPE：

$$\begin{equation}\begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} 
\end{pmatrix}\otimes\begin{pmatrix}\cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1} 
\end{pmatrix} + \begin{pmatrix}-q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2} 
\end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} 
\end{pmatrix}\end{equation}$$
  
其中$⊗$是逐位对应相乘，即 Numpy、Tensorflow 等计算框架中的$∗$运算。从这个实现也可以看到，RoPE 可以视为是乘性位置编码的变体。

可以看到，RoPE 形式上和 Sinusoidal 位置编码有点相似，只不过 Sinusoidal 位置编码是加性的，而 RoPE 可以视为乘性的。在$θ_i$的选择上，我们同样沿用了 Sinusoidal 位置编码的方案，即$\theta_i = 10000^{-2i/d}$，它可以带来一定的远程衰减性。

![RoPE 的远程衰减性](assents/Pasted%20image%2020230524002351.png)

从图中我们可以可以看到随着相对距离的变大，内积结果有衰减趋势的出现。因此，选择$\theta_i = 10000^{-2i/d}$，确实能带来一定的远程衰减性。当然，同上一篇文章说的一样，能带来远程衰减性的不止这个选择，几乎任意的光滑单调函数都可以，这里只是沿用了已有的选择而已。笔者还试过以$\theta_i = 10000^{-2i/d}$为初始化，将$θ_i$视为可训练参数，然后训练一段时间后发现$θ_i$并没有显著更新，因此干脆就直接固定$\theta_i = 10000^{-2i/d}$了。

LLaMA 代码：

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

看懂这一部分代码，最关键的是弄清楚其中的变量 freqs_cis 所指是什么东西。

为了搞懂这部分，我们需要先了解几个 torch 中不太常用的方法：

（1）torch.view_as_complex

把一个 tensor 转为复数形式，要求这个 tensor 的最后一个维度形状为 2。

```python
torch.view_as_complex(torch.Tensor([[1, 2], [3, 4], [5, 6]]))
# tensor([1.+2.j, 3.+4.j, 5.+6.j])

x=torch.randn(4, 2)
tensor([[ 1.2013,  1.1121],
        [ 1.0219,  1.1691],
        [-0.6020, -0.3902],
        [ 1.8236, -1.2965]])
x = torch.view_as_complex(x)
tensor([ 1.2013+1.1121j,  1.0219+1.1691j, -0.6020-0.3902j,  1.8236-1.2965j])
```

（2）torch.view_as_real

把复数 tensor 变回实数，可以看做是是刚才操作的逆变换。

```python
torch.view_as_real(torch.view_as_complex(torch.Tensor([[1, 2], [3, 4], [5, 6]])))
# tensor([[1., 2.],
#         [3., 4.],
#         [5., 6.]])

x = torch.view_as_real(x)
tensor([[ 0.2282, -0.0675],
        [-1.1504, -0.2244],
        [ 0.0075,  0.7334],
        [-0.4255,  1.0590]])
```

（3）torch.outer

一个向量的转置乘以另一个向量：torch.outer(a, b) = a^T * b

```python
a = torch.arange(1, 5)
tensor([1, 2, 3, 4])
b = torch.arange(1, 4)
tensor([1, 2, 3])
torch.outer(a, b)
# tensor([[ 1,  2,  3],
#         [ 2,  4,  6],
#         [ 3,  6,  9],
#         [ 4,  8, 12]])

outer_product = a.unsqueeze(1) * b.unsqueeze(0)
a.unsqueeze(1), b.unsqueeze(0), outer_product
tensor([[1], [2], [3], [4]])
tensor([[1, 2, 3]])
tensor([[ 1, 2, 3], [ 2, 4, 6], [ 3, 6, 9], [ 4, 8, 12]])
```

（4）torch.polar

构造一个复数张量，其元素是笛卡尔坐标对应的极坐标绝对值 abs 和 angle angle。

$$out=abs⋅cos(angle)+abs⋅sin(angle)⋅j$$

```python
torch.polar(torch.tensor([1], dtype=torch.float64), torch.tensor([np.pi / 2], dtype=torch.float64))
# tensor([6.1232e-17+1.j], dtype=torch.complex128)

abs = torch.tensor([1, 2], dtype=torch.float64)
tensor([1., 2.], dtype=torch.float64)

angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
tensor([1.5708, 3.9270], dtype=torch.float64)

torch.polar(abs, angle)
tensor([ 6.1232e-17+1.0000j, -1.4142e+00-1.4142j], dtype=torch.complex128)
```

首先是 precompute_freqs_cis 函数：

freqs_cis 所指的就是需要计算出来的 $m_θ$ 也就是跟绝对位置相关的旋转的角度，在极坐标下对应的复数 tensor。

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
```

假设 batch_size 为 2，seq_len 固定为 512，attention_head 的数量为 12，每个 attention_head 的维度为 64，那么，对于输入到 multi-head attn 中的输入 $x_q$ 的尺寸就是 (2, 512, 12, 64)。

而函数 precompute_freqs_cis 就是提前将这些旋转角度对应的 tensor 给创建出来，并可以重复利用。因为确定了序列的最大长度，所以这个 tensor 是固定死的。根据后续的数据流我们可以发现，在调用该函数时，传入的两个参数分别是 attention_head 的维度，以及最大长度的两倍，具象地，也就是 64 和 1024。

```python
freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
```

$$\frac{1}{\theta^{i/dim}}$$
首先 torch.arange 创建了一个 tensor[0,2,4,...,60,62]，然后统一除以 dim=64，把它变成分数，然后整体作为基础角度的指数，它的 shape 是 dim/2=(32)

```python
t = torch.arange(end, device=freqs.device)
```

t 比较容易理解，也就是**绝对位置信息**，它的 shape 是`(1024)`。

```python
freqs = torch.outer(t, freqs).float()
```

于是根据 torch.outer 运算，我们得到了一个 shape 为 (1024, 32) 的 tensor。其意义也就是将每一个绝对位置，分配到对应的角度，相乘。直观理解一下，就是每一个绝对位置上，都有 32 个角度。为什么是这样的呢，回顾计算的公式，对于旋转矩阵，每两个元素为一组，它们乘以的角度是同一个 $\theta$ ，所以这个 (1024, 32)，在后续的过程中，就可以 reshape 成 (512, 64)，并且在 64 的那个维度上，每两个是相同的。

```python
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
```

这一步就是在**生成我们需要的位置信息**，直观理解一下，像是在复平面内，以原点为中心，转了 1024 组，每一组 64 个的单位向量，它的 shape 是 (1024, 64)。

然后是 reshape_for_broadcast 函数是把 freqs_cis 变成和输入的 tensor 相同的形状，结合下边的另一个方法一起介绍。

最后来看 apply_rotary_emb 函数

```python
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

其实就是把位置信息添加到原有的编码结果上，在 multi-head attention 阶段调用：

```python
xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
```

上文中，我们假设了输入$x​_q$的尺寸就是 (2, 512, 12, 64)，那么这一句操作的 reshape，就是把它变成 (2, 512, 12, -1, 2)，也就是 (2, 512, 12, 32, 2)。$x_k$​同理。紧接着把它变成复数形式，也就是变成了 (2, 512, 12, 32) 的形状。

然后进入到 reshape_for_broadcast 方法：

```python
shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
return freqs_cis.view(*shape)
```

这个方法的作用是为了把 freqs_cis 变成和输入的 tensor 相同的形状。需要注意的是，这里的 freqs_cis 并不是 precompute_freqs_cis 生成的形状为 (1024, 64) 的那个 tensor，而是根据输入的绝对位置，在 (1024, 64) 的 tensor 中，**截取了长度为当前 seq_len 的一部分**，代码在 Transformer 类的 forward 方法中：

```python
freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
```

也就是说，假如当前输入的序列长度是 512，那么截取出来的这个新的 freqs_cis，形状就是 (512, 64)，reshape 之后，形状就变成了 (1, 512, 1, 32)，也就是在每一个位置上，都对应有 32 个角度，根据刚刚 torch.polar 的介绍，当我们固定绝对值（也就是向量的模长）时，角度就可以在笛卡尔坐标系下唯一确定一个复数，这样一来也就是 32 个复数，即 64 个特征维度，所以就可以对应的将它融合到每个 attention head 的 64 个特征中去了。

reshape 之后，就是将位置信息融入 query 和 key 中：

```python
xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
```

这一步将二者相乘得到的复数 tensor，重新转换为实数形式，得到的 shape 为 (2, 512, 12, 32, 2)，然后再 flatten 成 (2, 512, 12, 64)，这样一来，就变回了和最开始$x_q​$相同的形状，也就完成了将位置信息融入到$x_q$的这一操作。$x_k​$ 同理。


