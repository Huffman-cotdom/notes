# LLaMA

> 论文：[LLaMA](assents/LLaMA.pdf)
> GitHub 地址：https://github.com/facebookresearch/llama

## 简介

- 大模型

大规模语言模型（LLM）使用大量文本数据进行训练，展示了根据文本指导或少量示例执行新任务的能力。

当将模型扩展到足够大的规模时，这些 few-shot 特性首次出现。

- LLAMA 初衷 - 更小的模型、更多的数据、更好的性能

这些工作都是基于这样的假设：更多的参数会带来更好的性能。然而 [Training Compute-Optimal Large Language Models](Training Compute-Optimal Large Language Models.md) 最近的工作发现，在给定的计算预算下，最好的性能不是由最大的模型实现的，而是由在更多数据上训练的较小的模型实现的。

给定一个目标性能水平，**首选的模型不是训练速度最快的，而是推理速度最快的**，尽管训练一个大的模型以达到一定的性能水平可能会更简单，但训练时间较长的小模型最终会在推理中更快。

[Training Compute-Optimal Large Language Models](Training%20Compute-Optimal%20Large%20Language%20Models.md) 曾建议在 200B 的 token 上训练一个 10B 的模型，但作者发现 7B 的模型的性能甚至在 1T 的 token 之后还能继续提高。

本文的重点是训练一系列语言模型，在各种推理预算下实现最佳性能，通过训练更多的标记来实现。产生的模型称为 LLaMA，参数范围从 7B 到 65B。

- 模型性能

**尽管 LLaMA-13B 比 GPT-3 小 10 倍，但在大多数基准测试中都超过了 GPT-3**；65B 参数模型也可以与最好的大型语言模型（如 Chinchilla 或 PaLM-540B）竞争。

- 训练数据开源

与 Chinchilla、PaLM 或 GPT-3 不同的是，该工作只使用公开可用的数据，这使得工作符合开源原则，而大多数现有模型所依赖的数据要么没有公开可用，要么没有记录（例如 "书籍 -2TB "或 "社交媒体对话"）。

## 预训练数据处理

### 开源数据


![image-20230422232811593](assents/image-20230422232811593.png)

1. **英语 CommonCrawl**，占比 67%

由于 CommonCraw（网页爬虫数据：包含原始网页数据、元数据提取和文本提取）数据较为杂乱，该工作采用 CCNet pipleline 的方式（Wenzek 等人，2020）预处理了从 2017 年到 2020 年的 CommonCrawl 网页。

具体的工作：

- 首先，在行的层面上对数据进行了去重，用 fastText 线性分类器进行语言识别，以去除非英语页面，并用 n-gram 语言模型过滤低质量内容。

- 其次，训练了一个线性模型来对维基百科中用作参考的页面与随机抽样的页面进行分类，并丢弃了未被归类为参考的页面。

2. **C4** ，占比 15%

在探索性实验中观察到，使用不同的预处理 Com-monCrawl 数据集可以提高性能。

因此，该工作将公开的 C4 数据集（Raffel 等人，2020）也纳入我们的数据。

C4 的预处理也包含重复数据删除和语言识别步骤，**其与 CCNet 的主要区别在于质量过滤，主要依赖于一些启发式方法，比如网页中是否存在标点符号、单词和句子的数量等。**

3. **GitHub**，占比 4.5%

在代码方面，该工作使用了谷歌 BigQuery 上的 GitHub 公共数据集，并只保留在 Apache、BSD 和 MIT 许可下发布的项目。

此外，为了提高数据质量，还用**基于行长或字母数字字符比例**的启发式方法过滤了**低质量的文件**，并用**规范的表达式删除了如标题在内的模板化内容**。

最后在文件层面上对结果数据集进行去重，并进行精确匹配。

4. **维基百科**，占比 4.5%

该工作添加了 2022 年 6 月至 8 月期间的维基百科转储数据，涵盖 20 种语言，这些语言使用拉丁字母或西里尔字母，具体是：BG、CA、CS、DA、DE、EN、ES、FR、HR、HU、IT、NL、PL、UP、RO、RU、SL、SR、SV、UK。

此外，该工作对数据进行处理，以**删除超链接、评论和其他格式化的模板**。

5. **GutenbergProject 和 Books3**，占比 4.5%

书籍也是重要的语料来源，该工作的训练数据集包括两个书籍语料库：古腾堡计划（GutenbergProject）和 ThePile（Gao 等人，2020）的 Books3 部分，后者是一个可用于训练大型语言模型的公开数据集。

**在数据处理上，该工作在书的层面上进行了去重处理，删除了内容重叠度超过 90% 的书。**

6. **ArXiv**，占比 2.5%

科研文献对于提升专业性也有重要作用，该工作对 arXiv 的 Latex 文件进行处理，将科学数据添加到预训练数据集中。

**按照 Lewkowycz 等人（2022 年）的做法，该工作删除了第一节之前的所有内容以及书目。**

**此外，还删除了.tex 文件中的评论，以及用户写的内联扩展定义和宏，以增加论文之间的一致性。**

7. **Stack Exchange**，占比 2%

QA 数据对于提升垂直的专业问题也有帮助。

该工作还使用了 Stack Exchange 的开放数据，Stack Exchange 是一个高质量的问题和答案的网站，涵盖了从计算机科学到化学的不同领域。

具体的，该工作保留了 28 个最大的网站的数据，从文本中去除 HTML 标签，并按分数（从高到低）对答案进行排序。

### Tokenizer

使用 bytepair coding（BPE）算法（Sennrich 等人，2015）对数据进行标记化，使用 SentencePiece（Kudo 和 Richardson，2018）的实现。

值得注意的是，**对于数字，将它们拆分成单独的数字，对于未知的 UTF-8 字符，则使用字节来进行分解**。

![超参数](assents/截屏2023-04-22%2023.50.41.png)

另外，**在数据采样方面，对于大多数训练数据，每个 token 在训练过程中只采样一次，但维基百科和图书领域除外，对这些领域进行了大约两个 epochs。**

## 模型

### 模型结构

![超参数](assents/Pasted%20image%2020230516152617.png)

> 基于 Transformer Decoder 架构，根据其他优化工作对架构进行优化

#### Pre-normalization[GPT3]

> [Pre-Norm 和 Post-Norm 的理解](../Optimizer/Pre-Norm%20和%20Post-Norm.md)
> [RMS-Norm](../Optimizer/RMS-Norm.md)

LLaMA normalization 是指的 Layer norm（一般 NLP 任务都是 LN）

- Pre-Norm：为了提高训练稳定性，对每个变换器子层的输入进行 normalizing，而不是对输出 normalizing
```python
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
```

- 使用 RMSNorm 作为 Norm Func

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

#### SwiGLU 激活函数[PaLM]

> [SwiGLU](../Optimizer/SwiGLU.md)

- 使用 SwiGLU 激活函数替换 ReLU 非线性，以提高性能
- 使用 $\frac{2}{3}4d$ 的维度，而不是 PaLM 中的 $4d$
- 采用 SwiGLU 的 FNN，在论文中以如下公式进行表述：
$$F F N_{s w i G L U}\left(x, W, V, W_2\right)=\left(\operatorname{Swish}_1(x W) \otimes x V\right) W_2$$
```python
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

```

#### 旋转位置嵌入 RoPE[GPTNeo]

删除绝对位置嵌入，而在网络的每一层旋转位置嵌入（RoPE）。

### Optimizer

该模型使用 AdamW 优化器进行训练，超参数设置为β1=0.9，β2=0.95。

使用余弦学习率方式，使最终学习率等于最大学习率的 10%，并使用 0.1 的权重衰减和 1.0 的梯度剪裁。最并使用 2,000 个 warm up 策略，并根据模型的大小改变学习率和批次大小。

![](assents/Pasted%20image%2020230516152910.png)

### 模型加速优化

#### 因果多头注意力（causal multi-head attention）

使用了一个高效的因果多头注意力（causal multi-head attention）方式的实现，这个实现可在 xformers 库中找到，可以有效减少了内存的使用和计算。

具体原理为通过不存储注意力权重和不计算由于语言建模任务的因果性质而被掩盖的键/查询分数来实现的。


#### 减少反向传播中重新计算的激活量

我们通过检查点技术减少了在反向传播时重新计算的激活值数量。更具体地说，我们保存了一些昂贵计算的激活值，例如线性层的输出。这通过手动实现转换器层的反向函数来实现，而不是依赖于 PyTorch autograd。

#### 模型和 sequence 并行

为了充分减少模型的内存使用，使用模型和 sequence 并行方式。

#### 重叠激活计算和 GPU 之间的通信过程

还最大程度地重叠激活计算和 GPU 之间的通信（由于 all_reduce 操作）过程。

#### 最终效果

当训练一个 65B 参数的模型时，代码在 2048 个 A100 的 GPU 上处理大约 380 个 token/秒/GPU，并耗费 80GB 的内存，这意味着对包含 1.4Ttoken 的数据集进行训练大约花费了 21 天。

## 实验效果

### zero shot 与 few shot 性能对比测试

Zero-shot 任务指的是提供了任务的文字描述和一个测试例子，该任务要么使用开放式生成提供一个答案，要么对提议的答案进行排序。

Few-shot 任务指的是提供任务的几个例子（1 到 64 个之间）和一个测试例子。该任务将这些文本作为输入，并生成答案或对不同的选项进行排序。

![性能对比](assents/Pasted%20image%2020230516154637.png)
#### Common Sense Reasoning

这些数据集包括 Cloze 和 Winograd style 的任务，以及多选题回答。

![common sense reasoning](assents/Pasted%20image%2020230516154924.png)
####  Closed-book Question Answering

闭卷答题测评任务指的是闭卷情况下的精确匹配性能，即模型不能访问包含回答问题的证据的文件。

![TriviaQA](assents/Pasted%20image%2020230516155311.png)

#### Reading Comprehension

RACE 阅读理解评测指的是从为中国初中和高中学生设计的英语阅读理解考试，效果如表 6 所示：
![RACE](assents/Pasted%20image%2020230516155423.png)
#### Mathematical reasoning

为了验证模型的推理能力，该工作在两个数学推理基准上 MATH 进行了测试。

其中，MATH 是一个用 LaTeX 编写的 12K 初中和高中数学问题的数据集。GSM8k 是一套初中数学问题。

![MATH](assents/Pasted%20image%2020230516155552.png)
#### Code generation

在 HumanEval 测试中，它会收到一个函数签名，提示被格式化为自然码，并在 docstring 中提供文本描述和测试。该模型需要生成一个符合描述并满足测试案例的 Python 程序。

![code gen](assents/Pasted%20image%2020230516155658.png)
对于类似的参数数量，LLaMA 优于其他通用模型，如 LaMDA 和 PaLM，它们没有专门针对代码进行训练或微调。

LLaMA 在 HumanEval 和 MBPP 上以 13B 以上的参数优于 LaMDA 137B。

即使它的训练时间更长，LLaMA 65B 也优于 PaLM 62B。

#### Massive Multitask Language Understanding

由涵盖各种知识领域的多项选择题组成，包括人文、STEM 和社会科学。该工作在 5-shot 的环境中进行了模型评估

![MMLU](assents/Pasted%20image%2020230516155845.png)

LLaMA-65B 在大多数领域都比 Chinchilla-70B 和 PaLM-540B 平均落后几个百分点。

**一个潜在的解释是，该模型在预训练数据中使用了有限的书籍和学术论文，即 ArXiv、Gutenberg 和 Books3，总共只有 177GB，而这些模型是在高达 2TB 的书籍上训练的。**

因此，Gopher、Chinchilla 和 PaLM 所使用的大量书籍可能也解释了为什么 Gopher 在这个基准上优于 GPT-3，而在其他基准上却不相上下。

#### Evolution of performance during training

![Evolution of performance during training](assents/Pasted%20image%2020230516160127.png)

在大多数基准上，性能很快就会提高，并与模型的训练困惑度相关。

不过，SIQA 和 WinoGrande 很例外，最值得注意的是，在 SIQA 上，该工作发现很多性能上的差异，这可能表明这个基准并不可靠。

此外，在 WinoGrande 上，性能与训练困惑度的相关性不大：LLaMA-33B 和 LLaMA-65B 在训练期间的性能相似。

### Instruction Finetuning

Instruction Finetuning 的实验表明：

尽管非微调版本的 LLaMA-65B 已经能够遵循基本指令，但非常小的微调就能提高 MMLU 的性能，并进一步提高模型遵循指令的能力。

由于这不是本文的重点，该工作只进行了一次实验。在模型上采用与 Chung 等人（2022）相同的方法训练一个指令模型，得到 LLaMA-I。

![](assents/Pasted%20image%2020230516160425.png)

尽管这里使用的指令微调方法很简单，但该模型在 MMLU 上达到了 68.9%。

LLaMA-I（65B）在 MMLU 上超过了现有的中等规模的指令微调模型，但离最先进的水平有较大的差距，即 GPT 代码-DAVINCI-002 在 MMLU 上的表现为 77.4%。

## Bias, Toxicity and Misinformation（偏见、有害性和误导性）

大型语言模型已被证明可以重现和放大训练数据中存在的偏见，并产生有毒或攻击性内容。

