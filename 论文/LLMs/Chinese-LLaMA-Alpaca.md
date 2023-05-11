> GitHub 地址：https://github.com/ymcui/Chinese-LLaMA-Alpaca
> 
> 论文地址：[Chinese-LLaMA](assents/Chinese-LLaMA.pdf)
> 
> ChatDoc： https://chatdoc.com/chatdoc/#/chat/35282232-3e57-4d68-9cef-c0563b852a34

# 摘要

> https://github.com/facebookresearch/llama/issues/58 LLaMA tokenizer 似乎只有 700 个汉字

原始 LLaMA 对中文支持较差，通过增加 20K 个中文 token 扩展了原始 LLaMA 的中文词汇表，增加了**编码效率并提高了基本语义理解**；通过将中文数据进行辅助预训练并使用中文指令数据进行微调，可大大改善模型对指令的理解和执行能力。

# 主要贡献

- 通过向原始 LLaMA 的词汇表中添加 20k 个中文词汇，增强了中文编码和解码效率并改善了 LLaMA 的中文理解能力。
- 采用 Low-Rank Adaptation（LoRA）方法，实现了中文 LLaMA 和 Alpaca 模型的高效训练和部署。
- 评估了中文 Alpaca 7B 和 13B 模型在各种自然语言理解 NLU 和自然语言生成 NLG 任务中的表现，在中文语言任务中相比原始 LLaMA 模型取得了显著的提高。

# Chinese LLaMA

## LLaMA 简介

LLaMA 是一种基于 Transformer 架构的仅解码器型、基础性大语言模型。与其他基于 Transformer 的语言模型类似，LLaMA 包括嵌入层、多个 Transformer 块和一个 LM 头层。此外，它还采用了各种改进，如 Pre-Norm、SwiGLU 激活 和 旋转嵌入（Rotary Embeddings）。LLaMA 的总参数数从 7B 到 65B 不等。实验证明，LLaMA 在保持更小的模型大小的同时，达到了与其他语言模型（如 GPT-3）竞争性的性能表现。

## LLaMA 中文支持有限

LLaMA 已经在公开可用的语料库中预先训练了 1T 到 1.4T 个标记，其中大部分数据为英语，只有少部分为采用拉丁或西里尔字母的其他语言。因此，LLaMA 对于理解和生成中文的能力有限。

## 中文语料预训练 LLaMA

### 挑战

- 原始的 LLaMA 分词器词汇表中只有不到一千个中文字符。虽然 LLaMA 分词器支持通过回退到字节编码方式来处理中文字符，但这种回退策略会显著增加序列长度，并减慢处理中文文本的效率。
- 字节标记不仅用于表示中文字符，还用于表示其他 UTF-8 标记，这使得字节标记难以学习中文字符的语义含义

### 解决方案

为增强分词器对中文文本的支持，首先使用 SentencePiece 在中文语料上训练一个中文分词器，使用词汇表大小为 20,000。然后，将中文分词器与原始 LLaMA 分词器合并，结合它们的词汇表，最终得到一个合并的分词器，我们称之为中文 LLaMA 分词器，其词汇表大小为 49,953。

为了适应中文 LLaMA 分词器，我们将词嵌入和语言模型头从形状 V×H 调整为 V′×H，其中 V=32,000 表示原始词汇表大小，V′=49,953 是中文 LLaMA 分词器的词汇表大小。新行附加到原始嵌入矩阵的末尾，确保原始词汇表中标记的嵌入不受影响。

中文 LLaMA 分词器生成的标记数量约为原始 LLaMA 分词器的一半。从表中可以看出使用中文 LLaMA 分词器显著减少了编码长度，相比原始分词器，给定固定上下文长度的情况下，模型可以容纳大约两倍的信息量，并且生成速度比原始 LLaMA 分词器快两倍。

![原始 LLaMA 与 Chinese LLaMA 编码对比](assents/Pasted%20image%2020230511234417.png)

# Chinese Alpaca

> 使用 Stanford Alpaca 的方式微调 Chinese LLaMa

每个示例包括一条指令 instruction 和一个输出 output。我们将指令 instruction 输入到模型中，并提示模型自回归地生成输出。这个过程类似于常规的语言建模任务。我们采用自我指导微调的斯坦福 Alpaca 提供的以下提示模板，在推理过程中也会使用该模板：

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response: {output}
```

斯坦福 Alpaca 的一个**关键区别**在于，没有针对输入 input 字段的示例设计的提示模板，而 Stanford Alpaca 则分别使用两个示例模板，一个用于有输入 input 字段的示例，另一个用于没有输入 input 字段的示例。如果示例包含非空输入 input 字段，则使用“\n”将指令和输入连接起来形成新的指令。请注意，Alpaca 模型还有一个**额外的填充令牌**，因此词汇量为 49,954。

# Train

## Pre-Training

使用原始 LLaMA 权重初始化 Chinese LLaMA 模型，并在一般的中文文本语料库上进行预训练，这与用于 Chinese BERT-wwm、MacBERT 等的文本语料库一致，总共为 20GB。预训练包括两个阶段：
- 阶段一：固定模型中的 Transformer 编码器参数，仅训练嵌入，适应新添加的中文词向量，同时最小化对原始模型的干扰。
- 阶段 2：向注意力机制添加 LoRA 权重（适配器），并训练嵌入、LM heads 和新添加的 LoRA 参数。

## Fine-tuning

### 7B

使用 LoRA 进行微调，通过向 MLP 层添加 LoRA 适配器来增加可训练参数的数量。利用了约 200 万个数据，包括翻译、pCLUE 3、Stanford Alpaca 和爬取的 SFT 数据，以调整 7B 模型。

### 13B

预训练 13B 模型的预训练过程与 7B 模型的大部分相同，但我们跳过了第一阶段的预训练。我们直接将 LoRA 应用于注意力机制和 MLP 进行训练，同时将嵌入层和 LM 头设置为可训练。

指令微调 LoRA 的设置和可训练参数与预训练阶段相同。使用了额外的 100 万个爬取的 ChatGPT 数据，用于 13B 模型微调，从而使总数据量为 3M。

### 超参数

上下文大小：我们将上下文大小设置为 2048，这决定了模型在生成文本时可以同时考虑的最大标记数量。

最大序列长度：我们将生成的序列长度限制为 512 个标记，以确保输出始终与输入提示保持专注和相关。

温度：我们将温度设置为 0.2，这控制着采样过程的随机性。较低的值使模型生成更加专注和确定的输出，而较高的值在增加多样性的同时降低一致性。

前 k 个采样：我们使用顶部 k 个采样，k = 40，这意味着模型在每步中从概率最高的前 40 个标记中选择下一个标记，增加了生成文本的随机性和多样性。

前 p 个采样：我们还采用前 p 个采样，p = 0.9，通过考虑动态一起占据 90％概率质量的标记集合，进一步增强了多样性。

重复惩罚：为了防止模型生成重复文本，我们应用一个重复惩罚因数为 1.3，对已经选择的标记进行惩罚。

![超参数](assents/Pasted%20image%2020230512000456.png)

