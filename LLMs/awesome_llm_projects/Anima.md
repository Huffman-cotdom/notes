# Anima

## Backbone 模型选择

Anima 模型基于 QLoRA 开源的[33B guanaco](https://huggingface.co/timdettmers/guanaco-33b)训练了 10000 steps。训练使用一个 H100 GPU。

- **思考逻辑**：本工作主要为了验证 QLoRA 训练方法的有效性，因此选择了基于 QLoRA 的 Guanaco 33B finetune 训练，这个训练更多的是增强模型的中文能力。Assume 模型的基础 logical reasoning 和 Knowledge 能力已经足够。

## 训练数据选择

使用[Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna)项目开放的训练数据集[guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)进行 finetune 训练。

- **思考逻辑**：按照[QLoRA](https://arxiv.org/abs/2305.14314) Appendix B.4 和 Table 9 中的 Grid Search 的结论：对于 QLoRA finetune，training sample 量不一定越大越好。10000 个 steps 是一个 ROI 比较优的 size。因此我们希望选择一个不小于 10000 个 steps 的数据集。[Belle 10M](https://github.com/LianjiaTech/BELLE/blob/main/data/10M)数据集似乎太大了，不确定数据质量如何。时间有限，先选择 guanaco_belle_merge_v1.0。后边会进一步更系统性的测试更多的数据集和数据质量筛选的效果。

## 超参选择

基于成本 ROI 平衡的考虑，没有做太多的 grid search，基本的思路是 follow [QLoRA paper](https://arxiv.org/abs/2305.14314) 的结论，因为 QLoRA 做了相对比较详尽的超参 Grid Search 实验：

- Batch size: 16 ([QLoRA](https://arxiv.org/abs/2305.14314) Appendix B.4 和 Table 9)
- Max steps: 10000 ([QLoRA](https://arxiv.org/abs/2305.14314) Appendix B.4 和 Table 9)，更多的 steps 和更大的数据集的训练在进一步实验中，后续会持续更新。
- Learning rate: 1e-4 ([QLoRA](https://arxiv.org/abs/2305.14314) Appendix B.4 和 Table 9)
- LoRA r=64, alpha=16 ([QLoRA](https://arxiv.org/abs/2305.14314) Appendix B.2)
- source_max_len=512, target_max_len=512，需要保证大部分的 training sample 没有 truncate，能完整的把信息训练到模型中，根据[脚本](https://github.com/lyogavin/Anima/blob/main/scripts/test_cn_dataset_lenghts.py)中的估计，512 大概可以覆盖大部分的样本长度。

## 验证评估

#### Elo rating tournament 结论

| Model             | Elo         | Rank  |
| ----------------- | ----------- | ----- |
| ChatGPT-3.5 turbo | 1341.98     | 1     |
| **Anima 33B**     | **1096.69** | **2** |
| Belle             | 937.71      | 3     |
| Chinese Vicuna    | 623.62      | 4     |

#### 评估方法论

- **数据集的选择**：如[Belle Paper](https://github.com/LianjiaTech/BELLE/blob/main/docs/Towards Better Instruction Following Language Models for Chinese.pdf)中论述，评估集的不同类型分布对于评估结论影响巨大。如田忌赛马，以己之长攻人之短，很容易占优势。因此我们选择了英文 chatbot 模型研究工作中比较普遍公认的[Vicuna benchmark](https://lmsys.org/blog/2023-03-30-vicuna/)。为了评测中文，我们使用 GPT4 对于问题做了翻译。[![Open Anima in Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/lyogavin/Anima/blob/main/data/gpt4_translate_vicuna_eval_set.ipynb) [翻译代码](https://github.com/lyogavin/Anima/blob/main/data/gpt4_translate_vicuna_eval_set.ipynb)和[数据集](https://github.com/lyogavin/Anima/blob/main/data/translated_vicuna_eval_set.json)。
- **评估方法**: 为了平衡成本，我们主要采用 GPT4 进行评估。如[QLoRA](https://arxiv.org/abs/2305.14314) 论证，单纯 GPT4 打分进行模型的对比随机波动性较大。这与我们的观察一致。因此采用了[QLoRA](https://arxiv.org/abs/2305.14314) 推荐的，现在比较普遍采用的 Elo Rating tournament 评测方法。
- **超参选择**：出于成本考虑，我们选择：300 轮随机评估，随机选择模型 PK 的先后顺序以抵消先后顺序的影响，随机种子为：42。Elo rating 的实现代码和其他超参参照[Vicuna 的 Elo 代码](https://raw.githubusercontent.com/lm-sys/FastChat/833d65032a715240a3978f4a8f08e7a496c83cb1/fastchat/serve/monitor/elo_analysis.py): K=32, init rating=1000。