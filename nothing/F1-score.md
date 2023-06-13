## Accuracy-准确率



**准确率是分类模型的一个指标，用于衡量正确的预测数量占所做预测总数的百分比。**

$$Acc=\frac{total\ correct\ predictions}{total\ predictions}$$

### 缺点

**只有当分类中的类分布均等时，准确率才是有用的指标。这意味着，如果你用例中的一类数据点多于另一类的数据点，那么准确率就不是一个有用的指标。**

示例：

如果 100 个样本中 99 个为分类 0，1 个为分类 1。

如果模型预测 100 个样本全部是分类 0，这时 Acc 为 99%，但是模型效果很差。

解决方案：

1. **通过重采样解决不平衡数据**：包括欠采样、过采样、smoth。
2. **通过指标解决不平衡数据**：例如 F1-score。

## F1-score

基本概念：

**TP：**True Positive，分类器预测结果为正样本，实际也为正样本，即正样本被正确识别的数量。

**FP：**False Positive，分类器预测结果为正样本，实际为负样本，即**误报**的负样本数量。

**TN：**True Negative，分类器预测结果为负样本，实际为负样本，即负样本被正确识别的数量。

**FN：**False Negative，分类器预测结果为负样本，实际为正样本，即**漏报**的正样本数量。

### Precision - 精准度

> 模型预测得准不准

$$Precision=\frac{TP}{TP+FP}$$

- 一个**不精确**的模型可能会找到很多正样本，但它的选择方式很杂乱：它也会错误地预测到许多实际上不是正样本的正样本。
- 一个**精确**的模型是非常“纯粹的”：也许它没有找到所有的正样本，但预测结果很可能是正确的。

### Recall - 召回率

> 模型预测得全不全

$$Recall=\frac{TP}{TP+FN}$$

- 具有高召回率的模型可以很好地找到数据中所有正样本，即使它们可能错误地将一些负样本识别为正样本。
- 召回率低的模型无法找到数据中所有*（或大部分）*正样本。

### F1-score

> **精确率和召回率权衡**

**在许多情况下，你可以调整模型降低召回率来提高精确率，或者另一方面以降低精确率为代价来提高召回率。**

F1-score 定义为精确率和召回率的 harmonic mean - 调和平均值。调和平均值是算术平均值的替代指标。

harmonic mean: https://zhuanlan.zhihu.com/p/95763963

根据 harmonic mean 的公式可得：

$$F1\ score=(\frac{Precision^{-1}+Recall^{-1}}{2})$$

最终得到：

$$F1\ score=\frac{2*Precision*Recall}{Precision+Recall}$$

**由于 F1-score 是精确率和召回率的平均值，这意味着它对精确率和召回率的权重相同：**

- 如果精确率和召回率都很高，模型将获得较高的 F1-score
- 如果精确率和召回率都较低，模型将获得较低的 F1-score
- 如果精确率和召回率中的一个低而另一个高，则模型将获得中等 F1-score

有的时候，我们对 recall 与 precision 赋予不同的权重，表示对分类模型的偏好：

$$F_\beta=\frac{(1+\beta^2)\mathrm{TP}}{(1+\beta^2)\mathrm{TP}+\beta^2\mathrm{FN}+\mathrm{FP}}=\frac{(1+\beta^2)\cdot\mathrm{Precision}\cdot\mathrm{Recall}}{\beta^2\cdot\mathrm{Precision}+\mathrm{Recall}}$$

可以看到，当 $β = 1$ ，那么$F_{\beta}$ 就退回到$F1$了，$β$ 其实反映了模型分类能力的偏好。

$\beta>1$ 的时候，precision 的权重更大，为了提高$F_{\beta}$ ，我们希望 precision 越小，而 recall 应该越大，说明模型更偏好于提升 recall，意味着模型更看重对正样本的识别能力。

 $\beta<1$ 的时候，recall 的权重更大，因此，我们希望 recall 越小，而 precision 越大，模型更偏好于提升 precision，意味着模型更看重对负样本的区分能力。







