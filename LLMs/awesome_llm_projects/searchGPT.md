> GitHub 地址：https://github.com/michaelthwan/searchGPT

架构图：

![searchGPT 架构图](assents/Pasted%20image%2020230526000952.png)

大致流程：

1. query: “What is ChatGPT？”使用搜索引擎搜索，并取前几篇文档。
2. 将这几篇文章切分成句子并且转换为 embedding，并通过 SemanticSearchService 检索 embedding。
3. 将召回的 embedding 对应句子使用 Prompt 构建为摘要并返回给用户
	- Prompt：Summarize above sources with query: "What is ChatGPT" using 80 words
