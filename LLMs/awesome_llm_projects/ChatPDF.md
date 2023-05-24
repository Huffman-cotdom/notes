> GitHub 地址：https://github.com/akshata29/chatpdf

1. 把 PDF 切分成小的文本片段，通过 OpenAI 的 Ada 模型创建 Embedding 放到本地或远程向量数据库。
2. 把用户的提问也创建成 Embedding，用它和之前创建的 PDF 向量比对，通过语义相似性搜索（余弦相似度算法），找到最相关的文本片段。比关键词搜索好的一点是不要求关键词包含，也能发现文本相关性，比如汽车和公路。
3. 把用户提问和相似文本片段发给 OpenAI，写 Prompt 要求 ChatGPT 基于给定的内容生成回答，如果没有相似文本或关联度不高，回答不知道。

为避免 ChatGPT 乱发挥，一般 Temperture 会设置的很低甚至为 0。

注意：这种方法实用性仍然比较有限，质量也不好，虽有一定调优空间（文本切片，问答对）现在这么做也是不得已，因为 ChatGPT 的上下文记忆 token 有限（32k 会好一些），不能直接丢超长文档让它分析。

![Architecture](assents/Pasted%20image%2020230522232538.png)