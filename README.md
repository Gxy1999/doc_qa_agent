# 文档智能问答 Agent（DocQA Agent）

一个基于 **DeepSeek 大语言模型** + **多 Agent 协作** 的企业内部文档问答系统。能够自动从私有文档（Markdown、TXT 等）中检索信息，并通过长链推理生成准确、带引用的答案。支持混合检索（向量 + BM25）、时效性过滤、敏感词脱敏。

## ✨ 核心特性

- **多 Agent 协作**：检索 Agent、评审 Agent、合成 Agent、安全 Agent 各司其职，系统可扩展、可替换。
- **长链推理**：自动拆解用户问题，生成多个子查询，合并多路召回结果，提升复杂问答的完整度。
- **混合检索**：向量检索（BAAI/bge-large-zh）+ BM25 关键词检索，兼顾语义理解与精确匹配。
- **时效性控制**：评审 Agent 可根据问题中的“最新/最近”等关键词过滤陈旧文档片段。
- **安全审计**：输出前自动屏蔽内部敏感词，防止信息泄露。
- **开箱即用**：提供完整的索引构建、查询、API 化示例，支持增量更新。

## 🛠️ 技术栈

- **Python 3.10+**
- **DeepSeek API**（可替换为任意 OpenAI 兼容接口）
- **ChromaDB**（向量数据库）
- **BM25s + jieba**（中文 BM25 检索）
- **Sentence-Transformers**（BGE 中文嵌入模型）

## 📦 安装

```bash
# 克隆项目
git clone https://github.com/yourname/docqa-agent.git
cd docqa-agent

# 安装依赖
pip install -r requirements.txt
