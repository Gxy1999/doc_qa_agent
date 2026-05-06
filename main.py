"""
文档智能问答 Agent - 基于 DeepSeek + 多 Agent 协作
依赖安装: pip install chromadb sentence-transformers openai bm25s jieba
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple
import chromadb
from chromadb.utils import embedding_functions
import bm25s
import jieba
from openai import OpenAI

# ==================== 配置区 ====================
DEEPSEEK_API_KEY = "xxxxxxx"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
CHROMA_PERSIST_DIR = "./chroma_db"
DOCS_ROOT = "./internal_docs"      # 存放所有 .txt/.md 的文件夹
CHUNK_SIZE = 500                   # 每个块最大长度（中文字符）

# 初始化 LLM 客户端
llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 初始化向量库 + 中文嵌入模型
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-zh"
)
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
vector_collection = chroma_client.get_or_create_collection(
    name="docs_chunks",
    embedding_function=embedding_fn
)

# 全局 BM25 索引（在启动时构建）
bm25_index = None
bm25_corpus = []      # 存储原始文本块
bm25_ids = []         # 存储块对应的 metadata id

# ==================== 1. 离线索引构建 ====================
def chunk_document(text: str, doc_name: str, chunk_size: int = CHUNK_SIZE) -> List[Dict]:
    """将长文档按字符切块，保留元数据"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i+chunk_size]
        chunk_id = hashlib.md5(f"{doc_name}_{i}".encode()).hexdigest()
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "source": doc_name,
                "chunk_index": i // chunk_size,
                "update_time": datetime.now().isoformat()
            }
        })
    return chunks

def build_index():
    """扫描 DOCS_ROOT 下所有文件，构建向量库 + BM25 索引"""
    global bm25_index, bm25_corpus, bm25_ids
    all_chunks = []
    for fname in os.listdir(DOCS_ROOT):
        if not fname.endswith((".txt", ".md")):
            continue
        with open(os.path.join(DOCS_ROOT, fname), "r", encoding="utf-8") as f:
            content = f.read()
        chunks = chunk_document(content, fname)
        all_chunks.extend(chunks)
    
    # 插入向量库（如果已存在则先清空）
    if vector_collection.count() > 0:
        vector_collection.delete(ids=[str(i) for i in range(vector_collection.count())])
    for chunk in all_chunks:
        vector_collection.add(
            ids=[chunk["id"]],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]]
        )
    
    # 构建 BM25 索引（中文分词）
    corpus_tokens = [list(jieba.cut(chunk["text"])) for chunk in all_chunks]
    bm25_index = bm25s.BM25()
    bm25_index.index(corpus_tokens)
    bm25_corpus = [chunk["text"] for chunk in all_chunks]
    bm25_ids = [chunk["id"] for chunk in all_chunks]
    print(f"索引构建完成，共 {len(all_chunks)} 个块")

# ==================== 2. 检索 Agent ====================
def retrieve_agent(query: str, top_k: int = 12) -> List[Tuple[str, Dict, float]]:
    """
    混合检索：向量检索 + BM25，然后融合排序（RRF）
    返回 [(text, metadata, score), ...]
    """
    # 向量检索
    vector_results = vector_collection.query(query_texts=[query], n_results=top_k)
    vector_ids = vector_results['ids'][0]
    vector_scores = vector_results['distances'][0]   # L2距离，越小越好，转成相似度
    vector_map = {vid: (1.0 - (score / 2)) for vid, score in zip(vector_ids, vector_scores)}  # 粗略归一化
    
    # BM25 检索
    query_tokens = list(jieba.cut(query))
    bm25_results = bm25_index.retrieve(query_tokens, k=top_k)
    bm25_map = {}
    for doc_id, score in zip(bm25_results[0], bm25_results[1]):
        bm25_map[bm25_ids[doc_id]] = score
    
    # 融合 (RRF: reciprocal rank fusion)
    all_ids = set(vector_map.keys()) | set(bm25_map.keys())
    fused = {}
    k = 60
    for doc_id in all_ids:
        rank_v = 1.0 / (k + sorted(vector_map.keys(), key=lambda x: vector_map[x], reverse=True).index(doc_id) + 1) if doc_id in vector_map else 0
        rank_b = 1.0 / (k + sorted(bm25_map.keys(), key=lambda x: bm25_map[x], reverse=True).index(doc_id) + 1) if doc_id in bm25_map else 0
        fused[doc_id] = rank_v + rank_b
    
    sorted_docs = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # 获取真实文本和元数据
    results = []
    for doc_id, score in sorted_docs:
        # 从向量库获取完整信息（也可维护内存映射）
        single = vector_collection.get(ids=[doc_id])
        results.append((single['documents'][0], single['metadatas'][0], score))
    return results

# ==================== 3. 评审 Agent（过滤过时内容）====================
def reviewer_agent(chunks: List[Tuple[str, Dict, float]], query: str) -> List[Tuple[str, Dict, float]]:
    """
    检查每个 chunk 是否过时（如果用户询问特定时间点后的信息）
    简单实现：若 metadata 中的 update_time 晚于 query 中隐含的时间则保留
    （生产环境可调用 LLM 判断）
    """
    # 简单启发式：如果 query 包含“最近”、“最新”、“2025”等词，则过滤掉较旧的块（示例）
    if "最近" in query or "最新" in query:
        # 只保留最近 30 天更新的文档
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=30)
        filtered = []
        for text, meta, score in chunks:
            update_time = datetime.fromisoformat(meta["update_time"])
            if update_time >= cutoff:
                filtered.append((text, meta, score))
        return filtered
    return chunks

# ==================== 4. 安全 Agent（输出前敏感词过滤）====================
def security_agent(answer: str) -> str:
    """扫描并屏蔽内部敏感词（演示用）"""
    sensitive_words = ["未公开版本号", "内部服务器密码", "客户A"]
    for word in sensitive_words:
        if word in answer:
            answer = answer.replace(word, "[已屏蔽]")
    return answer

# ==================== 5. 合成 Agent（调用 DeepSeek 生成最终答案）====================
def synthesize_agent(query: str, retrieved_chunks: List[Tuple[str, Dict, float]], conversation_history: List[Dict] = None) -> str:
    """将检索块 + 历史对话 送入 LLM，要求带引用推理"""
    context = "\n\n---\n\n".join([f"[来源: {meta['source']} 更新时间:{meta['update_time']}]\n{text}" for text, meta, _ in retrieved_chunks])
    
    system_prompt = """你是一个严谨的文档问答助手。你需要：
1. 只基于下面提供的【参考文档片段】回答问题。
2. 如果片段不足以回答问题，请明确说“根据现有文档无法回答”。
3. 在答案末尾用 [来源: xxx] 标注引用了哪些文档。
4. 如果你认为不同文档片段之间存在冲突，请指出来。
5. 采用逐步推理的方式，先列出关键点，再给出最终答案。
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"参考文档片段：\n{context}\n\n用户问题：{query}\n\n请回答："}
    ]
    # 如果有对话历史，插入（略）
    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.2,
        max_tokens=2000
    )
    return response.choices[0].message.content

# ==================== 6. 主流程（长链推理 + 多 Agent 编排）====================
def long_chain_reasoning(query: str, history: List[Dict] = None) -> str:
    """
    长链推理步骤:
    1. 意图拆解（是否多源？是否需时间过滤？是否需对比？）
    2. 改写 query 生成3个变体（示例）
    3. 调用检索 Agent 获取混合结果
    4. 评审 Agent 过滤过时内容
    5. 合成 Agent 生成答案
    6. 安全 Agent 脱敏输出
    """
    # Step 1: 简单意图识别（可用 LLM，此处用规则）
    sub_queries = [query]
    if "对比" in query or "差异" in query:
        # 扩展对比查询
        sub_queries.append(query + " 版本A vs 版本B")
    if "如何" in query or "步骤" in query:
        # 加重流程类查询
        sub_queries.append(query + " 操作步骤")
    
    # Step 2: 合并所有子查询的检索结果
    all_chunks = []
    for q in sub_queries:
        chunks = retrieve_agent(q, top_k=8)
        all_chunks.extend(chunks)
    # 按融合得分去重保留最高分
    unique = {}
    for text, meta, score in all_chunks:
        key = meta["source"] + str(meta["chunk_index"])
        if key not in unique or unique[key][2] < score:
            unique[key] = (text, meta, score)
    merged_chunks = list(unique.values())
    
    # Step 3: 评审 Agent
    reviewed = reviewer_agent(merged_chunks, query)
    if not reviewed:
        return "抱歉，根据现有文档（且经过时效性过滤后）没有找到相关信息。"
    
    # Step 4: 合成 Agent
    raw_answer = synthesize_agent(query, reviewed, history)
    
    # Step 5: 安全 Agent
    safe_answer = security_agent(raw_answer)
    return safe_answer

# ==================== 示例调用 ====================
if __name__ == "__main__":
    # 首先构建索引（只需运行一次，后续可注释）
    if not os.path.exists(DOCS_ROOT):
        os.makedirs(DOCS_ROOT)
        # 创建一个示例文档
        with open(os.path.join(DOCS_ROOT, "example.txt"), "w", encoding="utf-8") as f:
            f.write("""# 内部发布流程
1. 拉取最新代码
2. 运行单元测试
3. 执行构建命令 npm run build
4. 将 dist/ 上传至 OSS
发布时间：2026-05-01
""")
    build_index()
    
    # 提问
    answer = long_chain_reasoning("最新的发布流程是什么？")
    print("=== Agent 回答 ===")
    print(answer)