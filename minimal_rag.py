#!/usr/bin/env python3
"""
增强版最小 RAG 脚本
功能：文档加载 → 智能切分 → 向量化 → TopK检索 → LLM生成
特性：错误处理 / 流式输出 / 缓存 / 重试 / 超时控制 / 多策略优化
"""

import os
import sys
import re
import math
import time
import json
import copy
import hashlib
import argparse
import threading
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

# ============================================================
# 1. 配置管理（从环境变量读取，不硬编码）
# ============================================================
def load_env():
    """加载 .env 文件"""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

load_env()

# 配置常量
@dataclass
class Config:
    knowledge_dir: str = str(Path(__file__).parent / "knowledge_docs")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    llm_base_url: str = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    llm_api_key: str = os.getenv("OPENAI_API_KEY", "sk-placeholder")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "30"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

CFG = Config()

# ============================================================
# 2. 数据结构
# ============================================================
@dataclass
class Chunk:
    content: str
    source: str
    chunk_id: int
    score: float = 0.0

@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    query: str
    total_searched: int
    search_time: float
    from_cache: bool = False

@dataclass
class Answer:
    text: str
    sources: list[dict]
    confidence: float
    retrieval_result: Optional[RetrievalResult] = None

# ============================================================
# 3. 缓存层
# ============================================================
class LRUCache:
    """简单的 LRU 缓存"""
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache: dict[str, tuple[float, any]] = {}
        self.order: list[str] = []
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[any]:
        with self._lock:
            if key in self.cache:
                expire_time, value = self.cache[key]
                # 检查是否过期
                if time.time() > expire_time:
                    del self.cache[key]
                    self.order.remove(key)
                    return None
                self.order.remove(key)
                self.order.append(key)
                return value
        return None

    def put(self, key: str, value: any, ttl: float = 3600):
        with self._lock:
            if key in self.cache:
                self.order.remove(key)
            elif len(self.cache) >= self.capacity:
                oldest = self.order.pop(0)
                del self.cache[oldest]
            self.cache[key] = (time.time() + ttl, value)
            self.order.append(key)

    def make_key(self, prefix: str, *args) -> str:
        raw = f"{prefix}:{':'.join(str(a) for a in args)}"
        return hashlib.md5(raw.encode()).hexdigest()

search_cache = LRUCache(capacity=200)

# ============================================================
# 4. 文档加载
# ============================================================
def load_documents(doc_dir: str) -> list[dict]:
    """递归加载目录下所有 .md 文件"""
    docs = []
    doc_path = Path(doc_dir)
    if not doc_path.exists():
        print(f"  [ERROR] 文档目录不存在: {doc_dir}")
        return docs
    for md_file in doc_path.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            if content.strip():
                docs.append({
                    "content": content,
                    "source": str(md_file.relative_to(doc_path)),
                })
        except Exception as e:
            print(f"  [WARN] 跳过 {md_file.name}: {e}")
    return docs

# ============================================================
# 5. 智能文本切分（重叠切分 + 语义感知）
# ============================================================
def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """智能文本切分：优先按段落/句子边界切分，支持重叠"""
    if not text.strip():
        return []

    # 如果文本很短，直接返回
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks = []
    # 按段落分割
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 当前块 + 新段落 不超过限制
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        else:
            # 保存当前块
            if current_chunk:
                chunks.append(current_chunk.strip())

            # 段落本身超过限制，按句子切分
            if len(para) > chunk_size:
                sentences = _split_by_sentences(para)
                sub_chunk = ""
                for sent in sentences:
                    if len(sub_chunk) + len(sent) + 1 <= chunk_size:
                        sub_chunk = f"{sub_chunk}{sent}" if sub_chunk else sent
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                        sub_chunk = sent
                if sub_chunk:
                    current_chunk = sub_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # 添加重叠
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, chunk_overlap)

    return chunks


def _split_by_sentences(text: str) -> list[str]:
    """按句子边界切分"""
    # 中英文句子分隔符
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    return [s for s in sentences if s.strip()]


def _add_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    """为相邻 chunk 添加重叠（从尾部找最近的自然断点）"""
    result = [chunks[0]]
    separators = ["。", ".", "！", "!", "？", "?", "\n", "；", ";", "，", ",", "、", " "]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap_chars:]
        # 从尾部向前找最近的自然断点
        best_pos = -1
        for sep in separators:
            idx = prev_tail.rfind(sep)  # 从右侧查找
            if idx > best_pos:
                best_pos = idx
        if best_pos > 0:
            prev_tail = prev_tail[best_pos + 1:]
        result.append(f"{prev_tail}{chunks[i]}")
    return result


def build_chunks(docs: list[dict], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """将文档切分为 Chunk 对象"""
    all_chunks = []
    for doc in docs:
        text_chunks = split_text(doc["content"], chunk_size, chunk_overlap)
        for i, chunk_text in enumerate(text_chunks):
            all_chunks.append(Chunk(
                content=chunk_text,
                source=doc["source"],
                chunk_id=i,
            ))
    return all_chunks

# ============================================================
# 6. 向量索引（FAISS IndexFlatIP）
# ============================================================
class VectorIndex:
    """FAISS 向量索引封装"""

    def __init__(self, model_name: str = CFG.embedding_model):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks: list[Chunk] = []
        self._built = False

    def build(self, chunks: list[Chunk]):
        """构建索引"""
        import numpy as np

        self.chunks = chunks
        print(f"  加载 Embedding 模型: {self.model_name}")

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"  [ERROR] Embedding 模型加载失败: {e}")
            raise

        texts = [c.content for c in chunks]
        print(f"  编码 {len(texts)} 个 chunks...")

        try:
            embeddings = self.model.encode(
                texts, show_progress_bar=True, normalize_embeddings=True
            )
            embeddings = np.array(embeddings, dtype="float32")
        except Exception as e:
            print(f"  [ERROR] 向量编码失败: {e}")
            raise

        dim = embeddings.shape[1]
        import faiss
        self.index = faiss.IndexFlatIP(dim)  # 内积（归一化后=余弦相似度）
        self.index.add(embeddings)
        self._built = True
        print(f"  索引构建完成: {len(chunks)} 个向量, 维度 {dim}")

    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> list[Chunk]:
        """检索最相似的 chunks，带过滤"""
        if not self._built:
            print("  [ERROR] 索引未构建")
            return []

        import numpy as np

        try:
            query_vec = self.model.encode([query], normalize_embeddings=True)
            query_vec = np.array(query_vec, dtype="float32")
            scores, indices = self.index.search(query_vec, top_k * 2)  # 多取一些用于过滤
        except Exception as e:
            print(f"  [ERROR] 检索失败: {e}")
            return []

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            if score < threshold:
                continue
            chunk = Chunk(
                content=self.chunks[idx].content,
                source=self.chunks[idx].source,
                chunk_id=self.chunks[idx].chunk_id,
                score=float(score),
            )
            results.append(chunk)

        # 去重：同来源的合并（保留最高分）
        results = _deduplicate(results)
        # 排序：按相似度降序
        results.sort(key=lambda c: c.score, reverse=True)
        # 截取 top_k
        return results[:top_k]


def _deduplicate(chunks: list[Chunk], max_per_source: int = 2) -> list[Chunk]:
    """同来源去重：每个来源最多保留 max_per_source 个最高分 chunk"""
    from collections import defaultdict
    source_chunks: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        source_chunks[c.source].append(c)
    result = []
    for source, cks in source_chunks.items():
        cks.sort(key=lambda c: c.score, reverse=True)
        result.extend(cks[:max_per_source])
    return result

# ============================================================
# 7. BM25 稀疏检索 + 混合检索
# ============================================================
class BM25Index:
    """轻量 BM25 索引（基于词频的稀疏检索）"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_freqs: list[dict[str, int]] = []  # 每个文档的词频
        self.idf: dict[str, float] = {}
        self.avg_dl = 0
        self._built = False

    def _tokenize(self, text: str) -> list[str]:
        """简单分词：按字符级别 + 英文单词"""
        # 英文单词提取
        words = re.findall(r'[a-zA-Z]+', text.lower())
        # 中文字符级分词（bigram）
        chinese = re.findall(r'[一-鿿]', text)
        bigrams = [chinese[i] + chinese[i+1] for i in range(len(chinese)-1)]
        # 单字也保留
        return words + bigrams + chinese

    def build(self, chunks: list[Chunk]):
        """构建 BM25 索引"""
        from collections import Counter
        self.chunks = chunks
        self.doc_freqs = []
        df = Counter()  # 文档频率

        total_dl = 0
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            total_dl += sum(freq.values())
            for term in set(tokens):
                df[term] += 1

        n = len(chunks)
        self.avg_dl = total_dl / max(n, 1)
        # IDF: log((N - df + 0.5) / (df + 0.5))，标准 BM25 IDF
        self.idf = {term: max(0, math.log((n - freq + 0.5) / (freq + 0.5))) for term, freq in df.items()}
        self._built = True

    def search(self, query: str, top_k: int = 10) -> list[Chunk]:
        """BM25 检索"""
        if not self._built:
            return []
        query_tokens = self._tokenize(query)
        scores = []
        for i, freq in enumerate(self.doc_freqs):
            dl = sum(freq.values())
            score = 0.0
            for qt in query_tokens:
                if qt in freq:
                    tf = freq[qt]
                    idf_val = self.idf.get(qt, 0)  # 已预计算 log 值
                    tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1)))
                    score += idf_val * tf_norm
            scores.append((score, i))

        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            if score > 0:
                results.append(Chunk(
                    content=self.chunks[idx].content,
                    source=self.chunks[idx].source,
                    chunk_id=self.chunks[idx].chunk_id,
                    score=score,
                ))
        return results


def hybrid_search(vector_index: VectorIndex, bm25_index: BM25Index,
                  query: str, top_k: int, threshold: float,
                  alpha: float = 0.7) -> list[Chunk]:
    """混合检索：向量检索 + BM25 稀疏检索，加权融合

    alpha: 向量检索权重 (1-alpha 为 BM25 权重)
    """
    # 向量检索（多取一些）
    vec_results = vector_index.search(query, top_k=top_k * 3, threshold=0.0)
    # BM25 检索
    bm25_results = bm25_index.search(query, top_k=top_k * 3)

    # 归一化分数到 [0, 1]
    def normalize(chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        max_score = max(c.score for c in chunks)
        min_score = min(c.score for c in chunks)
        score_range = max_score - min_score if max_score > min_score else 1
        for c in chunks:
            c.score = (c.score - min_score) / score_range
        return chunks

    vec_results = normalize(vec_results)
    bm25_results = normalize(bm25_results)

    # 合并：同一 chunk 的分数加权求和
    score_map: dict[str, tuple[float, Chunk]] = {}
    for c in vec_results:
        key = f"{c.source}:{c.chunk_id}"
        score_map[key] = (alpha * c.score, c)
    for c in bm25_results:
        key = f"{c.source}:{c.chunk_id}"
        if key in score_map:
            old_score, old_chunk = score_map[key]
            score_map[key] = (old_score + (1 - alpha) * c.score, old_chunk)
        else:
            score_map[key] = ((1 - alpha) * c.score, c)

    # 构建结果
    merged = []
    for score, chunk in score_map.values():
        merged.append(Chunk(
            content=chunk.content,
            source=chunk.source,
            chunk_id=chunk.chunk_id,
            score=score,
        ))

    merged.sort(key=lambda c: c.score, reverse=True)
    merged = _deduplicate(merged)
    # 过滤阈值
    merged = [c for c in merged if c.score >= threshold]
    return merged[:top_k]


def rerank_by_keywords(chunks: list[Chunk], query: str) -> list[Chunk]:
    """轻量重排序：基于关键词覆盖率对结果重新排序

    在向量相似度基础上，额外考虑查询关键词在文档中的命中比例。
    """
    # 提取查询关键词
    keywords = set(re.findall(r'[一-鿿]{2,}|[a-zA-Z]+', query.lower()))
    if not keywords:
        return chunks

    for chunk in chunks:
        content_lower = chunk.content.lower()
        hits = sum(1 for kw in keywords if kw in content_lower)
        keyword_bonus = hits / len(keywords) * 0.15  # 最多加 15% 权重
        chunk.score = chunk.score * (1 + keyword_bonus)

    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks

# ============================================================
# 8. System Prompt 设计
# ============================================================
SYSTEM_PROMPT = """你是一个专业的校园问答助手，基于提供的参考资料回答问题。

## 回答规范
1. **准确性**：仅基于提供的参考资料回答，不编造信息
2. **简洁性**：直接回答问题，避免冗余
3. **专业性**：使用准确的术语和数据
4. **诚实性**：如果参考资料中没有相关信息，明确说明"根据现有资料无法回答"
5. **引用性**：回答中标注信息来源

## 输出格式
回答时请结构化输出，关键信息加粗标注。"""


def build_context_prompt(chunks: list[Chunk], question: str, max_tokens: int = 3000) -> str:
    """构建上下文 prompt，控制长度"""
    # 按相似度排序（已在检索时排序）
    context_parts = []
    total_chars = 0
    char_limit = max_tokens * 2  # 粗略估计：1 token ≈ 2 个中文字符

    for i, chunk in enumerate(chunks, 1):
        content = chunk.content[:CFG.chunk_size]  # 按配置截断
        part = f"【参考{i}】(来源: {chunk.source}, 相似度: {chunk.score:.4f})\n{content}"
        if total_chars + len(part) > char_limit:
            break
        context_parts.append(part)
        total_chars += len(part)

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

## 参考资料
{context}

## 用户问题
{question}

## 回答"""
    return prompt

# ============================================================
# 9. LLM 调用（带重试、超时、降级，复用客户端）
# ============================================================
_llm_client = None  # 全局复用 OpenAI 客户端

def _get_llm_client(timeout: int):
    """获取或创建 LLM 客户端（复用连接池）"""
    global _llm_client
    if _llm_client is None:
        try:
            from openai import OpenAI
            _llm_client = OpenAI(
                base_url=CFG.llm_base_url,
                api_key=CFG.llm_api_key,
                timeout=timeout,
            )
        except ImportError:
            print("  [ERROR] openai 库未安装")
            return None
    return _llm_client


def call_llm(prompt: str, stream: bool = False, timeout: int = None, max_retries: int = None) -> str:
    """调用 LLM，带重试、超时和降级处理"""
    timeout = timeout if timeout is not None else CFG.llm_timeout
    max_retries = max_retries if max_retries is not None else CFG.llm_max_retries

    for attempt in range(max_retries):
        try:
            client = _get_llm_client(timeout)
            if client is None:
                return None

            if stream:
                return _call_llm_stream(client, prompt, timeout)
            else:
                resp = client.chat.completions.create(
                    model=CFG.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=CFG.llm_max_tokens,
                    temperature=CFG.llm_temperature,
                    timeout=timeout,
                )
                if not resp.choices:
                    print("  [WARN] LLM 返回空 choices")
                    return None
                return resp.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                print(f"  [WARN] LLM 超时 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
            elif "connection" in error_msg.lower():
                print(f"  [WARN] 连接失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
            else:
                print(f"  [WARN] LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
            # 连接失败时重置客户端，下次重建
            if "connection" in error_msg.lower():
                global _llm_client
                _llm_client = None

    print("  [ERROR] LLM 调用全部失败，降级为检索结果拼接")
    return None


def _call_llm_stream(client, prompt: str, timeout: int) -> str:
    """流式调用 LLM"""
    try:
        stream = client.chat.completions.create(
            model=CFG.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=CFG.llm_max_tokens,
            temperature=CFG.llm_temperature,
            stream=True,
            timeout=timeout,
        )
        full_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print()  # 换行
        return full_response
    except Exception as e:
        print(f"\n  [WARN] 流式输出中断: {e}")
        return full_response if full_response else None

# ============================================================
# 10. 查询改写（Query Rewrite）
# ============================================================
def rewrite_query(query: str) -> list[str]:
    """查询改写：将原始查询扩展为多个变体，提高召回率

    策略：
    1. 原始查询
    2. 去掉疑问词的关键词版本
    3. 同义替换版本（针对常见校园问法）
    """
    variants = [query]

    # 策略1：提取关键词（去掉常见疑问词/停用词）
    stop_words = {"吗", "呢", "啊", "呀", "吧", "的", "了", "是", "有", "在", "和", "与",
                  "什么", "哪些", "哪个", "怎么", "如何", "为什么", "请问", "能否", "可以"}
    keywords = [w for w in query if w not in stop_words and len(w.strip()) > 0]
    keyword_query = "".join(keywords)
    if keyword_query and keyword_query != query:
        variants.append(keyword_query)

    # 策略2：常见校园问法同义替换
    synonym_map = {
        "宿舍": ["寝室", "住宿", "公寓"],
        "食堂": ["餐厅", "餐饮", "吃饭"],
        "院系": ["学院", "专业", "学部"],
        "录取": ["招生", "分数线", "投档"],
        "怎么样": ["如何", "好不好", "评价"],
        "有哪些": ["包括什么", "都有什么", "列出"],
    }
    for key, synonyms in synonym_map.items():
        if key in query:
            for syn in synonyms[:1]:  # 每个同义词只取第一个，避免太多变体
                variants.append(query.replace(key, syn))

    # 去重
    seen = set()
    unique = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


# ============================================================
# 11. RAG 主流程（已集成混合检索+重排序）
# ============================================================
class RAGEngine:
    """RAG 引擎：整合检索和生成"""

    def __init__(self):
        self.index = VectorIndex()
        self.bm25_index = BM25Index()
        self.docs: list[dict] = []
        self._ready = False

    def initialize(self):
        """初始化：加载文档 + 构建索引（向量 + BM25）"""
        print("=" * 60)
        print("步骤 1: 加载知识文档")
        print("=" * 60)
        self.docs = load_documents(CFG.knowledge_dir)
        print(f"  加载了 {len(self.docs)} 个文档")

        if not self.docs:
            print("  [ERROR] 未找到任何文档")
            return False

        print("\n" + "=" * 60)
        print("步骤 2: 文本切分 + 向量索引 + BM25索引")
        print("=" * 60)
        chunks = build_chunks(self.docs, CFG.chunk_size, CFG.chunk_overlap)
        print(f"  切分完成: {len(chunks)} 个 chunks")

        self.index.build(chunks)
        self.bm25_index.build(chunks)
        print(f"  BM25 索引构建完成")
        self._ready = True
        return True

    def retrieve(self, query: str, top_k: int = None, threshold: float = None) -> RetrievalResult:
        """检索阶段：混合检索（向量+BM25）→ 重排序"""
        top_k = top_k if top_k is not None else CFG.default_top_k
        threshold = threshold if threshold is not None else CFG.similarity_threshold

        # 检查缓存（返回深拷贝避免数据污染）
        if CFG.cache_enabled:
            cache_key = search_cache.make_key("hybrid", query, top_k, threshold)
            cached = search_cache.get(cache_key)
            if cached:
                result = copy.deepcopy(cached)
                result.from_cache = True
                print("  [CACHE] 命中缓存")
                return result

        t0 = time.time()
        # 查询改写 → 多查询混合检索 → 重排序
        query_variants = rewrite_query(query)
        if len(query_variants) > 1:
            print(f"  查询改写: {len(query_variants)} 个变体")
        all_chunks = []
        for q in query_variants:
            chunks = hybrid_search(self.index, self.bm25_index, q, top_k=top_k, threshold=threshold)
            all_chunks.extend(chunks)
        # 合并去重
        seen = {}
        for c in all_chunks:
            key = f"{c.source}:{c.chunk_id}"
            if key not in seen or c.score > seen[key].score:
                seen[key] = c
        chunks = list(seen.values())
        # 重排序
        chunks = rerank_by_keywords(chunks, query)
        chunks = chunks[:top_k]
        search_time = time.time() - t0

        result = RetrievalResult(
            chunks=chunks,
            query=query,
            total_searched=len(self.index.chunks),
            search_time=search_time,
        )

        # 存入缓存
        if CFG.cache_enabled:
            search_cache.put(cache_key, result)

        return result

    def generate(self, query: str, retrieval: RetrievalResult, stream: bool = False) -> Answer:
        """生成阶段"""
        if not retrieval.chunks:
            return Answer(
                text="抱歉，未找到与您问题相关的知识信息。请尝试换个方式提问。",
                sources=[],
                confidence=0.0,
                retrieval_result=retrieval,
            )

        # 构建 prompt
        prompt = build_context_prompt(retrieval.chunks, query, CFG.max_context_tokens)

        # 调用 LLM
        llm_answer = call_llm(prompt, stream=stream)

        if llm_answer:
            answer_text = llm_answer
            confidence = max(c.score for c in retrieval.chunks)
        else:
            # 降级：直接拼接检索结果
            answer_text = _fallback_answer(query, retrieval.chunks)
            confidence = max(c.score for c in retrieval.chunks) * 0.8

        sources = [
            {"source": c.source, "score": round(c.score, 4), "preview": c.content[:100]}
            for c in retrieval.chunks
        ]

        return Answer(
            text=answer_text,
            sources=sources,
            confidence=round(confidence, 4),
            retrieval_result=retrieval,
        )

    def query(self, question: str, top_k: int = None, stream: bool = False) -> Answer:
        """完整的 RAG 查询流程"""
        if not self._ready:
            return Answer(text="系统未初始化，请先调用 initialize()", sources=[], confidence=0.0)

        # 检索
        retrieval = self.retrieve(question, top_k=top_k)

        # 生成
        return self.generate(question, retrieval, stream=stream)


def _fallback_answer(query: str, chunks: list[Chunk]) -> str:
    """降级回答：直接拼接检索结果"""
    parts = [f"根据知识库检索，找到 {len(chunks)} 条相关信息：\n"]
    for i, c in enumerate(chunks, 1):
        preview = c.content[:200].replace("\n", " ")
        parts.append(f"【来源{i}】(相似度: {c.score:.4f}, 文件: {c.source})")
        parts.append(f"{preview}...\n")
    return "\n".join(parts)

# ============================================================
# 12. 参数对比实验
# ============================================================
def run_experiment(engine: RAGEngine, query: str, chunk_size: int, chunk_overlap: int, top_k: int):
    """运行一次对比实验（重新切分+建索引）"""
    print(f"\n{'='*60}")
    print(f"实验: chunk_size={chunk_size}, overlap={chunk_overlap}, top_k={top_k}")
    print(f"{'='*60}")

    t0 = time.time()
    chunks = build_chunks(engine.docs, chunk_size, chunk_overlap)
    t_split = time.time() - t0

    temp_index = VectorIndex()
    t0 = time.time()
    temp_index.build(chunks)
    t_index = time.time() - t0

    t0 = time.time()
    results = temp_index.search(query, top_k=top_k, threshold=CFG.similarity_threshold)
    t_search = time.time() - t0

    best_score = results[0].score if results else 0

    print(f"  切分: {len(chunks)} chunks ({t_split:.2f}s)")
    print(f"  索引: {t_index:.2f}s | 检索: {t_search:.4f}s")
    print(f"  最佳相似度: {best_score:.4f}")

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "num_chunks": len(chunks),
        "split_time": round(t_split, 2),
        "index_time": round(t_index, 2),
        "search_time": round(t_search, 4),
        "best_score": round(best_score, 4),
        "results": [
            {"source": c.source, "score": round(c.score, 4), "preview": c.content[:80]}
            for c in results
        ],
    }

# ============================================================
# 13. 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="增强版最小 RAG 脚本")
    parser.add_argument("--query", type=str, default=None, help="查询问题（不指定则交互式输入）")
    parser.add_argument("--top_k", type=int, default=None, help="检索返回数量")
    parser.add_argument("--compare", action="store_true", help="运行参数对比实验")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    parser.add_argument("--no-llm", action="store_true", help="不调用LLM，仅检索")
    parser.add_argument("--output", type=str, default="", help="输出结果到 JSON 文件")
    args = parser.parse_args()

    # 初始化引擎
    engine = RAGEngine()
    if not engine.initialize():
        sys.exit(1)

    # 获取查询
    query = args.query
    if not query:
        query = input("\n请输入查询问题: ").strip()
        if not query:
            query = "河海大学有哪些院系？"
            print(f"  未输入，使用默认问题: {query}")

    if args.compare:
        # 参数对比实验
        print("\n" + "=" * 60)
        print("参数对比实验")
        print("=" * 60)

        experiments = [
            {"chunk_size": 200, "chunk_overlap": 50, "top_k": 3},
            {"chunk_size": 500, "chunk_overlap": 50, "top_k": 3},
            {"chunk_size": 1000, "chunk_overlap": 50, "top_k": 3},
            {"chunk_size": 500, "chunk_overlap": 50, "top_k": 1},
            {"chunk_size": 500, "chunk_overlap": 50, "top_k": 5},
            {"chunk_size": 500, "chunk_overlap": 0, "top_k": 3},
            {"chunk_size": 500, "chunk_overlap": 100, "top_k": 3},
        ]

        all_results = []
        for exp in experiments:
            result = run_experiment(engine, query, **exp)
            all_results.append(result)

        # 总结表
        print(f"\n{'='*60}")
        print("参数对比总结")
        print(f"{'='*60}")
        print(f"{'chunk_size':>12} {'overlap':>8} {'top_k':>6} {'chunks':>8} {'split':>8} {'index':>8} {'search':>8} {'score':>8}")
        print("-" * 76)
        for r in all_results:
            print(f"{r['chunk_size']:>12} {r['chunk_overlap']:>8} {r['top_k']:>6} {r['num_chunks']:>8} {r['split_time']:>8} {r['index_time']:>8} {r['search_time']:>8} {r['best_score']:>8}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}")

    else:
        # 单次查询
        print("\n" + "=" * 60)
        print("RAG 查询")
        print("=" * 60)

        top_k = args.top_k if args.top_k is not None else CFG.default_top_k

        if args.no_llm:
            # 仅检索，不调用 LLM
            retrieval = engine.retrieve(query, top_k=top_k)
            search_time = retrieval.search_time if retrieval else 0
            print(f"\n检索结果 ({len(retrieval.chunks)} 条, 耗时 {search_time:.4f}s):")
            for i, chunk in enumerate(retrieval.chunks, 1):
                print(f"  [{i}] 相似度: {chunk.score:.4f} | 来源: {chunk.source}")
                print(f"      预览: {chunk.content[:100].replace(chr(10), ' ')}")
            if args.output:
                result_data = {
                    "query": query,
                    "chunks": [{"source": c.source, "score": round(c.score, 4), "preview": c.content[:100]} for c in retrieval.chunks],
                    "search_time": search_time,
                }
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                print(f"\n结果已保存到: {args.output}")
        else:
            answer = engine.query(query, top_k=top_k, stream=args.stream)

            # 输出检索结果
            search_time = answer.retrieval_result.search_time if answer.retrieval_result else 0
            print(f"\n检索结果 ({len(answer.sources)} 条, 耗时 {search_time:.4f}s):")
            for i, src in enumerate(answer.sources, 1):
                print(f"  [{i}] 相似度: {src['score']} | 来源: {src['source']}")

            # 输出回答
            if not args.stream:
                print(f"\n{'='*60}")
                print("生成回答")
                print(f"{'='*60}")
                print(answer.text)

            print(f"\n置信度: {answer.confidence}")

        if args.output:
            result_data = {
                "query": query,
                "answer": answer.text,
                "confidence": answer.confidence,
                "sources": answer.sources,
                "search_time": answer.retrieval_result.search_time,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
