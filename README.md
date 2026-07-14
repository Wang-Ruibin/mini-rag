<div align="center">

# 🧠 Mini RAG

**A minimal, production-ready Retrieval-Augmented Generation system in a single Python file.**

**一个最小的、生产级的 RAG 检索增强生成系统，全部代码在一个 Python 文件中。**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)](https://github.com/facebookresearch/faiss)

*Document Loading → Smart Chunking → Hybrid Retrieval → LLM Generation*

[English](#-features) • [中文](#-特性)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔀 **Hybrid Retrieval** | Dense vectors (FAISS) + sparse keywords (BM25) with weighted fusion |
| 🔍 **Query Rewrite** | Automatic synonym expansion & keyword extraction for better recall |
| 📊 **Smart Reranking** | Keyword-coverage bonus on top of similarity scores |
| ✂️ **Intelligent Chunking** | Paragraph-first splitting with sentence boundary detection & overlap |
| ⚡ **LRU Cache** | Thread-safe cache with TTL expiry for repeated queries |
| 🔄 **Retry & Fallback** | Exponential backoff retries + graceful degradation when LLM is down |
| 🌊 **Streaming Output** | Real-time token-by-token streaming via `--stream` flag |
| ⚙️ **Env-based Config** | All parameters via `.env` — zero hardcoded values |

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Wang-Ruibin/mini-rag.git
cd mini-rag

# 2. Install dependencies
pip install faiss-cpu sentence-transformers openai

# 3. Configure
cp .env.example .env
# Edit .env to set your LLM endpoint (Ollama / OpenAI-compatible)

# 4. Add knowledge docs
mkdir -p knowledge_docs
# Place your .md files in knowledge_docs/

# 5. Run
python minimal_rag.py --query "your question"
```

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Document    │───▶│  Smart Text   │───▶│  Vector Index   │
│  Loading     │    │  Chunking     │    │  (FAISS + BM25) │
└─────────────┘    └──────────────┘    └────────┬────────┘
                                                │
                   ┌──────────────┐    ┌────────▼────────┐
                   │  Query       │───▶│  Hybrid Search  │
                   │  Rewrite     │    │  (α-weighted)   │
                   └──────────────┘    └────────┬────────┘
                                                │
                   ┌──────────────┐    ┌────────▼────────┐
                   │  LLM         │◀───│  Reranking +    │
                   │  Generation   │    │  TopK Selection │
                   └──────────────┘    └─────────────────┘
```

## ⚙️ Configuration

All parameters are configurable via `.env`:

```env
# LLM (OpenAI-compatible endpoint, e.g. Ollama)
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=sk-placeholder
LLM_MODEL=qwen2.5:7b

# Embedding
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# Retrieval
DEFAULT_TOP_K=5
SIMILARITY_THRESHOLD=0.3

# Chunking
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Performance
LLM_TIMEOUT=30
LLM_MAX_RETRIES=3
CACHE_ENABLED=true
```

## 📖 Usage

```bash
# Single query
python minimal_rag.py --query "your question"

# Interactive mode
python minimal_rag.py

# Parameter comparison experiment
python minimal_rag.py --query "your question" --compare

# Streaming output
python minimal_rag.py --query "your question" --stream

# Retrieval only (no LLM)
python minimal_rag.py --query "your question" --no-llm

# Custom top_k
python minimal_rag.py --query "your question" --top_k 10

# Save results to JSON
python minimal_rag.py --query "your question" --output results.json
```

## 🛠️ Tech Stack

- **Embeddings**: [sentence-transformers](https://github.com/UKPLab/sentence-transformers) (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Vector DB**: [FAISS](https://github.com/facebookresearch/faiss) (`IndexFlatIP`)
- **Sparse Retrieval**: Custom BM25 implementation
- **LLM**: Any OpenAI-compatible API ([Ollama](https://ollama.com), [vLLM](https://github.com/vllm-project/vllm), etc.)
- **Python**: 3.10+

## 📁 Project Structure

```
mini-rag/
├── minimal_rag.py      # Complete RAG system (~900 lines)
├── .env.example        # Configuration template
├── .gitignore          # Git ignore rules
├── LICENSE             # MIT License
├── README.md           # This file
└── knowledge_docs/     # Knowledge base (.md files)
```

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**If you find this useful, please give it a ⭐ — it means a lot!**

*Made with ❤️ and way too much coffee ☕*

</div>

---

<a id="-特性"></a>

## 🇨🇳 中文说明

### 特性

| 功能 | 说明 |
|------|------|
| 🔀 **混合检索** | 向量检索 (FAISS) + 关键词检索 (BM25)，加权融合 |
| 🔍 **查询改写** | 自动同义替换 + 关键词提取，提高召回率 |
| 📊 **智能重排** | 在相似度基础上叠加关键词覆盖率权重 |
| ✂️ **智能切分** | 段落优先 + 句子边界检测 + 重叠切分 |
| ⚡ **LRU 缓存** | 线程安全，支持 TTL 过期 |
| 🔄 **重试与降级** | 指数退避重试 + LLM 不可用时自动降级 |
| 🌊 **流式输出** | `--stream` 实时逐 token 输出 |
| ⚙️ **环境变量配置** | 所有参数通过 `.env` 配置，零硬编码 |

### 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/Wang-Ruibin/mini-rag.git
cd mini-rag

# 2. 安装依赖
pip install faiss-cpu sentence-transformers openai

# 3. 配置 LLM
cp .env.example .env
# 编辑 .env，填写你的 LLM 接口地址和 API Key

# 4. 添加知识文档
mkdir -p knowledge_docs
# 将你的 .md 文件放入 knowledge_docs/ 目录

# 5. 运行
python minimal_rag.py --query "你的问题"
```

### 系统架构

```
知识文档加载 → 智能文本切分（段落优先+重叠） → FAISS + BM25 索引构建
                                                      ↓
用户查询 → 查询改写（同义替换+关键词提取） → 混合检索（向量+BM25）
                                                      ↓
LLM 生成回答 ← 重排序（关键词覆盖率） ← TopK 选择
```

### 使用方式

```bash
# 单次查询
python minimal_rag.py --query "河海大学有哪些院系？"

# 交互式输入问题
python minimal_rag.py

# 参数对比实验（对比不同 chunk_size/overlap/top_k）
python minimal_rag.py --query "问题" --compare

# 流式输出
python minimal_rag.py --query "问题" --stream

# 仅检索，不调用 LLM
python minimal_rag.py --query "问题" --no-llm

# 指定 top_k
python minimal_rag.py --query "问题" --top_k 10

# 结果保存为 JSON
python minimal_rag.py --query "问题" --output results.json
```

### 技术栈

| 组件 | 选型 |
|------|------|
| Embedding 模型 | `paraphrase-multilingual-MiniLM-L12-v2`（384 维） |
| 向量检索 | FAISS `IndexFlatIP`（归一化内积 = 余弦相似度） |
| 稀疏检索 | 自实现 BM25（k1=1.5, b=0.75） |
| 混合策略 | α 加权融合（α=0.7 向量 + 0.3 BM25） |
| LLM | 任意 OpenAI 兼容 API（Ollama、MiMo、vLLM 等） |

### 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE)。

---

<div align="center">

**如果觉得有用，点个 ⭐ 吧！**

</div>
