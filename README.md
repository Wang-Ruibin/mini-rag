<div align="center">

# 🧠 Mini RAG

**A minimal, production-ready Retrieval-Augmented Generation system in a single Python file.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)](https://github.com/facebookresearch/faiss)

*Document Loading → Smart Chunking → Hybrid Retrieval → LLM Generation*

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Configuration](#-configuration) • [Usage](#-usage)

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
python minimal_rag.py --query "你的问题"
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

### Pipeline

1. **Load** — Recursively scan `.md` files from `knowledge_docs/`
2. **Chunk** — Paragraph-first → sentence boundary → configurable overlap
3. **Index** — FAISS `IndexFlatIP` (cosine similarity) + BM25 sparse index
4. **Retrieve** — Query rewrite → hybrid search (vector + BM25) → rerank → dedup
5. **Generate** — System prompt + context → LLM (with retry & fallback)

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
python minimal_rag.py --query "河海大学有哪些院系？"

# Interactive mode (default)
python minimal_rag.py

# Parameter comparison experiment
python minimal_rag.py --query "宿舍条件怎么样" --compare

# Streaming output
python minimal_rag.py --query "食堂有哪些" --stream

# Retrieval only (no LLM)
python minimal_rag.py --query "招生政策" --no-llm

# Custom top_k
python minimal_rag.py --query "图书馆" --top_k 10

# Save results to JSON
python minimal_rag.py --query "宿舍" --output results.json
```

## 📊 Parameter Comparison

Run `--compare` to automatically test different chunk_size / overlap / top_k combinations:

| chunk_size | overlap | top_k | chunks | best_score |
|-----------|---------|-------|--------|------------|
| 200 | 50 | 3 | ~4000 | 0.72 |
| 500 | 50 | 3 | ~1600 | 0.70 |
| 1000 | 50 | 3 | ~800 | 0.68 |
| 500 | 0 | 3 | ~1600 | 0.65 |
| 500 | 100 | 3 | ~1600 | 0.71 |

## 🛠️ Tech Stack

- **Embeddings**: [sentence-transformers](https://github.com/UKPLab/sentence-transformers) (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Vector DB**: [FAISS](https://github.com/facebookresearch/faiss) (`IndexFlatIP`)
- **Sparse Retrieval**: Custom BM25 implementation
- **LLM**: Any OpenAI-compatible API ([Ollama](https://ollama.com), [vLLM](https://github.com/vllm-project/vllm), etc.)
- **Python**: 3.10+

## 📁 Project Structure

```
mini-rag/
├── minimal_rag.py      # Complete RAG system (~850 lines)
├── .env.example        # Configuration template
├── .gitignore          # Git ignore rules
├── LICENSE             # MIT License
└── README.md           # This file
```

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork this repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**If you find this useful, please give it a ⭐ — it means a lot!**

*Made with ❤️ and way too much coffee ☕*

</div>
