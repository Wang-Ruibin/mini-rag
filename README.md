<div align="center">

# 🧠 Mini RAG

**一个最小的、生产级的 RAG 检索增强生成系统，全部代码在一个 Python 文件中。**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)](https://github.com/facebookresearch/faiss)

*文档加载 → 智能切分 → 混合检索 → LLM 生成*

</div>

---

## ✨ 特性

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

## 🚀 快速开始

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

## 🏗️ 系统架构

```
知识文档加载 → 智能文本切分（段落优先+重叠） → FAISS + BM25 索引构建
                                                      ↓
用户查询 → 查询改写（同义替换+关键词提取） → 混合检索（向量+BM25）
                                                      ↓
LLM 生成回答 ← 重排序（关键词覆盖率） ← TopK 选择
```

### 流程详解

1. **文档加载** — 递归扫描 `knowledge_docs/` 下所有 `.md` 文件
2. **智能切分** — 段落优先 → 句子边界 → 可配置重叠
3. **索引构建** — FAISS `IndexFlatIP`（余弦相似度）+ BM25 稀疏索引
4. **检索阶段** — 查询改写 → 混合检索（向量+BM25 加权融合）→ 重排序 → 去重
5. **生成阶段** — System Prompt + 上下文 → LLM（带重试与降级）

## ⚙️ 配置说明

所有参数通过 `.env` 文件配置：

```env
# LLM 配置（OpenAI 兼容接口，如 Ollama、MiMo）
OPENAI_BASE_URL=https://api.mimo.xiaomi.com/v1
OPENAI_API_KEY=your-api-key
LLM_MODEL=mimo-v2.5-pro

# Embedding 模型
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# 检索参数
DEFAULT_TOP_K=5
SIMILARITY_THRESHOLD=0.3

# 切分参数
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# LLM 参数
LLM_TIMEOUT=30
LLM_MAX_RETRIES=3
LLM_MAX_TOKENS=1024
LLM_TEMPERATURE=0.3

# 上下文与缓存
MAX_CONTEXT_TOKENS=3000
CACHE_ENABLED=true
```

## 📖 使用方式

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

## 🛠️ 技术栈

| 组件 | 选型 |
|------|------|
| Embedding 模型 | `paraphrase-multilingual-MiniLM-L12-v2`（384 维） |
| 向量检索 | FAISS `IndexFlatIP`（归一化内积 = 余弦相似度） |
| 稀疏检索 | 自实现 BM25（k1=1.5, b=0.75） |
| 混合策略 | α 加权融合（α=0.7 向量 + 0.3 BM25） |
| LLM | 任意 OpenAI 兼容 API（Ollama、MiMo、vLLM 等） |
| Python | 3.10+ |

## 📁 项目结构

```
mini-rag/
├── minimal_rag.py      # 完整 RAG 系统（约 900 行）
├── .env.example        # 配置模板
├── .gitignore          # Git 忽略规则
├── LICENSE             # MIT 许可证
├── README.md           # 本文件
└── knowledge_docs/     # 知识库（.md 文件）
```

## 📊 参数对比实验

运行 `--compare` 可自动测试不同参数组合的效果：

| chunk_size | overlap | top_k | chunks 数量 | 最佳相似度 |
|-----------|---------|-------|------------|-----------|
| 200 | 50 | 3 | 4056 | 0.7992 |
| 500 | 50 | 3 | 1661 | 0.7700 |
| 1000 | 50 | 3 | 878 | 0.7640 |
| 500 | 0 | 3 | 1661 | 0.7783 |
| 500 | 100 | 3 | 1661 | 0.7700 |

## 🤝 参与贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE)。

---

<div align="center">

**如果觉得有用，点个 ⭐ 吧！**

*用 ❤️ 和太多咖啡 ☕ 制作*

</div>
