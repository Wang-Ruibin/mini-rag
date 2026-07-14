#!/usr/bin/env python3
"""
最小可运行 RAG 脚本
功能：加载知识文档 → 文本切分 → 向量化 → 检索 → 生成回答
支持参数对比：chunk_size / overlap / top_k
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# ============================================================
# 1. 配置
# ============================================================
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "code", "knowledge_docs")
DEFAULT_QUERY = "河海大学有哪些院系？"

# ============================================================
# 2. 文档加载
# ============================================================
def load_documents(doc_dir: str) -> list[dict]:
    """递归加载目录下所有 .md 文件"""
    docs = []
    doc_path = Path(doc_dir)
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
# 3. 文本切分
# ============================================================
def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """简易文本切分器：按字符数切分，支持重叠"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - chunk_overlap
    return chunks


def build_chunks(docs: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """将文档切分为 chunks"""
    all_chunks = []
    for doc in docs:
        text_chunks = split_text(doc["content"], chunk_size, chunk_overlap)
        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "content": chunk,
                "source": doc["source"],
                "chunk_id": i,
            })
    return all_chunks


# ============================================================
# 4. 向量索引（使用 sentence-transformers + FAISS）
# ============================================================
def build_index(chunks: list[dict], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """构建 FAISS 向量索引"""
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    print(f"  加载 Embedding 模型: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["content"] for c in chunks]
    print(f"  编码 {len(texts)} 个 chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积相似度（已归一化 = 余弦相似度）
    index.add(embeddings)

    return index, model


# ============================================================
# 5. 检索
# ============================================================
def search(query: str, index, model, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """检索最相关的 chunks"""
    import numpy as np

    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            results.append({
                **chunks[idx],
                "score": float(score),
            })
    return results


# ============================================================
# 6. 回答生成（基于检索结果拼接）
# ============================================================
def generate_answer(query: str, results: list[dict], use_llm: bool = False) -> str:
    """基于检索结果生成回答"""
    context = "\n\n---\n\n".join([r["content"][:300] for r in results])

    if use_llm:
        # 尝试使用 OpenAI 兼容 API（如 Ollama、百炼等）
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder"),
            )
            prompt = f"""基于以下参考文档内容回答用户问题。如果文档中没有相关信息，请说明。

参考文档：
{context}

用户问题：{query}

回答："""
            resp = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.3,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  [WARN] LLM 调用失败 ({e})，使用基于文档的拼接回答")

    # 无 LLM 时，直接展示检索到的相关内容作为回答
    answer_parts = []
    answer_parts.append(f"根据知识库检索，找到 {len(results)} 条与「{query}」相关的信息：\n")
    for i, r in enumerate(results, 1):
        answer_parts.append(f"【来源 {i}】(相似度: {r['score']:.4f}, 文件: {r['source']})")
        # 截取前200字展示
        content_preview = r["content"][:200].replace("\n", " ")
        answer_parts.append(f"{content_preview}...\n")

    return "\n".join(answer_parts)


# ============================================================
# 7. 参数对比实验
# ============================================================
def run_experiment(docs: list[dict], query: str, chunk_size: int, chunk_overlap: int, top_k: int):
    """运行一次 RAG 实验"""
    print(f"\n{'='*60}")
    print(f"实验参数: chunk_size={chunk_size}, overlap={chunk_overlap}, top_k={top_k}")
    print(f"{'='*60}")

    # 切分
    t0 = time.time()
    chunks = build_chunks(docs, chunk_size, chunk_overlap)
    t_split = time.time() - t0
    print(f"  切分完成: {len(chunks)} 个 chunks, 耗时 {t_split:.2f}s")

    # 构建索引
    t0 = time.time()
    index, model = build_index(chunks)
    t_index = time.time() - t0
    print(f"  索引构建完成, 耗时 {t_index:.2f}s")

    # 检索
    t0 = time.time()
    results = search(query, index, model, chunks, top_k)
    t_search = time.time() - t0
    print(f"  检索完成, 耗时 {t_search:.4f}s")

    # 生成回答
    answer = generate_answer(query, results)

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "num_chunks": len(chunks),
        "split_time": round(t_split, 2),
        "index_time": round(t_index, 2),
        "search_time": round(t_search, 4),
        "results": results,
        "answer": answer,
    }


# ============================================================
# 8. 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="最小 RAG 脚本")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="查询问题")
    parser.add_argument("--chunk_size", type=int, default=500, help="文本切分块大小")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="切分重叠长度")
    parser.add_argument("--top_k", type=int, default=3, help="检索返回数量")
    parser.add_argument("--compare", action="store_true", help="运行参数对比实验")
    parser.add_argument("--use_llm", action="store_true", help="使用 LLM 生成回答")
    parser.add_argument("--output", type=str, default="", help="输出结果到 JSON 文件")
    args = parser.parse_args()

    # 加载文档
    print("=" * 60)
    print("步骤 1: 加载知识文档")
    print("=" * 60)
    docs = load_documents(KNOWLEDGE_DIR)
    print(f"  加载了 {len(docs)} 个文档")

    if not docs:
        print("  [ERROR] 未找到任何文档，请检查路径:", KNOWLEDGE_DIR)
        sys.exit(1)

    if args.compare:
        # 参数对比实验
        print("\n" + "=" * 60)
        print("参数对比实验")
        print("=" * 60)

        experiments = [
            # chunk_size 对比
            {"chunk_size": 200, "chunk_overlap": 50, "top_k": 3},
            {"chunk_size": 500, "chunk_overlap": 50, "top_k": 3},
            {"chunk_size": 1000, "chunk_overlap": 50, "top_k": 3},
            # top_k 对比
            {"chunk_size": 500, "chunk_overlap": 50, "top_k": 1},
            {"chunk_size": 500, "chunk_overlap": 50, "top_k": 5},
            # overlap 对比
            {"chunk_size": 500, "chunk_overlap": 0, "top_k": 3},
            {"chunk_size": 500, "chunk_overlap": 100, "top_k": 3},
        ]

        all_results = []
        for exp in experiments:
            result = run_experiment(docs, args.query, **exp)
            all_results.append(result)

            # 打印回答预览
            print(f"\n  回答预览: {result['answer'][:150]}...")
            print(f"  最高相似度: {result['results'][0]['score']:.4f}" if result['results'] else "")

        # 输出对比总结
        print("\n" + "=" * 60)
        print("参数对比总结")
        print("=" * 60)
        print(f"{'chunk_size':>12} {'overlap':>8} {'top_k':>6} {'chunks':>8} {'split(s)':>10} {'search(s)':>10} {'best_score':>12}")
        print("-" * 70)
        for r in all_results:
            best_score = r['results'][0]['score'] if r['results'] else 0
            print(f"{r['chunk_size']:>12} {r['chunk_overlap']:>8} {r['top_k']:>6} {r['num_chunks']:>8} {r['split_time']:>10} {r['search_time']:>10.4f} {best_score:>12.4f}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}")

    else:
        # 单次运行
        result = run_experiment(docs, args.query, args.chunk_size, args.chunk_overlap, args.top_k)

        print("\n" + "=" * 60)
        print("检索结果")
        print("=" * 60)
        for i, r in enumerate(result["results"], 1):
            print(f"\n[{i}] 相似度: {r['score']:.4f} | 来源: {r['source']}")
            print(f"    {r['content'][:100]}...")

        print("\n" + "=" * 60)
        print("生成回答")
        print("=" * 60)
        print(result["answer"])

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
