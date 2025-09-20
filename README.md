# RAG Experiment Report — Chunking, Retrieval, and Indexing (Chonkie + FAISS + BM25), For Daraz

> **Taks 2**  
> *“Experiment with different chunking strategies using Chonkie (e.g., semantic-based, sentence-based).  
> Evaluate various retrieval methods (such as reranking and hybrid approaches) to determine which performs best.  
> Test multiple vector database indexing algorithms for efficiency and accuracy.”*

---

## 🌟 Executive Summary

- **Best quality (MRR@5 = 0.2052):** `Recursive` chunking + **BM25** retrieval.  
- **Fastest near‑par variant (≈276 ms avg):** `Sentence` chunking + **BM25** with **IVF** index (dense index didn’t affect BM25; the time difference is dominated by our per‑query pipeline).  
- **Hybrid (Dense+BM25) & LLM re‑rank:** **did not improve quality** on this dataset; added latency.  
- **Methodology fix for production:** In the experiment BM25 re‑ranked a **dense‑prefiltered shortlist**. In production we should run **BM25 over the full corpus** (un‑gated), and use late‑fusion (**RRF**) only when enabling hybrid.

**Final decision for the app:**
- **Default:** `Recursive` + **BM25 (full corpus)** for best accuracy.  
- **Optional:** `--mode hybrid` = Dense (FAISS **HNSW**) ∪ BM25 with **RRF** late fusion.  
- **LLM rerank:** off by default; toggle with `--rerank` for special queries.

---
```
## 📁 Repository Layout



.
├─ 20sep/rag
│  ├─ rag\_analysis.py        # our experiment harness (provided)
│  ├─ rag\_app.py           # final application (production path)
│  └─ store/                 # persisted chunks/embeddings/indexes (generated)
├─ layer\_23/
│  └─ result/\*/products.json # dataset (per-category)
└─ README.md                 # this report

````

---

## 📦 Environment & Setup

**Python:** 3.9+ recommended

```bash
pip install chonkie rank-bm25 faiss-cpu tiktoken python-dotenv openai reportlab
# reportlab is optional; used only for PDF exports
````

**Environment variables:**

```bash
export OPENAI_API_KEY=sk-...
# Optional:
export OPENAI_EMBED_MODEL=text-embedding-3-small
export OPENAI_CHAT_MODEL=gpt-3.5-turbo
```

---

## 🧰 Data & Preprocessing

* **Dataset root:** `../layer_23/result`
* **File pattern:** `<category>/products.json` (array of product dicts)
* **Normalization:** Each product is normalized into a `ProductDoc` with fields like `title`, `brand`, `price`, `rating`, `description`, `url`, etc.
* **Document text template:** structured lines (ID, CATEGORY, TITLE, BRAND, PRICE, RATING, DESCRIPTION, URL) to keep lexical signals strong.

---

## 🧪 Experiment Design

### Chunking strategies (Chonkie)

* **Recursive** (Chonkie’s tokenizer‑aware splitter)
* **Sentence** (simple regex sentence splitter with optional overlap)
* **Semantic** *(available but off by default due to cost; uses embeddings to segment by sentence similarity threshold)*

### Retrieval variants

* **Dense** (dot‑product on OpenAI embeddings; FAISS/Numpy backends)
* **BM25** (lexical; rank-bm25)
* **Hybrid** (dense+bm25 score blending in the harness; **production uses RRF**)
* **LLM rerank** (optional second‑stage rerank over a short list)

> **Note:** In the experiment harness, BM25 re‑scores **only a vector‑index shortlist**. This favors dense retrieval and can mask BM25’s full strength. The final app corrects this by running **BM25 over the entire corpus** and uses **RRF** for hybrid.

### Vector index backends (for dense)

* `numpy` (flat)
* `faiss_flat` (exact IP)
* `faiss_ivf` (IVF‑Flat; approximate)
* `faiss_hnsw` (graph‑based ANN)

### Metrics

* **Hit\@K** (K=5)
* **MRR\@K** (K=5)
* **Latency:** avg and p95 per query
* Additional notes: embedding time per chunker and index build time.

### Query generation

* **Synthetic queries** per doc from: `title`, `brand + category`, `best <category>` (1 per doc, capped at 100 total).
* This biases toward **lexical** matching (brand/title/category tokens), which helps explain BM25’s advantage.

---

## 📊 Results (from our run)

**Top findings:**

* **Best MRR\@5:** `Recursive + BM25` (0.2052) with `faiss_hnsw` or `faiss_flat` present in the pipeline.
* **Fastest good performer:** `Sentence + BM25` with `faiss_ivf` (\~276 ms avg, MRR\@5≈0.2012).
* **Hybrid & LLM rerank:** did not beat BM25 on this data; added latency.

### Summary (best first by MRR\@5; ties broken by avg latency)

| chunker   | index       | retriever       | K | hit\@K | MRR\@K     | avg\_ms | p95\_ms | build\_s | #chunks | avg/doc |
| --------- | ----------- | --------------- | - | ------ | ---------- | ------- | ------- | -------- | ------- | ------- |
| recursive | faiss\_hnsw | **bm25**        | 5 | 0.29   | **0.2052** | 538     | 998     | 0.90     | 2267    | 4.53    |
| recursive | faiss\_flat | **bm25**        | 5 | 0.29   | **0.2052** | 687     | 1119    | 0.00     | 2267    | 4.53    |
| sentence  | faiss\_ivf  | **bm25**        | 5 | 0.26   | 0.2012     | **276** | **269** | 3.29     | 1790    | 3.58    |
| recursive | faiss\_ivf  | **bm25**        | 5 | 0.28   | 0.2005     | 605     | 1400    | 0.55     | 2267    | 4.53    |
|recursive| faiss_ivf|  hybrid_rerank| 5| 0.25|  0.1982| 1247|         5691|         0.55|          2267|       4.53|              
|recursive| faiss_hnsw| hybrid|        5| 0.25|  0.1982| 1262|         6220|         0.9|           2267|       4.53|              
|recursive| faiss_ivf|  hybrid|        5| 0.25|  0.1982| 1269|         6131|         0.55|          2267|       4.53|              
|recursive| faiss_flat| hybrid_rerank| 5| 0.25|  0.1982| 1306|         6722|         0.0|           2267|       4.53|              
|recursive| faiss_hnsw| hybrid_rerank| 5| 0.25|  0.1982| 1333|         5089|         0.9|           2267|       4.53|              
|recursive| faiss_flat| hybrid|        5| 0.25|  0.1982| 1371|         6763|         0.0|           2267|       4.53|              
|sentence|  faiss_hnsw| bm25|          5| 0.25|  0.1912| 322|          431|          1.16|          1790|       3.58|              
|sentence|  faiss_flat| bm25|          5| 0.25|  0.1912| 444|          1039|         0.12|          1790|       3.58|              
|sentence|  faiss_ivf|  hybrid_rerank| 5| 0.27|  0.1873| 539|          704|          3.29|          1790|       3.58|              
|sentence|  faiss_ivf|  hybrid|        5| 0.27|  0.1873| 687|          2684|         3.29|          1790|       3.58|              
|sentence|  faiss_hnsw| hybrid|        5| 0.25|  0.1823| 567|          1264|         1.16|          1790|       3.58|              
|sentence|  faiss_flat| hybrid_rerank| 5| 0.25|  0.1823| 617|          2194|         0.12|          1790|       3.58|              
|sentence|  faiss_hnsw| hybrid_rerank| 5| 0.25|  0.1823| 627|          2883|         1.16|          1790|       3.58|              
|sentence|  faiss_flat| hybrid|        5| 0.25|  0.1823| 739|          2735|         0.12|          1790|       3.58|

> IVF warnings in our logs were due to **nlist=256 with only \~2k vectors**. In the final app we adapt `nlist` to dataset size to avoid this.

---

## 🧭 Decision & Rationale

1. **Default production path:**
   **Recursive chunking + BM25 (full corpus)** → best quality; predictable latency; robust to lexical queries that appear in our data (brand/title/category).

2. **Optional hybrid path (toggle):**
   **BM25 ∪ Dense with RRF fusion** (late fusion). This is useful if real user queries become more semantic; RRF is robust and easy to toggle.

3. **LLM rerank:**
   Keep **off** by default; enable per query if needed. In our experiments it didn’t help, likely because the queries are lexically anchored.

---

## 🧱 Final Application

File: **`20sep/rag/rag_app.py`** (provided earlier)

### Key architecture

* **Chunking:** `--chunker recursive` (default) or `--chunker sentence`
* **Retrieval modes:**

  * `--mode bm25` (default): BM25 **over full corpus**
  * `--mode hybrid`: Dense (FAISS **HNSW** by default) ∪ BM25 using **RRF** late fusion
* **Dense index options:** `--dense-index hnsw|ivf|flat|numpy`
* **LLM rerank:** `--rerank` (reranks only the top‑M candidates)

### Persistence & caching

* Chunks, metadata, embeddings, and FAISS index are saved in `./store`
* Embedding cache is SHA1(model+text) keyed to reduce API calls

### Citations

* Answer synthesis uses only retrieved chunks and cites as `[DOC#CHUNK]`.

---

## ▶️ How to Run

**Build & single query:**

```bash
python 20sep/rag/rag_app.py --root ../layer_23/result --query "best budget phone" --top-k 5
```

**Hybrid with dense recall + RRF:**

```bash
python 20sep/rag/rag_app.py --mode hybrid --query "wireless earbuds for running"
```

**Speed profile (sentence chunking):**

```bash
python 20sep/rag/rag_app.py --chunker sentence --mode bm25 --query "gaming laptop under 70k"
```

**Force rebuild caches & indexes:**

```bash
python 20sep/rag/rag_app.py --build
```

**Optional LLM re‑rank:**

```bash
python 20sep/rag/rag_app.py --mode hybrid --rerank --query "noise cancelling headphones under 5k"
```

---

## 🔁 Reproducing the Experiments

**Our harness:** `20sep/rag/rag_analysis.py`

* It enumerates chunkers (`sentence`, `recursive`), dense indexes (`faiss_flat`, `faiss_ivf`, `faiss_hnsw`), and retrievers (`bm25`, `hybrid`, `hybrid_rerank`).
* **Important limitation:** BM25 re‑scores only the **dense shortlist**, so BM25’s full strength is artificially constrained. Treat those numbers as *lower bounds* for BM25.

**Run:**

```bash
python 20sep/rag/rag_analysis.py
```

> For a fairer A/B with production, consider adding a **BM25‑full‑corpus** branch in the harness (no vector gating) and a **Hybrid‑RRF** branch for apples‑to‑apples comparison with the final app.

---

## 🧩 Implementation Notes & Gotchas

* **IVF on small datasets:** choose `nlist ≈ sqrt(N)` and cap `nprobe ≤ nlist` to avoid “please provide at least … training points” warnings. The final app adapts this.
* **Token counting:** uses `tiktoken` if available; heuristic fallback otherwise.
* **Synthetic queries:** favor lexical features → BM25 shines. Expect hybrid to help more when queries are conversational or paraphrased.
* **Exporting products:** set `export_products=True`; TXT always; PDF if `reportlab` present.
* **OpenAI usage:** embedding model defaults to `text-embedding-3-small`; chat model defaults to `gpt-3.5-turbo`. You can switch via env vars.

---

## 📈 What Improved in the Final App vs. Harness

* **BM25 over full corpus** (not shortlist‑gated by vectors).
* **Hybrid via RRF** (robust late fusion) instead of ad‑hoc alpha‑blends.
* **Adaptive IVF** parameters (avoid training warnings at small N).
* **Clean toggles** for chunker, mode, dense index, rerank; persisted artifacts for fast startup.
* **Inline citations** and structured prompts for answer quality.

---

## 🔮 Future Work

1. **Add a corrected evaluation mode** to `rag_app.py` to benchmark:

   * BM25 (full corpus) vs Hybrid (RRF) vs LLM rerank (on/off),
   * Both chunkers,
   * Both speed and quality (MRR\@K, nDCG\@K).
2. **Query expansion / rewriting** (e.g., LLM‑augmented queries) to test hybrid gains on semantic phrasing.
3. **Diversity/novelty controls** (MMR/dedup) to reduce near‑duplicate chunks.
4. **Adding Product Variation** Boost by using product variation data also.
5. **Ground‑truth curation** with real user queries to reduce synthetic‑query bias.


---

## 📝 Appendix: Metric Definitions

* **Hit\@K:** `1` if any correct document appears within top‑K; else `0`. Averaged across queries.
* **MRR\@K:** If correct document at rank `r < K`, score = `1/(r+1)`; else `0`. Averaged across queries.
* **Latency:** average and 95th percentile across queries (end‑to‑end retrieval path).

---