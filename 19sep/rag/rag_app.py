#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Application for Product Data (Hybrid Retrieval, FAISS, Chonkie)
==================================================================
- Defaults implement the recommended decision:
  * Chunking: Recursive(320)
  * Retrieval: Hybrid (Dense + BM25, alpha=0.6)
  * Indexing: FAISS-IVF with dynamic nlist, nprobe=12
  * Optional LLM re-rank of top-M (off by default)

Commands:
  python rag_app.py build     # builds artifacts from ../layer_23/result
  python rag_app.py ask "best budget earbuds"
  python rag_app.py eval      # synthetic evaluation (no LLM calls)

Env:
  OPENAI_API_KEY=...
  (optional) OPENAI_EMBED_MODEL, OPENAI_CHAT_MODEL

Author: you + hippo power 🦛  (2025-09-19)
"""

from __future__ import annotations
import os, re, json, glob, time, math, hashlib, pickle, random, sys, argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional deps
try:
    import tiktoken
    _enc = None
    for name in ("o200k_base", "cl100k_base"):
        try:
            _enc = tiktoken.get_encoding(name); break
        except Exception:
            continue
    def count_tokens(s: str) -> int:
        if _enc is None: return max(1, len(s)//4)
        return len(_enc.encode(s))
except Exception:
    def count_tokens(s: str) -> int:
        return max(1, len(s)//4)

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    # OpenAI (>=1.0 style)
    from openai import OpenAI
except Exception as e:
    print("[ERROR] openai package missing. pip install openai")
    raise

# Chonkie
try:
    from chonkie import RecursiveChunker
except Exception:
    print("[ERROR] chonkie package missing. pip install chonkie")
    raise

# ------------------------ CONFIG ------------------------
CONFIG = {
    # Data root (expects */products.json)
    "root": "../layer_23/result",
    # Artifacts directory
    "art_dir": "artifacts",

    # Models
    "embed_model": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
    "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),

    # Chunking
    # options: recursive | sentence | semantic
    "chunker": os.getenv("CHUNKER", "recursive"),
    "chunk_size_tokens": 320,
    "min_chars_per_chunk": 24,
    "sent_overlap": 1,
    "semantic_threshold": 0.72,   # only if semantic chunking enabled

    # Indexing (vector)
    # options: numpy | faiss_flat | faiss_ivf | faiss_hnsw
    "index_backend": os.getenv("INDEX_BACKEND", "faiss_ivf"),

    # Retrieval knobs
    "dense_topN": 200,            # vector candidates
    "bm25_topN": 200,             # lexical candidates
    "final_topK": 5,              # contexts for answer
    "hybrid_alpha": 0.6,          # blend: alpha*dense + (1-alpha)*bm25

    # Re-rank (optional: cost $$$) – off by default
    "use_llm_rerank": bool(int(os.getenv("USE_LLM_RERANK", "0"))),
    "rerank_topM": 24,

    # Embedding
    "embed_batch_size": 512,
    "embed_cache_path": ".emb_cache.pkl",

    # FAISS IVF/HNSW defaults
    "ivf_nprobe": 12,
    "hnsw_m": 32,
    "hnsw_ef_search": 64,

    # Answer generation
    "answer_max_tokens": 320,

    # Reproducibility
    "random_seed": 42,
}
random.seed(CONFIG["random_seed"]); np.random.seed(CONFIG["random_seed"])

# ------------------------ Data model ------------------------
@dataclass
class ProductDoc:
    doc_id: str
    category: str
    title: Optional[str]
    url: Optional[str]
    price_value: Optional[float]
    price_display: Optional[str]
    rating_avg: Optional[float]
    rating_count: Optional[int]
    sold_count: Optional[int]
    brand: Optional[str]
    seller_name: Optional[str]
    colors: Optional[List[str]]
    description: Optional[str]
    source_path: str
    raw: Dict[str, Any]

_number_re = re.compile(r"(\d[\d,\.]*)")

def _first(*vals):
    for v in vals:
        if isinstance(v, str) and v.strip(): return v.strip()
        if v not in (None, "", [], {}): return v
    return None

def _to_https(u: Optional[str]) -> Optional[str]:
    if not u: return None
    u = u.strip()
    if u.startswith("//"): return "https:" + u
    if u.startswith("http"): return u
    if u.startswith("/"): return "https://www.daraz.com.bd" + u
    if u.startswith("www."): return "https://" + u
    return u

def _parse_number(s: Optional[str]) -> Optional[float]:
    if not s: return None
    m = _number_re.search(str(s))
    if not m: return None
    try: return float(m.group(1).replace(",", ""))
    except Exception: return None

def _parse_int(s: Optional[str]) -> Optional[int]:
    v = _parse_number(s)
    return int(v) if v is not None else None

def normalize_product(prod: Dict[str, Any], category: str, source_path: str) -> ProductDoc:
    raw_id = _first(prod.get("data_item_id"), prod.get("data_sku_simple"))
    if not raw_id:
        base = (_first(prod.get("product_title"), prod.get("detail", {}).get("name", "")) or "") + "|" + (_first(prod.get("product_detail_url"), prod.get("detail_url"), prod.get("detail", {}).get("url", "")) or "")
        raw_id = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

    title = _first(prod.get("product_title"), prod.get("detail", {}).get("name"))
    url = _to_https(_first(prod.get("detail_url"), prod.get("product_detail_url"), prod.get("detail", {}).get("url")))
    brand = _first(prod.get("detail", {}).get("brand"), prod.get("brand"))

    price_display = _first(prod.get("detail", {}).get("price", {}).get("display"), prod.get("product_price"))
    price_value = _first(prod.get("detail", {}).get("price", {}).get("value"), _parse_number(price_display))

    rating_avg = _first(prod.get("detail", {}).get("rating", {}).get("average"))
    rating_count = _first(prod.get("detail", {}).get("rating", {}).get("count"))
    if isinstance(rating_count, str):
        rating_count = _parse_int(rating_count)

    sold_count = _parse_int(prod.get("location"))
    seller_name = _first(prod.get("detail", {}).get("seller", {}).get("name"))
    colors = prod.get("detail", {}).get("colors", [])
    if not isinstance(colors, list):
        colors = [str(colors)] if colors is not None else []

    desc = _first(
        prod.get("detail", {}).get("details", {}).get("description_text"),
        prod.get("detail", {}).get("details", {}).get("raw_text"),
    )

    return ProductDoc(
        doc_id=str(raw_id), category=category, title=str(title) if title else None,
        url=url, price_value=float(price_value) if price_value is not None else None,
        price_display=str(price_display) if price_display else None,
        rating_avg=float(rating_avg) if rating_avg is not None else None,
        rating_count=int(rating_count) if rating_count is not None else None,
        sold_count=int(sold_count) if sold_count is not None else None,
        brand=str(brand) if brand else None, seller_name=str(seller_name) if seller_name else None,
        colors=[str(c) for c in colors] if colors else [], description=str(desc) if desc else None,
        source_path=source_path, raw=prod,
    )

def iter_product_docs(root_dir: str) -> List[ProductDoc]:
    docs: List[ProductDoc] = []
    for products_json in glob.glob(os.path.join(root_dir, "*", "products.json")):
        category = os.path.basename(os.path.dirname(products_json))
        try:
            with open(products_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list): continue
        except Exception as e:
            print(f"[WARN] Skipping {products_json}: {e}"); continue
        for prod in data:
            if not isinstance(prod, dict): continue
            docs.append(normalize_product(prod, category, products_json))
    return docs

def product_text(doc: ProductDoc) -> str:
    parts = [f"PRODUCT_ID: {doc.doc_id}", f"CATEGORY: {doc.category}"]
    if doc.title: parts.append(f"TITLE: {doc.title}")
    if doc.brand: parts.append(f"BRAND: {doc.brand}")
    if doc.price_display: parts.append(f"PRICE: {doc.price_display}")
    elif doc.price_value is not None: parts.append(f"PRICE_VALUE: {doc.price_value}")
    if doc.rating_avg is not None: parts.append(f"RATING_AVG: {doc.rating_avg:.2f}")
    if doc.rating_count is not None: parts.append(f"RATING_COUNT: {doc.rating_count}")
    if doc.sold_count is not None: parts.append(f"SOLD: {doc.sold_count}")
    if doc.seller_name: parts.append(f"SELLER: {doc.seller_name}")
    if doc.colors: parts.append("COLORS: " + ", ".join(doc.colors))
    if doc.description: parts.append(f"DESCRIPTION: {doc.description}")
    if doc.url: parts.append(f"URL: {doc.url}")
    return "\n".join(parts)

# ------------------------ Chunkers ------------------------
class Chunker:
    def chunk(self, text: str) -> List[str]:
        raise NotImplementedError

class RecursiveChunkerWrapper(Chunker):
    def __init__(self, size_tokens: int, min_chars: int):
        self._c = RecursiveChunker(
            tokenizer_or_token_counter=count_tokens,
            chunk_size=size_tokens,
            min_characters_per_chunk=min_chars
        )
    def chunk(self, text: str) -> List[str]:
        return [c.text.strip() for c in self._c(text) if c.text and c.text.strip()]

class SentenceChunker(Chunker):
    def __init__(self, max_tokens: int = 320, overlap_sentences: int = 0):
        self.max_tokens = max_tokens
        self.overlap = overlap_sentences
    def _split_sentences(self, s: str) -> List[str]:
        parts = re.split(r"(?<=[.!?\n])\s+", s)
        return [p.strip() for p in parts if p.strip()]
    def chunk(self, text: str) -> List[str]:
        sents = self._split_sentences(text)
        chunks, cur, cur_tok = [], [], 0
        for sent in sents:
            t = count_tokens(sent)
            if cur and cur_tok + t > self.max_tokens:
                chunks.append(" ".join(cur).strip())
                if self.overlap > 0:
                    cur = cur[-self.overlap:]; cur_tok = sum(count_tokens(x) for x in cur)
                else:
                    cur, cur_tok = [], 0
            cur.append(sent); cur_tok += t
        if cur: chunks.append(" ".join(cur).strip())
        return chunks

class SemanticChunker(Chunker):
    def __init__(self, client: OpenAI, embed_model: str, max_tokens: int = 320, threshold: float = 0.72):
        self.client, self.model, self.max_tokens, self.threshold = client, embed_model, max_tokens, threshold
    def _split_sentences(self, s: str) -> List[str]:
        parts = re.split(r"(?<=[.!?\n])\s+", s)
        return [p.strip() for p in parts if p.strip()]
    def _embed(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=[t.replace("\n"," ") for t in texts], encoding_format="float")
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True); norms[norms==0.0]=1.0
        return arr/norms
    def chunk(self, text: str) -> List[str]:
        sents = self._split_sentences(text)
        if not sents: return []
        embs = self._embed(sents)
        chunks: List[str] = []; cur: List[str] = [sents[0]]; cur_tok = count_tokens(sents[0])
        for i in range(1, len(sents)):
            sim = float(embs[i] @ embs[i-1])
            need_new = sim < self.threshold or (cur_tok + count_tokens(sents[i]) > self.max_tokens)
            if need_new:
                chunks.append(" ".join(cur).strip()); cur, cur_tok = [sents[i]], count_tokens(sents[i])
            else:
                cur.append(sents[i]); cur_tok += count_tokens(sents[i])
        if cur: chunks.append(" ".join(cur).strip())
        return chunks

# ------------------------ Embeddings + cache ------------------------
class EmbeddingCache:
    def __init__(self, path: str):
        self.path = path
        try:
            with open(self.path, "rb") as f:
                self._store = pickle.load(f)
        except Exception:
            self._store = {}
    def get(self, key: str):
        return self._store.get(key)
    def set(self, key: str, vec: List[float]):
        self._store[key] = vec
    def save(self):
        try:
            with open(self.path, "wb") as f:
                pickle.dump(self._store, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[WARN] Could not save cache: {e}")

def _sha1_for_text_model(text: str, model: str) -> str:
    h = hashlib.sha1(); h.update(model.encode("utf-8")); h.update(b"\x00"); h.update(text.encode("utf-8"))
    return h.hexdigest()

def embed_texts(client: OpenAI, texts: List[str], model: str, batch_size: int, cache: Optional[EmbeddingCache]=None) -> np.ndarray:
    all_vecs: List[Optional[List[float]]] = [None] * len(texts)
    to_embed_idx: List[int] = []
    keys: List[Optional[str]] = [None] * len(texts)

    if cache:
        for i, t in enumerate(texts):
            key = _sha1_for_text_model(t.replace("\n", " "), model)
            keys[i] = key
            vec = cache.get(key)
            if vec is None: to_embed_idx.append(i)
            else: all_vecs[i] = vec
    else:
        to_embed_idx = list(range(len(texts)))

    for start in range(0, len(to_embed_idx), batch_size):
        batch_ids = to_embed_idx[start:start+batch_size]; 
        if not batch_ids: break
        batch = [texts[i].replace("\n"," ") for i in batch_ids]
        resp = client.embeddings.create(model=model, input=batch, encoding_format="float")
        for local_j, i in enumerate(batch_ids):
            vec = resp.data[local_j].embedding
            all_vecs[i] = vec
            if cache and keys[i]: cache.set(keys[i], vec)
    if cache: cache.save()

    arr = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True); norms[norms==0.0]=1.0
    return arr/norms

# ------------------------ Indexes ------------------------
class DenseIndex:
    def build(self, vecs: np.ndarray): raise NotImplementedError
    def search(self, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]: raise NotImplementedError

class NumpyFlat(DenseIndex):
    def __init__(self): self.vecs = None
    def build(self, vecs: np.ndarray): self.vecs = vecs.astype(np.float32, copy=False)
    def search(self, query_vec: np.ndarray, k: int):
        scores = self.vecs @ query_vec
        k = min(k, len(scores))
        if k <= 0: return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        idx = np.argpartition(-scores, k-1)[:k]; idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]

class FaissFlat(DenseIndex):
    def __init__(self):
        if faiss is None: raise RuntimeError("faiss not available")
        self.index = None
    def build(self, vecs: np.ndarray):
        d = vecs.shape[1]; index = faiss.IndexFlatIP(d); index.add(vecs); self.index = index
    def search(self, query_vec: np.ndarray, k: int):
        D, I = self.index.search(query_vec.reshape(1,-1).astype(np.float32), k)
        return I[0], D[0]

class FaissIVF(DenseIndex):
    def __init__(self, nlist: int = 64, nprobe: int = 12):
        if faiss is None: raise RuntimeError("faiss not available")
        self.nlist, self.nprobe, self.index = nlist, nprobe, None
    def build(self, vecs: np.ndarray):
        d = vecs.shape[1]
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vecs)
        index.add(vecs)
        index.nprobe = self.nprobe
        self.index = index
    def search(self, query_vec: np.ndarray, k: int):
        D, I = self.index.search(query_vec.reshape(1,-1).astype(np.float32), k)
        return I[0], D[0]

class FaissHNSW(DenseIndex):
    def __init__(self, hnsw_m: int = 32, ef_search: int = 64):
        if faiss is None: raise RuntimeError("faiss not available")
        self.hnsw_m, self.ef_search, self.index = hnsw_m, ef_search, None
    def build(self, vecs: np.ndarray):
        d = vecs.shape[1]
        index = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT) if hasattr(faiss, "METRIC_INNER_PRODUCT") else faiss.IndexHNSWFlat(d, self.hnsw_m)
        # efSearch tuning
        try: index.hnsw.efSearch = self.ef_search
        except Exception: pass
        index.add(vecs)
        self.index = index
    def search(self, query_vec: np.ndarray, k: int):
        D, I = self.index.search(query_vec.reshape(1,-1).astype(np.float32), k)
        return I[0], D[0]

INDEX_REGISTRY = {
    "numpy": NumpyFlat,
    "faiss_flat": FaissFlat,
    "faiss_ivf": FaissIVF,
    "faiss_hnsw": FaissHNSW,
}

# ------------------------ Build pipeline ------------------------
def choose_chunker(client: Optional[OpenAI]) -> Chunker:
    C = CONFIG
    if C["chunker"] == "recursive":
        return RecursiveChunkerWrapper(C["chunk_size_tokens"], C["min_chars_per_chunk"])
    elif C["chunker"] == "sentence":
        return SentenceChunker(C["chunk_size_tokens"], C["sent_overlap"])
    elif C["chunker"] == "semantic":
        if client is None: client = OpenAI()
        return SemanticChunker(client, C["embed_model"], C["chunk_size_tokens"], C["semantic_threshold"])
    else:
        raise ValueError("Unknown chunker: " + str(C["chunker"]))

def build_chunks(docs: List[ProductDoc], chunker: Chunker) -> Tuple[List[str], List[Dict[str, Any]]]:
    texts, meta = [], []
    for d in docs:
        base = product_text(d)
        parts = chunker.chunk(base)
        for i, ch in enumerate(parts):
            clean = ch.replace("\u0000", " ").strip()
            if not clean: continue
            texts.append(clean)
            meta.append({"doc_id": d.doc_id, "title": d.title, "category": d.category, "url": d.url, "chunk_idx": i})
    return texts, meta

def dynamic_ivf_nlist(num_chunks: int) -> int:
    # keep trainable with small corpora; avoid warnings
    return max(32, min(128, num_chunks // 40))  # ~25 chunks/centroid min

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def build_artifacts():
    t0 = time.time()
    C = CONFIG; client = OpenAI()
    print(f"[INFO] Loading products from: {C['root']}")
    docs = iter_product_docs(C["root"])
    if not docs:
        print("[ERROR] No products found under", C["root"]); sys.exit(1)
    print(f"[OK] Loaded {len(docs)} docs")

    chunker = choose_chunker(client if C["chunker"]=="semantic" else None)
    chunks_texts, chunks_meta = build_chunks(docs, chunker)
    print(f"[OK] {C['chunker']} -> {len(chunks_texts)} chunks (avg {len(chunks_texts)/len(docs):.2f}/doc)")

    cache = EmbeddingCache(C["embed_cache_path"])
    t = time.time()
    chunk_vecs = embed_texts(client, chunks_texts, C["embed_model"], batch_size=C["embed_batch_size"], cache=cache)
    print(f"[OK] Embedded chunks: {chunk_vecs.shape} in {time.time()-t:.2f}s")

    # Build vector index
    backend = C["index_backend"]
    if backend == "faiss_ivf":
        nlist = dynamic_ivf_nlist(len(chunks_texts))
        index = FaissIVF(nlist=nlist, nprobe=C["ivf_nprobe"])
    elif backend == "faiss_flat":
        index = FaissFlat()
    elif backend == "faiss_hnsw":
        index = FaissHNSW(hnsw_m=C["hnsw_m"], ef_search=C["hnsw_ef_search"])
    elif backend == "numpy":
        index = NumpyFlat()
    else:
        raise ValueError("Unknown index_backend: " + str(backend))

    t = time.time(); index.build(chunk_vecs); build_time = time.time() - t
    print(f"[OK] Built {backend} in {build_time:.2f}s")

    # Build BM25 over the *full* chunk corpus (no vector gating)
    bm25_data = None
    if BM25Okapi is not None:
        def tok(s: str) -> List[str]:
            s = s.lower(); s = re.sub(r"[^a-z0-9]+", " ", s); return [w for w in s.split() if w]
        bm25 = BM25Okapi([tok(t) for t in chunks_texts])
        bm25_data = {"ok": True}  # marker so we know BM25 exists
        print("[OK] BM25 index ready")
    else:
        print("[WARN] rank_bm25 not installed; BM25/Hybrid disabled at query time")

    # Persist artifacts
    ensure_dir(C["art_dir"])
    np.save(os.path.join(C["art_dir"], "emb.npy"), chunk_vecs)
    with open(os.path.join(C["art_dir"], "chunks.jsonl"), "w", encoding="utf-8") as f:
        for t in chunks_texts: f.write(t.replace("\n", " ") + "\n")
    save_json(os.path.join(C["art_dir"], "meta.json"), chunks_meta)
    save_json(os.path.join(C["art_dir"], "build_info.json"), {
        "chunker": C["chunker"], "chunk_count": len(chunks_texts), "backend": backend,
        "embed_model": C["embed_model"], "ivf_nlist": dynamic_ivf_nlist(len(chunks_texts)) if backend=="faiss_ivf" else None,
        "ivf_nprobe": C["ivf_nprobe"] if backend=="faiss_ivf" else None,
        "hnsw_m": C["hnsw_m"] if backend=="faiss_hnsw" else None,
        "hnsw_ef_search": C["hnsw_ef_search"] if backend=="faiss_hnsw" else None,
        "built_at": time.time()
    })

    # Save FAISS index file when applicable
    if backend.startswith("faiss") and faiss is not None:
        try:
            faiss_path = os.path.join(C["art_dir"], "faiss.index")
            faiss.write_index(index.index, faiss_path)
            print(f"[OK] Saved FAISS index -> {faiss_path}")
        except Exception as e:
            print(f"[WARN] Could not save FAISS index: {e}")

    # Save a tiny BM25 marker for presence (actual object rebuilt on load)
    if bm25_data:
        save_json(os.path.join(C["art_dir"], "bm25.json"), bm25_data)

    print(f"[DONE] Build in {time.time()-t0:.2f}s")

# ------------------------ Retrieval + Answering ------------------------
def load_artifacts():
    C = CONFIG
    chunks_path = os.path.join(C["art_dir"], "chunks.jsonl")
    meta_path = os.path.join(C["art_dir"], "meta.json")
    emb_path = os.path.join(C["art_dir"], "emb.npy")
    info_path = os.path.join(C["art_dir"], "build_info.json")

    if not (os.path.exists(chunks_path) and os.path.exists(meta_path) and os.path.exists(emb_path) and os.path.exists(info_path)):
        print("[INFO] Artifacts missing; building now...")
        build_artifacts()

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_texts = [line.rstrip("\n") for line in f]
    chunks_meta = load_json(meta_path)
    chunk_vecs = np.load(emb_path)
    info = load_json(info_path)

    # Recreate vector index
    backend = CONFIG["index_backend"]
    if backend == "faiss_ivf":
        nlist = info.get("ivf_nlist") or dynamic_ivf_nlist(len(chunks_texts))
        index = FaissIVF(nlist=nlist, nprobe=CONFIG["ivf_nprobe"])
    elif backend == "faiss_flat":
        index = FaissFlat()
    elif backend == "faiss_hnsw":
        index = FaissHNSW(hnsw_m=CONFIG["hnsw_m"], ef_search=CONFIG["hnsw_ef_search"])
    elif backend == "numpy":
        index = NumpyFlat()
    else:
        raise ValueError("Unknown index_backend: " + str(backend))

    # Try to load FAISS index from disk; else build in-memory
    if backend.startswith("faiss") and faiss is not None:
        faiss_path = os.path.join(CONFIG["art_dir"], "faiss.index")
        try:
            index.index = faiss.read_index(faiss_path)
            print(f"[OK] Loaded FAISS index from {faiss_path}")
        except Exception:
            print("[INFO] Could not read FAISS index file; rebuilding in memory")
            index.build(chunk_vecs)
    else:
        index.build(chunk_vecs)

    # Build BM25 on load if available
    bm25 = None
    if BM25Okapi is not None and os.path.exists(os.path.join(CONFIG["art_dir"], "bm25.json")):
        def tok(s: str) -> List[str]:
            s = s.lower(); s = re.sub(r"[^a-z0-9]+", " ", s); return [w for w in s.split() if w]
        bm25 = BM25Okapi([tok(t) for t in chunks_texts])
        print("[OK] BM25 ready")

    return chunks_texts, chunks_meta, chunk_vecs, index, bm25

def normalize_scores(x: np.ndarray) -> np.ndarray:
    if len(x)==0: return x
    if not np.isfinite(x).any(): return np.zeros_like(x)
    v = x.copy()
    # guard against all equal values
    m, M = float(np.min(v)), float(np.max(v))
    if not np.isfinite(m) or not np.isfinite(M) or abs(M-m) < 1e-12:
        return np.zeros_like(v)
    return (v - m) / (M - m + 1e-12)

def retrieve_hybrid(query: str, client: OpenAI, chunks_texts: List[str], chunk_vecs: np.ndarray, index: DenseIndex, bm25: Optional[BM25Okapi], alpha: float, dense_topN: int, bm25_topN: int) -> Tuple[List[int], List[float]]:
    # Dense candidates from index
    qv = embed_texts(client, [query], CONFIG["embed_model"], batch_size=1)[0]
    I, D = index.search(qv, k=min(dense_topN, len(chunks_texts)))
    dense_idx = list(map(int, I.tolist()))
    dense_scores = D.astype(np.float32)

    # BM25 candidates from full corpus (no vector gating)
    bm25_idx: List[int] = []
    bm25_scores_np = np.zeros(0, dtype=np.float32)
    if bm25 is not None:
        def tok(s: str) -> List[str]:
            s = s.lower(); s = re.sub(r"[^a-z0-9]+", " ", s); return [w for w in s.split() if w]
        scores_full = bm25.get_scores(tok(query))
        scores_full = np.asarray(scores_full, dtype=np.float32)
        topN = min(bm25_topN, len(scores_full))
        if topN > 0:
            order = np.argpartition(-scores_full, topN-1)[:topN]
            order = order[np.argsort(-scores_full[order])]
            bm25_idx = order.astype(int).tolist()
            bm25_scores_np = scores_full[order]

    # Union of candidates
    union = list(dict.fromkeys(dense_idx + bm25_idx))
    # Compute dense scores for union (exact dot with qv)
    dense_for_union = (chunk_vecs[union] @ qv).astype(np.float32)
    dense_norm = normalize_scores(dense_for_union)

    # BM25 scores for union (0 for non-BM25 candidates)
    bm25_for_union = np.zeros(len(union), dtype=np.float32)
    if bm25 is not None and len(bm25_idx) > 0:
        bm25_map = {int(i): float(s) for i, s in zip(bm25_idx, bm25_scores_np)}
        for j, gi in enumerate(union):
            bm25_for_union[j] = bm25_map.get(int(gi), 0.0)
    bm25_norm = normalize_scores(bm25_for_union)

    # Hybrid score
    hybrid = alpha*dense_norm + (1.0 - alpha)*bm25_norm
    order = np.argsort(-hybrid)
    ranked_idx = [union[i] for i in order]
    ranked_scores = hybrid[order].tolist()
    return ranked_idx, ranked_scores

def llm_rerank(query: str, client: OpenAI, texts: List[str], top_m: int) -> List[int]:
    """Return new order of indices [0..len(texts)-1] after LLM scoring."""
    bullets = "\n\n".join([f"[CANDIDATE {i}]\n{texts[i][:900]}" for i in range(len(texts))])
    prompt = (
        "Rate relevance (0-100) for each candidate to the Query. "
        "Return ONLY a JSON array of indices ordered from most to least relevant.\n\n"
        f"Query: {query}\n\n{bullets}\n\nJSON:"
    )
    try:
        resp = client.chat.completions.create(
            model=CONFIG["chat_model"], temperature=0,
            messages=[{"role":"user","content":prompt}],
            max_tokens=256,
        )
        import json as _json, re as _re
        txt = resp.choices[0].message.content.strip()
        arr = _json.loads(_re.findall(r"\[[^\]]*\]", txt)[0])
        arr = [i for i in arr if isinstance(i, int) and 0 <= i < len(texts)]
        if not arr: return list(range(min(top_m, len(texts))))
        return arr[:top_m]
    except Exception:
        return list(range(min(top_m, len(texts))))

def synthesize_answer(query: str, client: OpenAI, contexts: List[Tuple[str, Dict[str, Any]]]) -> str:
    """contexts: list of (chunk_text, meta)"""
    # Deduplicate by doc_id to avoid repeating the same page
    seen = set()
    unique_ctx = []
    for t, m in contexts:
        did = m.get("doc_id")
        if did in seen: continue
        seen.add(did); unique_ctx.append((t, m))
        if len(unique_ctx) >= CONFIG["final_topK"]: break

    context_block = "\n\n---\n\n".join(
        f"[{i+1}] {ctx}\nSOURCE: {m.get('title') or m.get('url') or m.get('doc_id')}"
        for i, (ctx, m) in enumerate(unique_ctx)
    )
    sys_prompt = (
        "You are a precise shopping assistant. Use only the context to answer. "
        "If unsure, say you don't know. Include short bullet points and name the matched products/titles when possible."
    )
    user_prompt = f"Question: {query}\n\nContext:\n{context_block}"

    try:
        resp = client.chat.completions.create(
            model=CONFIG["chat_model"],
            temperature=0.2,
            messages=[
                {"role":"system", "content": sys_prompt},
                {"role":"user", "content": user_prompt},
            ],
            max_tokens=CONFIG["answer_max_tokens"]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM answer unavailable: {e})"

def ask(query: str):
    t0 = time.time()
    chunks_texts, chunks_meta, chunk_vecs, index, bm25 = load_artifacts()
    client = OpenAI()

    ranked_idx, ranked_scores = retrieve_hybrid(
        query=query, client=client,
        chunks_texts=chunks_texts, chunk_vecs=chunk_vecs,
        index=index, bm25=bm25,
        alpha=CONFIG["hybrid_alpha"],
        dense_topN=CONFIG["dense_topN"],
        bm25_topN=CONFIG["bm25_topN"],
    )

    # optional LLM re-rank
    if CONFIG["use_llm_rerank"]:
        topM = min(CONFIG["rerank_topM"], len(ranked_idx))
        sub_texts = [chunks_texts[i] for i in ranked_idx[:topM]]
        order_local = llm_rerank(query, OpenAI(), sub_texts, top_m=topM)
        ranked_idx = [ranked_idx[i] for i in order_local] + ranked_idx[topM:]

    # Gather top contexts
    topK = CONFIG["final_topK"]
    contexts = [(chunks_texts[i], chunks_meta[i]) for i in ranked_idx[:max(topK*3, topK)]]
    answer = synthesize_answer(query, client, contexts)

    # Pretty print
    print("\n=== Answer ===")
    print(answer)
    print("\n=== Top Sources ===")
    shown = set()
    for i in ranked_idx[:max(10, topK*2)]:
        m = chunks_meta[i]; did = m.get("doc_id")
        if did in shown: continue
        shown.add(did)
        title = m.get("title") or "(untitled)"
        url = m.get("url") or "(no url)"
        print(f"- {title}  <-- {url}")
        if len(shown) >= topK: break
    print(f"\n[done in {time.time()-t0:.2f}s]")

# ------------------------ Evaluation (synthetic) ------------------------
def synthesize_queries(docs: List[ProductDoc], per_doc: int = 1) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for d in docs:
        cands = []
        if d.title: cands.append(d.title)
        if d.brand and d.category: cands.append(f"{d.brand} {d.category}")
        if d.category: cands.append(f"best {d.category}")
        # unique
        seen = set(); cands = [x for x in cands if x and not (x in seen or seen.add(x))]
        random.shuffle(cands)
        for q in cands[:per_doc]:
            out.append((q, d.doc_id))
    random.shuffle(out)
    return out

def hit_at_k(ranks: List[Optional[int]], k: int) -> float:
    return float(np.mean([1.0 if (r is not None and r < k) else 0.0 for r in ranks]))

def mrr_at_k(ranks: List[Optional[int]], k: int) -> float:
    vals = []
    for r in ranks:
        if r is not None and r < k: vals.append(1.0 / (r + 1))
        else: vals.append(0.0)
    return float(np.mean(vals))

def run_eval(max_docs: int = 1000, queries_per_doc: int = 1):
    t0 = time.time()
    # Load full docs directly (we recompute queries from docs)
    docs = iter_product_docs(CONFIG["root"])
    if max_docs: docs = docs[:max_docs]
    qpairs = synthesize_queries(docs, per_doc=queries_per_doc)
    if not qpairs:
        print("[ERROR] No queries synthesized"); return

    chunks_texts, chunks_meta, chunk_vecs, index, bm25 = load_artifacts()
    client = OpenAI()
    K = CONFIG["final_topK"]

    ranks: List[Optional[int]] = []
    times: List[float] = []

    for q, target_id in qpairs:
        t = time.time()
        ranked_idx, _ = retrieve_hybrid(
            q, client, chunks_texts, chunk_vecs, index, bm25,
            alpha=CONFIG["hybrid_alpha"],
            dense_topN=CONFIG["dense_topN"],
            bm25_topN=CONFIG["bm25_topN"],
        )
        # rank of first chunk matching the doc
        rank = None
        for rpos, ci in enumerate(ranked_idx[:K]):
            if chunks_meta[ci]["doc_id"] == target_id:
                rank = rpos; break
        ranks.append(rank)
        times.append(time.time()-t)

    Hk = hit_at_k(ranks, K); MRRk = mrr_at_k(ranks, K)
    print("\n==== EVAL SUMMARY ====")
    print(f"Queries: {len(qpairs)}  K={K}")
    print(f"hit@K={Hk:.4f}  mrr@K={MRRk:.4f}")
    print(f"avg_query_ms={int(1000*np.mean(times))}  p95_ms={int(1000*np.percentile(times,95))}")
    print(f"[done in {time.time()-t0:.2f}s]")

# ------------------------ CLI ------------------------
def main():
    ap = argparse.ArgumentParser(description="RAG app (Hybrid + FAISS + Chonkie)")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("build", help="Build artifacts (chunks, embeddings, indexes)")
    ask_p = sub.add_parser("ask", help="Ask a question")
    ask_p.add_argument("query", type=str, help="User query")
    sub.add_parser("eval", help="Synthetic evaluation (no LLM)")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help(); sys.exit(0)

    if args.cmd == "build":
        build_artifacts()
    elif args.cmd == "ask":
        ask(args.query)
    elif args.cmd == "eval":
        run_eval()
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
