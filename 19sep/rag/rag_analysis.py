#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: you + hippo power 🦛 (2025-09-19)
"""

from __future__ import annotations
import os, re, json, glob, time, math, hashlib, pickle, random, sys, gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Optional deps (used only if present)
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None
try:
    import faiss  # type: ignore
except Exception:
    faiss = None
try:
    import tiktoken
    _enc = None
    for name in ("o200k_base", "cl100k_base"):
        try:
            _enc = tiktoken.get_encoding(name)
            break
        except Exception:
            continue
    def count_tokens(s: str) -> int:
        if _enc is None:
            return max(1, len(s) // 4)
        return len(_enc.encode(s))
except Exception:
    def count_tokens(s: str) -> int:
        return max(1, len(s) // 4)

# Optional PDF deps
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.units import inch
    from reportlab.lib.utils import simpleSplit
except Exception:
    pdf_canvas = None

# OpenAI SDK
from openai import OpenAI

# Chonkie
from chonkie import RecursiveChunker

# ========================= CONFIG (edit me) =========================
CONFIG = {
    # Paths
    "root": "../layer_23/result",   # folder with */products.json

    # Models (low-cost defaults)
    "embed_model": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
    "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),   # only used if llm_rerank=True

    # Chunking strategies to evaluate
    # Choose among: "recursive", "sentence", "semantic"
    "chunkers": ["recursive", "sentence",],   # keep semantic off by default (turn on to test)

    # Retrieval methods to evaluate
    # Options: "dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"
    "retrievers": ["bm25", "hybrid", "hybrid_rerank"],

    # Vector index backends
    # Options always safe: "numpy"
    # Extra (if FAISS available): "faiss_flat", "faiss_ivf", "faiss_hnsw"
    "indexes": ["faiss_flat", "faiss_ivf", "faiss_hnsw"],

    # Sampling + costs
    "max_docs": 5000,              # limit dataset size
    "queries_per_doc": 1,         # synth queries per doc (low cost)
    "max_queries": 100,           # safety cap across all queries
    "K": 5,                       # metrics@K

    # Embedding batch + cache
    "embed_batch_size": 512,
    "embed_cache_path": ".emb_cache.pkl",

    # Chunk sizes
    "chunk_size_tokens": 320,
    "min_chars_per_chunk": 24,

    # Sentence chunker
    "sent_overlap": 1,

    # Semantic chunker (costly: embeds sentences)
    "semantic_threshold": 0.72,

    # Hybrid retriever blend
    "hybrid_alpha": 0.5,          # alpha*dense + (1-alpha)*bm25

    # Re-ranking (kept OFF to lower cost)
    "use_llm_rerank": False,      # set True to enable
    "rerank_top_m": 5,

    # Reproducibility
    "random_seed": 42,

    # ===== Exports =====
    "export_products": True,        # write products_used.txt (+ PDF if possible)
    "export_dir": "./exports",      # where to save the files
}
# ===================================================================

random.seed(CONFIG["random_seed"]) ; np.random.seed(CONFIG["random_seed"])

# ========================= Data model =========================
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
        if isinstance(v, str) and v.strip():
            return v.strip()
        if v not in (None, "", [], {}):
            return v
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


def iter_product_docs(root_dir: str, max_docs: Optional[int]=None) -> List[ProductDoc]:
    docs: List[ProductDoc] = []
    for products_json in glob.glob(os.path.join(root_dir, "*", "products.json")):
        category = os.path.basename(os.path.dirname(products_json))
        try:
            with open(products_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
        except Exception as e:
            print(f"[WARN] Skipping {products_json}: {e}")
            continue
        for prod in data:
            if not isinstance(prod, dict):
                continue
            docs.append(normalize_product(prod, category, products_json))
            if max_docs and len(docs) >= max_docs:
                return docs
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

# ========================= Exports =========================
def export_products_txt(docs: List[ProductDoc], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, d in enumerate(docs, 1):
            f.write(f"### PRODUCT {i}\n")
            f.write(product_text(d))
            f.write("\n\n" + ("-"*80) + "\n\n")
    print(f"[OK] Wrote products TXT: {path}")

def export_products_pdf(docs: List[ProductDoc], path: str):
    if pdf_canvas is None:
        print(f"[WARN] reportlab not installed; skipping PDF export for {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = pdf_canvas.Canvas(path, pagesize=A4)
    width, height = A4
    left_margin = 0.75 * inch
    right_margin = 0.75 * inch
    top_margin = 0.75 * inch
    bottom_margin = 0.75 * inch
    usable_width = width - left_margin - right_margin
    y = height - top_margin

    c.setFont("Helvetica", 10)

    def draw_wrapped(text: str, header: Optional[str] = None):
        nonlocal y
        if header:
            c.setFont("Helvetica-Bold", 12)
            header_lines = simpleSplit(header, "Helvetica-Bold", 12, usable_width)
            for line in header_lines:
                if y < bottom_margin + 12:
                    c.showPage(); c.setFont("Helvetica", 10); y = height - top_margin
                c.drawString(left_margin, y, line)
                y -= 14
            c.setFont("Helvetica", 10)
            y -= 4

        lines = simpleSplit(text, "Helvetica", 10, usable_width)
        for line in lines:
            if y < bottom_margin + 12:
                c.showPage(); c.setFont("Helvetica", 10); y = height - top_margin
            c.drawString(left_margin, y, line)
            y -= 12

        if y < bottom_margin + 12:
            c.showPage(); c.setFont("Helvetica", 10); y = height - top_margin
        c.drawString(left_margin, y, "-" * 50)
        y -= 16

    for i, d in enumerate(docs, 1):
        draw_wrapped(product_text(d), header=f"PRODUCT {i}")

    c.save()
    print(f"[OK] Wrote products PDF: {path}")

def export_products(docs: List[ProductDoc], out_dir: str, stem: str = "products_used"):
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"{stem}.txt")
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    export_products_txt(docs, txt_path)
    export_products_pdf(docs, pdf_path)

# ========================= Chunkers =========================
class Chunker:
    def chunk(self, text: str) -> List[str]:
        raise NotImplementedError

class RecursiveChunkerWrapper(Chunker):
    def __init__(self, size_tokens: int, min_chars: int):
        self._c = RecursiveChunker(tokenizer_or_token_counter=count_tokens,
                                   chunk_size=size_tokens,
                                   min_characters_per_chunk=min_chars)
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
                    cur = cur[-self.overlap:]
                    cur_tok = sum(count_tokens(x) for x in cur)
                else:
                    cur, cur_tok = [], 0
            cur.append(sent)
            cur_tok += t
        if cur:
            chunks.append(" ".join(cur).strip())
        return chunks

class SemanticChunker(Chunker):
    def __init__(self, client: OpenAI, embed_model: str, max_tokens: int = 320, threshold: float = 0.75):
        self.client = client
        self.model = embed_model
        self.max_tokens = max_tokens
        self.threshold = threshold
    def _split_sentences(self, s: str) -> List[str]:
        parts = re.split(r"(?<=[.!?\n])\s+", s)
        return [p.strip() for p in parts if p.strip()]
    def _embed(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=[t.replace("\n"," ") for t in texts], encoding_format="float")
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms
    def chunk(self, text: str) -> List[str]:
        sents = self._split_sentences(text)
        if not sents:
            return []
        embs = self._embed(sents)
        chunks: List[str] = []
        cur: List[str] = [sents[0]]
        cur_tok: int = count_tokens(sents[0])
        for i in range(1, len(sents)):
            sim = float(embs[i] @ embs[i-1])
            need_new = sim < self.threshold or (cur_tok + count_tokens(sents[i]) > self.max_tokens)
            if need_new:
                chunks.append(" ".join(cur).strip())
                cur, cur_tok = [sents[i]], count_tokens(sents[i])
            else:
                cur.append(sents[i])
                cur_tok += count_tokens(sents[i])
        if cur:
            chunks.append(" ".join(cur).strip())
        return chunks

# ========================= Embeddings + cache =========================
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
            if vec is None:
                to_embed_idx.append(i)
            else:
                all_vecs[i] = vec
    else:
        to_embed_idx = list(range(len(texts)))

    for start in range(0, len(to_embed_idx), batch_size):
        batch_ids = to_embed_idx[start:start+batch_size]
        if not batch_ids:
            break
        batch = [texts[i].replace("\n", " ") for i in batch_ids]
        resp = client.embeddings.create(model=model, input=batch, encoding_format="float")
        for local_j, i in enumerate(batch_ids):
            vec = resp.data[local_j].embedding
            all_vecs[i] = vec
            if cache and keys[i]:
                cache.set(keys[i], vec)
    if cache:
        cache.save()

    arr = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms

# ========================= Indexes =========================
class DenseIndex:
    def build(self, vecs: np.ndarray):
        raise NotImplementedError
    def search(self, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class NumpyFlat(DenseIndex):
    def __init__(self):
        self.vecs = None
    def build(self, vecs: np.ndarray):
        self.vecs = vecs.astype(np.float32, copy=False)
    def search(self, query_vec: np.ndarray, k: int):
        scores = self.vecs @ query_vec
        if k >= len(scores):
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, k)[:k]
            idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]

class FaissFlat(DenseIndex):
    def __init__(self):
        if faiss is None: raise RuntimeError("faiss not available")
        self.index = None
    def build(self, vecs: np.ndarray):
        d = vecs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(vecs)
    def search(self, query_vec: np.ndarray, k: int):
        D, I = self.index.search(query_vec.reshape(1,-1).astype(np.float32), k)
        return I[0], D[0]

class FaissIVF(DenseIndex):
    def __init__(self, nlist: int = 256, nprobe: int = 8):
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
        index = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = self.ef_search
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

# ========================= Retrieval =========================
class Retriever:
    def prepare(self, corpus_texts: List[str]):
        pass
    def score(self, query: str, candidates: List[str], candidate_idx: List[int]) -> List[float]:
        raise NotImplementedError

class DenseRetriever(Retriever):
    def __init__(self, client: OpenAI, model: str, cache: EmbeddingCache, chunk_vecs: np.ndarray):
        self.client, self.model, self.cache = client, model, cache
        self.chunk_vecs = chunk_vecs
    def score(self, query: str, candidates: List[str], candidate_idx: List[int]) -> List[float]:
        qv = embed_texts(self.client, [query], self.model, batch_size=1, cache=self.cache)[0]
        return (self.chunk_vecs[candidate_idx] @ qv).tolist()

class BM25Retriever(Retriever):
    def __init__(self):
        if BM25Okapi is None:
            raise RuntimeError("rank_bm25 not installed")
        self.bm25 = None
    def prepare(self, corpus_texts: List[str]):
        tok = [self._tokenize(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(tok)
    def _tokenize(self, s: str) -> List[str]:
        s = s.lower(); s = re.sub(r"[^a-z0-9]+", " ", s)
        return [w for w in s.split() if w]
    def score(self, query: str, candidates: List[str], candidate_idx: List[int]) -> List[float]:
        q = self._tokenize(query)
        scores_full = self.bm25.get_scores(q)
        return [float(scores_full[i]) for i in candidate_idx]

class HybridRetriever(Retriever):
    def __init__(self, dense: DenseRetriever, bm25: Optional[BM25Retriever], alpha: float = 0.5):
        self.dense, self.bm25, self.alpha = dense, bm25, alpha
    def score(self, query: str, candidates: List[str], candidate_idx: List[int]) -> List[float]:
        d = np.array(self.dense.score(query, candidates, candidate_idx), dtype=np.float32)
        if self.bm25 is None: return d.tolist()
        b = np.array(self.bm25.score(query, candidates, candidate_idx), dtype=np.float32)
        # per-query normalize
        def norm(x):
            if len(x)==0: return x
            v = x - x.min() if np.isfinite(x).all() else x
            m = v.max(); return v/(m+1e-6) if m>0 else v
        h = self.alpha*norm(d) + (1-self.alpha)*norm(b)
        return h.tolist()

# Optional LLM re-ranker (OFF by default)
class LLMReranker:
    def __init__(self, client: OpenAI, model: str):
        self.client, self.model = client, model
    def rerank(self, query: str, texts: List[str], top_m: int = 5) -> List[int]:
        bullets = "\n\n".join([f"[CANDIDATE {i}]\n{texts[i][:800]}" for i in range(len(texts))])
        prompt = (
            "Rate relevance (0-100) of each candidate to the query. Return a JSON list of indices in new rank order.\n"
            f"Query: {query}\n\n{bullets}\n\nReturn only JSON list of integers."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=0,
                messages=[{"role":"user","content":prompt}], max_tokens=256,
            )
            import json as _json
            txt = resp.choices[0].message.content.strip()
            arr = _json.loads(re.findall(r"\[[^\]]*\]", txt)[0])
            arr = [i for i in arr if isinstance(i, int) and 0 <= i < len(texts)]
            return arr[:top_m]
        except Exception:
            return list(range(min(top_m, len(texts))))

# ========================= Pipeline helpers =========================

def build_chunks(docs: List[ProductDoc], chunker: Chunker) -> Tuple[List[str], List[Dict[str, Any]]]:
    texts, meta = [], []
    for doc in docs:
        t = product_text(doc)
        parts = chunker.chunk(t)
        for i, ch in enumerate(parts):
            clean = ch.replace("\u0000", " ").strip()
            if not clean: continue
            texts.append(clean)
            meta.append({"doc_id": doc.doc_id, "title": doc.title, "category": doc.category, "url": doc.url, "chunk_idx": i})
    return texts, meta


def synthesize_queries(docs: List[ProductDoc], per_doc: int = 1) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for d in docs:
        cands = []
        if d.title: cands.append(d.title)
        if d.brand and d.category: cands.append(f"{d.brand} {d.category}")
        if d.category: cands.append(f"best {d.category}")
        seen = set(); cands = [x for x in cands if x and not (x in seen or seen.add(x))]
        random.shuffle(cands)
        for q in cands[:per_doc]:
            out.append((q, d.doc_id))
    random.shuffle(out)
    return out

# Metrics

def hit_at_k(ranks: List[Optional[int]], k: int) -> float:
    return float(np.mean([1.0 if (r is not None and r < k) else 0.0 for r in ranks]))

def mrr_at_k(ranks: List[Optional[int]], k: int) -> float:
    vals = []
    for r in ranks:
        if r is not None and r < k: vals.append(1.0 / (r + 1))
        else: vals.append(0.0)
    return float(np.mean(vals))

# ========================= Runner =========================

def run():
    C = CONFIG
    t0 = time.time()
    client = OpenAI()

    print(f"[INFO] Loading products from: {C['root']}")
    docs = iter_product_docs(C['root'], max_docs=C['max_docs'])
    if not docs:
        print("[ERROR] No products found. Check CONFIG['root'] path.")
        sys.exit(1)
    print(f"[OK] Loaded {len(docs)} docs")

    # >>> Save the exact set of products this run will use
    if C.get("export_products", False):
        export_products(docs, C.get("export_dir", "./exports"))

    qpairs = synthesize_queries(docs, per_doc=C['queries_per_doc'])
    if C['max_queries']:
        qpairs = qpairs[:C['max_queries']]
    print(f"[OK] Using {len(qpairs)} synthetic queries")

    rows = []

    # Prepare chunkers
    for chunker_name in C['chunkers']:
        if chunker_name == "recursive":
            chunker = RecursiveChunkerWrapper(C['chunk_size_tokens'], C['min_chars_per_chunk'])
        elif chunker_name == "sentence":
            chunker = SentenceChunker(max_tokens=C['chunk_size_tokens'], overlap_sentences=C['sent_overlap'])
        elif chunker_name == "semantic":
            chunker = SemanticChunker(client, C['embed_model'], max_tokens=C['chunk_size_tokens'], threshold=C['semantic_threshold'])
        else:
            print(f"[WARN] Unknown chunker {chunker_name}; skipping"); continue

        t = time.time()
        
        # ----------------------------------------
        chunks_texts, chunks_meta = build_chunks(docs, chunker)
        print(f"[OK] {chunker_name} -> {len(chunks_texts)} chunks (avg {len(chunks_texts)/len(docs):.2f}/doc) in {time.time()-t:.2f}s")

        # Embed chunks once
        cache = EmbeddingCache(C['embed_cache_path'])
        emb_path = f"emb_{chunker_name}.npy"
        if os.path.exists(emb_path):
            chunk_vecs = np.load(emb_path)
            print(f"[OK] Loaded cached embeddings: {chunk_vecs.shape} from {emb_path}")
        else:
            t = time.time()
            chunk_vecs = embed_texts(client, chunks_texts, C['embed_model'],
                                    batch_size=C['embed_batch_size'], cache=cache)
            print(f"[OK] Embedded chunks: {chunk_vecs.shape} in {time.time()-t:.2f}s")
            np.save(emb_path, chunk_vecs)
            with open(f"chunks_{chunker_name}.jsonl","w",encoding="utf-8") as f:
                for t_line in chunks_texts:
                    f.write(t_line.replace("\n"," ") + "\n")

        # Optional BM25
        bm25_retriever = None
        if any(r in ("bm25","hybrid","hybrid_rerank") for r in C['retrievers']):
            if BM25Okapi is None: print("[WARN] rank_bm25 not installed; skipping BM25/Hybrid")
            else:
                bm25_retriever = BM25Retriever(); bm25_retriever.prepare(chunks_texts)

        # Build indexes
        for index_name in C['indexes']:
            if index_name not in INDEX_REGISTRY:
                print(f"[WARN] Unknown index {index_name}; skipping"); continue
            IndexCls = INDEX_REGISTRY[index_name]
            try:
                index = IndexCls()
            except Exception as e:
                print(f"[WARN] Index backend {index_name} unavailable: {e}"); continue

            t = time.time(); index.build(chunk_vecs); build_time = time.time() - t

            dense = DenseRetriever(client, C['embed_model'], cache, chunk_vecs)
            retrievers: Dict[str, Retriever] = {}
            for rname in C['retrievers']:
                if rname == "dense": retrievers[rname] = dense
                elif rname == "bm25" and bm25_retriever is not None: retrievers[rname] = bm25_retriever
                elif rname == "hybrid" and bm25_retriever is not None: retrievers[rname] = HybridRetriever(dense, bm25_retriever, alpha=C['hybrid_alpha'])
                elif rname.endswith("rerank"):
                    base = HybridRetriever(dense, bm25_retriever, alpha=C['hybrid_alpha']) if rname=="hybrid_rerank" else dense
                    retrievers[rname] = base
                else:
                    print(f"[WARN] Retriever {rname} unavailable; skipping")

            for retr_name, retr in retrievers.items():
                ranks: List[Optional[int]] = []
                q_times: List[float] = []
                reranker = LLMReranker(client, C['chat_model']) if C['use_llm_rerank'] and retr_name.endswith("rerank") else None

                for (query, target_id) in qpairs:
                    t = time.time()
                    # Candidate set from vector index (top 5*K to be safe)
                    qv = embed_texts(client, [query], C['embed_model'], batch_size=1, cache=cache)[0]
                    I, D = index.search(qv, k=C['K']*3)
                    cand_idx = I.tolist()
                    cand_texts = [chunks_texts[i] for i in cand_idx]

                    scores = retr.score(query, cand_texts, cand_idx)
                    order = np.argsort(-np.array(scores))
                    ordered_idx = [cand_idx[i] for i in order]

                    if reranker is not None:
                        topM = min(C['rerank_top_m'], len(ordered_idx))
                        sub_texts = [chunks_texts[i] for i in ordered_idx[:topM]]
                        reord_local = reranker.rerank(query, sub_texts, top_m=topM)
                        ordered_idx = [ordered_idx[i] for i in reord_local] + ordered_idx[topM:]

                    # rank of correct doc (first matching chunk)
                    rank = None
                    for rpos, ci in enumerate(ordered_idx):
                        if chunks_meta[ci]["doc_id"] == target_id:
                            rank = rpos; break
                    ranks.append(rank)
                    q_times.append(time.time() - t)

                Hk = hit_at_k(ranks, C['K']); MRRk = mrr_at_k(ranks, C['K'])
                row = {
                    "chunker": chunker_name,
                    "index": index_name,
                    "retriever": retr_name,
                    "K": C['K'],
                    "hit@K": round(Hk, 4),
                    "mrr@K": round(MRRk, 4),
                    "avg_query_ms": int(1000*np.mean(q_times)) if q_times else None,
                    "p95_query_ms": int(1000*np.percentile(q_times, 95)) if q_times else None,
                    "index_build_s": round(build_time, 2),
                    "num_chunks": len(chunks_texts),
                    "avg_chunks_per_doc": round(len(chunks_texts)/len(docs), 2),
                }
                rows.append(row)
                print("[RESULT]", row)
                gc.collect()

    if not rows:
        print("[ERROR] No results collected"); return

    # Sort best first (mrr desc, then avg latency asc)
    rows.sort(key=lambda r: (-r["mrr@K"], r["avg_query_ms"] if r["avg_query_ms"] is not None else 1e9))

    # Pretty summary
    cols = ["chunker","index","retriever","K","hit@K","mrr@K","avg_query_ms","p95_query_ms","index_build_s","num_chunks","avg_chunks_per_doc"]
    widths = {c: max(len(c), max((len(str(r[c])) for r in rows), default=0)) for c in cols}
    print("\n==== SUMMARY (best first) ====")
    print(" ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print(" ".join(str(r[c]).ljust(widths[c]) for c in cols))

if __name__ == "__main__":
    run()
