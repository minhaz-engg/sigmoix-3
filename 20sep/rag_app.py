#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final RAG app based on experiment decision:
- Default: Recursive chunking + BM25 over full corpus (best MRR@5 in your results)
- Optional Hybrid: BM25 ∪ Dense (FAISS HNSW) fused with RRF (toggle with --mode hybrid)
- Optional LLM rerank over top-M
- Answer synthesis with inline [DOC#CHUNK] citations

Usage examples:
  python final_rag.py --root ../layer_23/result --query "best budget phone" --top-k 5
  python final_rag.py --mode hybrid --query "wireless earbuds for running"
  python final_rag.py --chunker sentence --mode bm25 --query "gaming laptop under 70k"
  python final_rag.py --build   # force rebuild caches & indexes

Author: you + hippo power 🦛
"""

from __future__ import annotations
import os, re, json, glob, time, hashlib, pickle, argparse, random, sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
load_dotenv()

# ---------- Optional deps ----------
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

# ---------- Config ----------
CONFIG = {
    "root": "../layer_23/result",        # folder with */products.json
    "persist_dir": "./store",            # cache/index directory
    "export_dir": "./exports",           # optional exports

    "chunker": "recursive",              # "recursive" (default) or "sentence"
    "chunk_size_tokens": 280,
    "min_chars_per_chunk": 24,
    "sentence_overlap": 1,

    "embed_model": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
    "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),

    "retrieval_mode": "bm25",            # "bm25" (default) or "hybrid"
    "top_k": 5,

    # Hybrid knobs
    "bm25_topn": 200,                    # internal candidate pool size
    "dense_topn": 200,
    "rrf_k": 60,                         # RRF constant (larger -> flatter)
    "dense_index": "hnsw",               # "hnsw" | "flat" | "ivf" | "numpy"

    # LLM rerank
    "use_llm_rerank": False,
    "rerank_top_m": 10,

    "embed_batch_size": 128,
    "random_seed": 42,

    # Exports
    "export_products": True,
}

random.seed(CONFIG["random_seed"]); np.random.seed(CONFIG["random_seed"])

# ---------- Data model ----------
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

# ---------- Exports ----------
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
    left_margin = 0.75 * inch; right_margin = 0.75 * inch
    top_margin = 0.75 * inch; bottom_margin = 0.75 * inch
    usable_width = width - left_margin - right_margin
    y = height - top_margin
    c.setFont("Helvetica", 10)

    def draw_wrapped(text: str, header: Optional[str] = None):
        nonlocal y
        if header:
            c.setFont("Helvetica-Bold", 12)
            for line in simpleSplit(header, "Helvetica-Bold", 12, usable_width):
                if y < bottom_margin + 12:
                    c.showPage(); c.setFont("Helvetica", 10); y = height - top_margin
                c.drawString(left_margin, y, line); y -= 14
            c.setFont("Helvetica", 10); y -= 4
        for line in simpleSplit(text, "Helvetica", 10, usable_width):
            if y < bottom_margin + 12:
                c.showPage(); c.setFont("Helvetica", 10); y = height - top_margin
            c.drawString(left_margin, y, line); y -= 12
        if y < bottom_margin + 12:
            c.showPage(); c.setFont("Helvetica", 10); y = height - top_margin
        c.drawString(left_margin, y, "-" * 50); y -= 16

    for i, d in enumerate(docs, 1):
        draw_wrapped(product_text(d), header=f"PRODUCT {i}")
    c.save()
    print(f".[OK] Wrote products PDF: {path}")

def export_products(docs: List[ProductDoc], out_dir: str, stem: str = "products_used"):
    os.makedirs(out_dir, exist_ok=True)
    export_products_txt(docs, os.path.join(out_dir, f"{stem}.txt"))
    export_products_pdf(docs, os.path.join(out_dir, f"{stem}.pdf"))

# ---------- Chunking ----------
class SentenceChunker:
    def __init__(self, max_tokens: int = 320, overlap: int = 0):
        self.max_tokens = max_tokens
        self.overlap = overlap
    def _split(self, s: str) -> List[str]:
        parts = re.split(r"(?<=[.!?\n])\s+", s)
        return [p.strip() for p in parts if p.strip()]
    def chunk(self, text: str) -> List[str]:
        sents = self._split(text)
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
            cur.append(sent); cur_tok += t
        if cur: chunks.append(" ".join(cur).strip())
        return chunks

class RecursiveChunkerWrapper:
    def __init__(self, size_tokens: int, min_chars: int):
        self._c = RecursiveChunker(tokenizer_or_token_counter=count_tokens,
                                   chunk_size=size_tokens,
                                   min_characters_per_chunk=min_chars)
    def chunk(self, text: str) -> List[str]:
        return [c.text.strip() for c in self._c(text) if c.text and c.text.strip()]

def build_chunks(docs: List[ProductDoc], chunker_name: str, size_tokens: int, min_chars: int, sent_overlap: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    if chunker_name == "sentence":
        chunker = SentenceChunker(max_tokens=size_tokens, overlap=sent_overlap)
    else:
        chunker = RecursiveChunkerWrapper(size_tokens, min_chars)
    texts, meta = [], []
    for d in docs:
        for i, ch in enumerate(chunker.chunk(product_text(d))):
            ch = ch.replace("\u0000", " ").strip()
            if not ch: continue
            texts.append(ch)
            meta.append({"doc_id": d.doc_id, "title": d.title, "category": d.category, "url": d.url, "chunk_idx": i})
    return texts, meta

# ---------- Embeddings ----------
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
            key = _sha1_for_text_model(t.replace("\n"," "), model)
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
        if not batch_ids: break
        batch = [texts[i].replace("\n"," ") for i in batch_ids]
        resp = client.embeddings.create(model=model, input=batch, encoding_format="float")
        for local_j, i in enumerate(batch_ids):
            vec = resp.data[local_j].embedding
            all_vecs[i] = vec
            if cache and keys[i]: cache.set(keys[i], vec)
    if cache: cache.save()

    arr = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True); norms[norms == 0.0] = 1.0
    return arr / norms

# ---------- Dense indexes ----------
class DenseIndex:
    def build(self, vecs: np.ndarray): ...
    def search(self, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]: ...

class NumpyFlat(DenseIndex):
    def __init__(self): self.vecs=None
    def build(self, vecs: np.ndarray): self.vecs = vecs.astype(np.float32, copy=False)
    def search(self, qv: np.ndarray, k: int):
        s = self.vecs @ qv
        if k >= len(s): idx = np.argsort(-s)
        else:
            idx = np.argpartition(-s, k)[:k]; idx = idx[np.argsort(-s[idx])]
        return idx, s[idx]

class FaissFlat(DenseIndex):
    def __init__(self):
        if faiss is None: raise RuntimeError("faiss not available")
        self.index=None
    def build(self, vecs: np.ndarray):
        d = vecs.shape[1]
        self.index = faiss.IndexFlatIP(d); self.index.add(vecs)
    def search(self, qv: np.ndarray, k: int):
        D, I = self.index.search(qv.reshape(1,-1).astype(np.float32), k); return I[0], D[0]

class FaissIVF(DenseIndex):
    def __init__(self, nlist: Optional[int] = None, nprobe: int = 8):
        if faiss is None: raise RuntimeError("faiss not available")
        self.nlist, self.nprobe, self.index = nlist, nprobe, None
    def build(self, vecs: np.ndarray):
        d = vecs.shape[1]
        n = vecs.shape[0]
        nlist = self.nlist or max(8, int(np.sqrt(n)))  # adaptive, avoids training warning at small n
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vecs); index.add(vecs); index.nprobe = min(self.nprobe, nlist)
        self.index = index
    def search(self, qv: np.ndarray, k: int):
        D, I = self.index.search(qv.reshape(1,-1).astype(np.float32), k); return I[0], D[0]

class FaissHNSW(DenseIndex):
    def __init__(self, hnsw_m: int = 32, ef_search: int = 64):
        if faiss is None: raise RuntimeError("faiss not available")
        self.hnsw_m, self.ef_search, self.index = hnsw_m, ef_search, None
    def build(self, vecs: np.ndarray):
        d = vecs.shape[1]
        index = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = self.ef_search
        index.add(vecs); self.index = index
    def search(self, qv: np.ndarray, k: int):
        D, I = self.index.search(qv.reshape(1,-1).astype(np.float32), k); return I[0], D[0]

INDEXES = {
    "numpy": NumpyFlat,
    "flat": FaissFlat if faiss is not None else NumpyFlat,
    "ivf": FaissIVF if faiss is not None else NumpyFlat,
    "hnsw": FaissHNSW if faiss is not None else NumpyFlat,
}

# ---------- BM25 ----------
class BM25Wrapper:
    def __init__(self):
        if BM25Okapi is None:
            raise RuntimeError("rank_bm25 not installed. Run: pip install rank-bm25")
        self.bm25 = None
        self.corpus_tokens: List[List[str]] = []
    @staticmethod
    def _tok(s: str) -> List[str]:
        s = s.lower(); s = re.sub(r"[^a-z0-9]+", " ", s)
        return [w for w in s.split() if w]
    def build(self, corpus_texts: List[str]):
        self.corpus_tokens = [self._tok(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)
    def search(self, query: str, topn: int) -> Tuple[np.ndarray, np.ndarray]:
        q = self._tok(query)
        scores = np.array(self.bm25.get_scores(q), dtype=np.float32)
        if topn >= len(scores): idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, topn)[:topn]; idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]

# ---------- Fusion ----------
def rrf_fusion(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for rl in rank_lists:
        for r, idx in enumerate(rl):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r + 1.0)
    return scores

# ---------- Engine ----------
class RAGEngine:
    def __init__(self, C: Dict[str, Any]):
        self.C = C
        self.client = OpenAI()
        self.embed_cache = EmbeddingCache(os.path.join(C["persist_dir"], "embed_cache.pkl"))

        self.docs: List[ProductDoc] = []
        self.chunks_texts: List[str] = []
        self.chunks_meta: List[Dict[str, Any]] = []

        self.chunk_vecs: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Wrapper] = None
        self.dense_index: Optional[DenseIndex] = None

        os.makedirs(C["persist_dir"], exist_ok=True)
        os.makedirs(C["export_dir"], exist_ok=True)

    # ---- Build / load ----
    def build_or_load(self, force_rebuild: bool = False):
        t0 = time.time()
        # 1) Load docs
        print(f"[INFO] Loading products from: {self.C['root']}")
        self.docs = iter_product_docs(self.C["root"])
        if not self.docs:
            print("[ERROR] No products found."); sys.exit(1)
        print(f"[OK] Loaded {len(self.docs)} docs")

        if self.C.get("export_products", False):
            export_products(self.docs, self.C["export_dir"])

        # 2) Chunks (persisted per chunker)
        stem = f"chunks_{self.C['chunker']}"
        chunks_path = os.path.join(self.C["persist_dir"], f"{stem}.jsonl")
        meta_path = os.path.join(self.C["persist_dir"], f"{stem}_meta.json")

        if (not force_rebuild) and os.path.exists(chunks_path) and os.path.exists(meta_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks_texts = [ln.rstrip("\n") for ln in f]
            with open(meta_path, "r", encoding="utf-8") as f:
                self.chunks_meta = json.load(f)
            print(f"[OK] Loaded chunks: {len(self.chunks_texts)} ({self.C['chunker']})")
        else:
            self.chunks_texts, self.chunks_meta = build_chunks(
                self.docs, self.C["chunker"], self.C["chunk_size_tokens"], self.C["min_chars_per_chunk"], self.C["sentence_overlap"]
            )
            with open(chunks_path, "w", encoding="utf-8") as f:
                for t in self.chunks_texts:
                    f.write(t.replace("\n"," ") + "\n")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks_meta, f)
            print(f"[OK] Built chunks: {len(self.chunks_texts)} (saved)")

        # 3) BM25 over full corpus
        print("[INFO] Building BM25 over full chunk corpus ...")
        self.bm25 = BM25Wrapper(); self.bm25.build(self.chunks_texts)
        print("[OK] BM25 ready")

        # 4) Dense embeddings + index (only if hybrid or for later toggle)
        need_dense = (self.C["retrieval_mode"] == "hybrid") or True  # build once so toggling mode later is instant
        emb_path = os.path.join(self.C["persist_dir"], f"emb_{self.C['chunker']}.npy")
        if need_dense:
            if (not force_rebuild) and os.path.exists(emb_path):
                self.chunk_vecs = np.load(emb_path); print(f"[OK] Loaded embeddings: {self.chunk_vecs.shape}")
            else:
                print("[INFO] Embedding chunks ...")
                self.chunk_vecs = embed_texts(self.client, self.chunks_texts, self.C["embed_model"],
                                              batch_size=self.C["embed_batch_size"], cache=self.embed_cache)
                np.save(emb_path, self.chunk_vecs); print(f"[OK] Embedded chunks: {self.chunk_vecs.shape} (saved)")

            idx_kind = self.C["dense_index"]
            IndexCls = INDEXES.get(idx_kind, NumpyFlat)
            try:
                self.dense_index = IndexCls()
            except Exception as e:
                print(f"[WARN] Dense index {idx_kind} not available ({e}); falling back to numpy.")
                self.dense_index = NumpyFlat()
            print(f"[INFO] Building dense index: {self.dense_index.__class__.__name__} ...")
            self.dense_index.build(self.chunk_vecs); print("[OK] Dense index ready")

        print(f"[READY] Build completed in {time.time()-t0:.2f}s (mode={self.C['retrieval_mode']}, chunker={self.C['chunker']})")

    # ---- Retrieval ----
    def _dense_search(self, query: str, topn: int) -> List[int]:
        qv = embed_texts(self.client, [query], self.C["embed_model"], batch_size=1, cache=self.embed_cache)[0]
        I, _ = self.dense_index.search(qv, topn)
        return I.tolist()

    def retrieve(self, query: str, top_k: Optional[int] = None, mode: Optional[str] = None) -> List[int]:
        mode = (mode or self.C["retrieval_mode"]).lower()
        k = top_k or self.C["top_k"]

        if mode == "bm25":
            idx, _ = self.bm25.search(query, topn=k)
            return idx.tolist()

        # Hybrid: union + RRF over BM25 and Dense
        bm25_idx, _ = self.bm25.search(query, topn=self.C["bm25_topn"])
        dense_idx = self._dense_search(query, topn=self.C["dense_topn"])
        fused_scores = rrf_fusion([bm25_idx.tolist(), dense_idx], k=self.C["rrf_k"])
        ordered = [i for i, _ in sorted(fused_scores.items(), key=lambda kv: -kv[1])]
        return ordered[:k]

    # ---- Optional LLM rerank ----
    def rerank_with_llm(self, query: str, indices: List[int], top_m: int) -> List[int]:
        if not self.C["use_llm_rerank"] or top_m <= 0:
            return indices
        texts = [self.chunks_texts[i][:800] for i in indices[:top_m]]
        bullets = "\n\n".join([f"[CANDIDATE {i}]\n{t}" for i, t in enumerate(texts)])
        prompt = (
            "Rate relevance (0-100) of each candidate to the query. "
            "Return a JSON list of indices in new rank order.\n"
            f"Query: {query}\n\n{bullets}\n\nReturn only JSON list of integers."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.C["chat_model"], temperature=0,
                messages=[{"role":"user","content":prompt}], max_tokens=200,
            )
            import json as _json, re as _re
            arr = _json.loads(_re.findall(r"\[[^\]]*\]", resp.choices[0].message.content.strip())[0])
            arr = [i for i in arr if isinstance(i, int) and 0 <= i < len(texts)]
            return [indices[i] for i in arr] + indices[top_m:]
        except Exception:
            return indices

    # ---- Generate answer with citations ----
    def generate_answer(self, query: str, mode: Optional[str] = None, top_k: Optional[int] = None) -> Dict[str, Any]:
        mode = (mode or self.C["retrieval_mode"]).lower()
        k = top_k or self.C["top_k"]

        idx = self.retrieve(query, top_k=k*2, mode=mode)  # fetch some extra for optional rerank
        if self.C["use_llm_rerank"]:
            idx = self.rerank_with_llm(query, idx, top_m=min(self.C["rerank_top_m"], len(idx)))
        idx = idx[:k]

        contexts, citations = [], []
        for i in idx:
            m = self.chunks_meta[i]
            tag = f"[{m['doc_id']}#{m['chunk_idx']}]"
            contexts.append(f"{tag}\n{self.chunks_texts[i]}")
            citations.append({
                "doc_id": m["doc_id"], "chunk_idx": m["chunk_idx"],
                "title": m.get("title"), "url": m.get("url"), "category": m.get("category")
            })

        sys_prompt = "You are a Ecommerce prouduct assistant. You answer strictly from the provided context. Cite sources using their [DOC#CHUNK] tags."
        user_prompt = (
            f"Question: {query}\n\n"
            "Context:\n" + "\n\n---\n\n".join(contexts) + "\n\n"
            "Write:\n product name \n- price with 2–4 concise bullets with specifics\nproduct detail link\n"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.C["chat_model"], temperature=0.2,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user_prompt}],
                max_tokens=500,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception:
            answer = "Model call failed. Top contexts:\n\n" + "\n\n---\n\n".join(contexts)

        return {"query": query, "mode": mode, "top_k": k, "answer": answer, "citations": citations}

# ---------- CLI ----------
def main():
    C = CONFIG.copy()
    ap = argparse.ArgumentParser(description="Final RAG app (Recursive+BM25 default; Hybrid via RRF optional).")
    ap.add_argument("--root", type=str, default=C["root"])
    ap.add_argument("--persist-dir", type=str, default=C["persist_dir"])
    ap.add_argument("--export-dir", type=str, default=C["export_dir"])
    ap.add_argument("--mode", type=str, default=C["retrieval_mode"], choices=["bm25", "hybrid"])
    ap.add_argument("--chunker", type=str, default=C["chunker"], choices=["recursive", "sentence"])
    ap.add_argument("--dense-index", type=str, default=C["dense_index"], choices=["hnsw", "flat", "ivf", "numpy"])
    ap.add_argument("--top-k", type=int, default=C["top_k"])
    ap.add_argument("--build", action="store_true", help="Force rebuild of chunks/embeddings/index.")
    ap.add_argument("--rerank", action="store_true", help="Enable LLM rerank over top-M candidates.")
    ap.add_argument("--query", type=str, required=False, help="Ad-hoc query.")
    args = ap.parse_args()

    C["root"] = args.root
    C["persist_dir"] = args.persist_dir
    C["export_dir"] = args.export_dir
    C["retrieval_mode"] = args.mode
    C["chunker"] = args.chunker
    C["dense_index"] = args.dense_index
    C["top_k"] = int(args.top_k)
    C["use_llm_rerank"] = bool(args.rerank)

    engine = RAGEngine(C)
    engine.build_or_load(force_rebuild=args.build)

    if args.query:
        result = engine.generate_answer(args.query, mode=args.mode, top_k=args.top_k)
        print("\n=== ANSWER ===\n")
        print(result["answer"])
        # print("\n=== CITATIONS ===\n")
        # for c in result["citations"]:
        #     title = c.get("title") or "(untitled product)"
        #     url = c.get("url") or "(no url)"
        #     print(f"- {title}  [{c['doc_id']}#{c['chunk_idx']}]  {url}")
    else:
        print("\n[INFO] Engine ready. Try:")
        here = os.path.basename(__file__)
        print(f"  python {here} --query \"best budget phone\"")
        print(f"  python {here} --mode hybrid --query \"wireless earbuds for running\"")
        print(f"  python {here} --chunker sentence --mode bm25 --query \"gaming laptop under 70k\"")

if __name__ == "__main__":
    main()
