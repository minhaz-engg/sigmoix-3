#!/usr/bin/env python3
"""
RAG for Daraz product Q&A over local JSON dumps.

- Expects folder layout:
  result/
    www_daraz_com_bd_.../
      products.json
      (other files ignored)

- Uses Chonkie for chunking and OpenAI embeddings + LLM for answers.
- Embedding cache is persisted to: .rag_cache/embeddings.sqlite

Run:
  python products_rag.py

Edit the QUERY constant below to ask a different question.
"""

import os
import sys
import json
import glob
import time
import math
import sqlite3
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# --- OpenAI client (Responses + Embeddings) ---
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # load OPENAI_API_KEY from .env if present

# --- Chunking with Chonkie ---
# You can swap to TokenChunker if you install extras: from chonkie import TokenChunker
from chonkie import RecursiveChunker

# =========================
# Configuration
# =========================
RESULTS_DIR = "result"  # root folder you described
EMBED_MODEL = "text-embedding-3-small"  # fast + strong multilingual
EMBED_DIMS = 512  # shrink dims to 512 for speed/memory (supported for -3 models)
LLM_MODEL = "gpt-4o-mini"  # good cost/latency balance
TOP_K = 15  # retrieved chunks to send to the LLM
CHUNK_MAX_CHARS = 1800  # let chonkie decide; recursive chunking keeps these sane
CHUNK_OVERLAP = 180
CACHE_DIR = ".rag_cache"
CACHE_DB_PATH = os.path.join(CACHE_DIR, "embeddings.sqlite")

# Put your question here (no CLI args needed)
QUERY = (
    "From all categories, list 8 laptop of highest price "
    "with the best rating and highest number of ratings. Include name, price, "
    "rating (avg & count), seller, and the product URL."
)

# =========================
# Utilities
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def now_ts() -> int:
    return int(time.time())

def to_float_bdt(x: Any) -> float:
    """
    Try to parse values like '৳ 172', '172.0', 'BDT 199', etc. Return float or NaN.
    """
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    # remove currency symbol and commas/spaces
    for tok in ["৳", "BDT", "Tk", "tk", "৳.", "৳. "]:
        s = s.replace(tok, "")
    s = s.replace(",", " ").replace("\u00a0", " ").strip()
    # keep first numeric token
    num = ""
    for ch in s:
        if ch.isdigit() or ch in ".-":
            num += ch
        elif num:
            break
    try:
        return float(num) if num else float("nan")
    except Exception:
        return float("nan")

def safe_get(d: Dict, *path, default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def join_unique(items: Iterable[str], sep: str = ", ") -> str:
    seen = []
    for x in items:
        if not x:
            continue
        x = str(x).strip()
        if x and x not in seen:
            seen.append(x)
    return sep.join(seen)

# =========================
# Loading & Normalization
# =========================

@dataclass
class ProductMeta:
    category: str
    product_id: str
    title: str
    url: str
    price_value: float
    price_display: str
    rating_avg: float
    rating_count: int
    brand: str
    seller: str

@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    meta: ProductMeta
    chunk_index: int

def product_to_text(prod: Dict[str, Any], category: str, source_path: str) -> Tuple[str, ProductMeta]:
    """
    Flatten a product JSON into a compact text record suitable for retrieval.
    Covers top-level, detail.*, seller.*, variants.*, etc.
    """
    detail = prod.get("detail", {}) or {}
    variants = detail.get("variants", []) or []

    pid = (
        prod.get("data_item_id")
        or safe_get(detail, "url")
        or safe_get(detail, "name")
        or sha256(json.dumps(prod, sort_keys=True)[:256])
    )
    title = (
        prod.get("product_title")
        or safe_get(detail, "name")
        or f"Product {pid}"
    )

    url = (
        prod.get("detail_url")
        or prod.get("product_detail_url", "")
        or safe_get(detail, "url", default="")
    )
    # normalize //www.daraz.com.bd/... to https
    if url.startswith("//"):
        url = "https:" + url

    # prices
    price_disp = (
        prod.get("product_price")
        or safe_get(detail, "price", "display", default="")
    )
    price_val = (
        to_float_bdt(safe_get(detail, "price", "value"))
        if not price_disp
        else to_float_bdt(price_disp)
    )
    orig_disp = safe_get(detail, "price", "original_display", default="")
    discount_disp = safe_get(detail, "price", "discount_display", default="")
    discount_pct = safe_get(detail, "price", "discount_percent", default=None)

    # rating
    rating_avg = float(safe_get(detail, "rating", "average", default=float("nan")) or float("nan"))
    rating_count = int(safe_get(detail, "rating", "count", default=0) or 0)

    brand = safe_get(detail, "brand", default="No Brand") or "No Brand"
    seller_name = safe_get(detail, "seller", "name", default="") or ""
    seller_metrics = safe_get(detail, "seller", "metrics", default={}) or {}
    metrics_str = "; ".join(f"{k}: {v}" for k, v in seller_metrics.items()) if seller_metrics else ""
    seller = (seller_name + (f" ({metrics_str})" if metrics_str else "")).strip()

    sold_str = prod.get("location", "")  # often "536 sold"
    colors = (safe_get(detail, "colors") or []) + [v.get("color") for v in variants if v.get("color")]
    sizes = []
    for v in variants:
        sizes.extend(v.get("sizes") or [])

    # delivery, returns, description
    delivery_options = safe_get(detail, "delivery_options", default=[]) or []
    delivery_str = "; ".join(
        " / ".join(str(x.get(k, "")).strip() for k in ("title", "time", "fee") if x.get(k))
        for x in delivery_options
    )
    returns = join_unique(safe_get(detail, "return_and_warranty", default=[]) or [])
    desc = (
        safe_get(detail, "details", "description_text")
        or safe_get(detail, "details", "raw_text")
        or ""
    )
    # images (limit to a few—helps answer "does it include X?" queries)
    imgs = (safe_get(detail, "images") or [])[:4]

    # variants summary
    var_lines = []
    for v in variants[:12]:
        v_price = safe_get(v, "price", "display") or safe_get(v, "price", "value")
        v_sizes = join_unique(v.get("sizes") or [])
        v_imgs = "; ".join((v.get("images") or [])[:2])
        var_lines.append(
            f"- color: {v.get('color') or 'N/A'} | price: {v_price} | sizes: {v_sizes or '—'} | images: {v_imgs}"
        )
    variants_str = "\n".join(var_lines)

    lines = [
        f"TITLE: {title}",
        f"CATEGORY: {category}",
        f"PRODUCT_ID: {pid}",
        f"URL: {url}",
        f"PRICE: {price_disp or price_val} | ORIGINAL: {orig_disp or ''} | DISCOUNT: {discount_disp or discount_pct or ''}",
        f"PRICE_VALUE_NUMERIC: {price_val}",
        f"RATING: avg={rating_avg} | count={rating_count}",
        f"BRAND: {brand}",
        f"SELLER: {seller}",
        f"SOLD: {sold_str}",
        f"COLORS: {join_unique(colors)}",
        f"SIZES: {join_unique(sizes)}",
        f"DELIVERY: {delivery_str}",
        f"RETURNS: {returns}",
        "IMAGES: " + "; ".join(imgs),
        "DESCRIPTION:",
        desc.strip(),
    ]
    if variants_str:
        lines.append("VARIANTS:\n" + variants_str)

    text = "\n".join(lines).strip()

    meta = ProductMeta(
        category=category,
        product_id=str(pid),
        title=str(title),
        url=str(url),
        price_value=float(price_val) if price_val == price_val else float("nan"),
        price_display=str(price_disp or ""),
        rating_avg=float(rating_avg) if rating_avg == rating_avg else float("nan"),
        rating_count=int(rating_count),
        brand=str(brand),
        seller=str(seller or ""),
    )
    return text, meta

def iter_products(result_dir: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Yields (category_name, product_dict)
    """
    pattern = os.path.join(result_dir, "*", "products.json")
    for path in glob.glob(pattern):
        category = os.path.basename(os.path.dirname(path))
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for prod in data:
                if isinstance(prod, dict):
                    yield category, prod
        except Exception as e:
            print(f"[warn] Failed to load {path}: {e}", file=sys.stderr)

# =========================
# Chunking
# =========================

def chunk_product(text: str, meta: ProductMeta, chunker: RecursiveChunker) -> List[ChunkRecord]:
    # Chonkie returns an iterable of chunks with .text
    chunks = chunker(text)
    records: List[ChunkRecord] = []
    for i, ch in enumerate(chunks):
        chunk_text = ch.text
        # Light size control: truncate overly huge chunks to keep LLM fast
        if len(chunk_text) > CHUNK_MAX_CHARS:
            chunk_text = chunk_text[:CHUNK_MAX_CHARS]
        rec = ChunkRecord(
            chunk_id=f"{meta.product_id}__{i}",
            text=chunk_text,
            meta=meta,
            chunk_index=i,
        )
        records.append(rec)
    return records

# =========================
# Embedding Cache (SQLite)
# =========================

class EmbeddingCache:
    def __init__(self, db_path: str):
        ensure_dir(os.path.dirname(db_path))
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dims INTEGER NOT NULL,
                vec BLOB NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)
        self.conn.commit()

    def get(self, key: str) -> np.ndarray | None:
        cur = self.conn.cursor()
        cur.execute("SELECT dims, vec FROM embeddings WHERE key = ?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        dims, blob = row
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != dims:
            return None
        return arr

    def set(self, key: str, model: str, dims: int, vec: np.ndarray):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO embeddings (key, model, dims, vec, created_at) VALUES (?, ?, ?, ?, ?)",
            (key, model, dims, vec.astype(np.float32).tobytes(), now_ts()),
        )
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

def embed_texts_with_cache(
    client: OpenAI,
    texts: List[str],
    model: str = EMBED_MODEL,
    dims: int = EMBED_DIMS,
    cache: EmbeddingCache | None = None,
    batch_size: int = 64,
) -> List[np.ndarray]:
    """
    Embeds a list of texts, caching results in SQLite by content hash.
    """
    out: List[np.ndarray] = [None] * len(texts)  # type: ignore
    missing_idx: List[int] = []
    keys: List[str] = []
    for i, t in enumerate(texts):
        key = sha256(f"{model}:{dims}:{t}")
        keys.append(key)
        if cache:
            vec = cache.get(key)
            if vec is not None:
                out[i] = vec
                continue
        missing_idx.append(i)

    # Embed the missing ones in batches
    for bstart in range(0, len(missing_idx), batch_size):
        bindices = missing_idx[bstart : bstart + batch_size]
        batch = [texts[i] for i in bindices]
        if not batch:
            continue
        resp = client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float",
            dimensions=dims,
        )
        # response order matches input order
        for local_i, i_global in enumerate(bindices):
            vec = np.array(resp.data[local_i].embedding, dtype=np.float32)
            out[i_global] = vec
            if cache:
                cache.set(keys[i_global], model, dims, vec)

    # fill type checker
    return [v if isinstance(v, np.ndarray) else np.zeros((dims,), dtype=np.float32) for v in out]

# =========================
# Retrieval
# =========================

@dataclass
class BuiltIndex:
    embeddings: np.ndarray  # (N, D) L2-normalized
    chunks: List[ChunkRecord]

def build_index(client: OpenAI, chunks: List[ChunkRecord]) -> BuiltIndex:
    texts = [c.text for c in chunks]
    cache = EmbeddingCache(CACHE_DB_PATH)
    try:
        vecs = embed_texts_with_cache(client, texts, cache=cache, dims=EMBED_DIMS)
    finally:
        cache.close()

    mat = np.vstack(vecs)  # (N, D)
    # L2 normalize
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat = mat / norms
    return BuiltIndex(embeddings=mat, chunks=chunks)

def search(
    client: OpenAI,
    index: BuiltIndex,
    query: str,
    top_k: int = TOP_K,
) -> List[Tuple[float, ChunkRecord]]:
    q_vec = embed_texts_with_cache(client, [query], dims=EMBED_DIMS)[0]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    scores = index.embeddings @ q_vec
    # Top-K
    if top_k >= len(scores):
        top_idx = np.argsort(-scores)
    else:
        # argpartition is faster
        part = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = part[np.argsort(-scores[part])]
    results = [(float(scores[i]), index.chunks[i]) for i in top_idx]
    return results

# =========================
# LLM Answer
# =========================

SYSTEM_INSTRUCTIONS = """You are a product QA assistant for a Daraz-like catalog.
Answer ONLY using the provided context chunks. If the answer is not present, say "I don't know".
Prefer numeric fields like PRICE_VALUE_NUMERIC, RATING avg/count, and always include product URLs.
If the user asks for filtering (e.g., price thresholds, rating), apply those using the numbers present.
When listing multiple items, sort by the user's criteria; if ambiguous, sort by rating (avg desc, then count desc).
Return a concise answer followed by a bulleted list or simple table of the chosen products.
"""

def make_context_block(hits: List[Tuple[float, ChunkRecord]], max_products: int = 12) -> str:
    """
    Merge top chunks, limiting to a handful of products and 1-2 chunks/product.
    """
    by_pid: Dict[str, List[Tuple[float, ChunkRecord]]] = {}
    for score, rec in hits:
        by_pid.setdefault(rec.meta.product_id, []).append((score, rec))

    # Keep best N products, 2 chunks each
    # score per product = max score among its chunks
    ranked_pids = sorted(
        by_pid.keys(),
        key=lambda pid: max(s for s, _ in by_pid[pid]),
        reverse=True,
    )[:max_products]

    blocks = []
    for pid in ranked_pids:
        entries = sorted(by_pid[pid], key=lambda x: x[0], reverse=True)[:2]
        header = f"[PRODUCT {pid}] {entries[0][1].meta.title} | URL: {entries[0][1].meta.url}"
        body = "\n-----\n".join(rec.text for _, rec in entries)
        blocks.append(header + "\n" + body)
    return "\n\n====================\n\n".join(blocks)

def answer_with_llm(client: OpenAI, query: str, context: str) -> str:
    prompt = (
        SYSTEM_INSTRUCTIONS
        + "\n\nQuestion:\n"
        + query
        + "\n\nContext (top retrieved chunks):\n"
        + context
        + "\n\nAnswer:"
    )
    resp = client.responses.create(
        model=LLM_MODEL,
        input=prompt,
        max_output_tokens=500,
        temperature=0.0,
    )
    # Try convenience property; fallback to manual traversal
    text = getattr(resp, "output_text", None)
    if not text:
        try:
            text = resp.output[0].content[0].text  # type: ignore[attr-defined]
        except Exception:
            text = str(resp)
    return text

# =========================
# Main pipeline
# =========================

def main():
    # 1) Collect & normalize products
    products_normalized: List[Tuple[str, ProductMeta]] = []
    chunker = RecursiveChunker()  # simple, fast default

    print("[info] Scanning categories and loading products...")
    all_chunks: List[ChunkRecord] = []
    count_products = 0

    for category, prod in iter_products(RESULTS_DIR):
        try:
            text, meta = product_to_text(prod, category, source_path="")
            # 2) Chunk with Chonkie
            chunks = chunk_product(text, meta, chunker)
            all_chunks.extend(chunks)
            count_products += 1
        except Exception as e:
            print(f"[warn] product skipped due to error: {e}", file=sys.stderr)

    if not all_chunks:
        print("[error] No chunks built. Ensure 'result/*/products.json' exist.", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Loaded {count_products} products, built {len(all_chunks)} chunks.")

    # 3) Build index (embeddings with cache)
    client = OpenAI()  # uses OPENAI_API_KEY
    print("[info] Embedding chunks (cached) and building index...")
    index = build_index(client, all_chunks)

    # 4) Retrieve
    print("[info] Retrieving top chunks for the query...")
    hits = search(client, index, QUERY, top_k=TOP_K)

    # 5) Build concise context (limit products & chunks)
    context = make_context_block(hits, max_products=12)

    # 6) Ask LLM
    print("[info] Asking the model...")
    answer = answer_with_llm(client, QUERY, context)

    # 7) Output
    print("\n================ ANSWER ================\n")
    print(answer)
    print("\n================ DEBUG (Top 10 Retrieved) ================\n")
    for i, (score, rec) in enumerate(hits[:10], 1):
        m = rec.meta
        print(
            f"{i:02d}. score={score:.4f} | {m.title[:80]} | ৳{m.price_value:.0f} | "
            f"rating={m.rating_avg} ({m.rating_count}) | {m.url}"
        )

if __name__ == "__main__":
    main()
