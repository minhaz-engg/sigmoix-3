#!/usr/bin/env python3
"""
RAG for Daraz product Q&A over local JSON using Hugging Face models (no OpenAI).

- Expects folder layout:
  result/
    <category>/
      products.json
      (other files ignored)

- Chunking: Chonkie (RecursiveChunker by default)
- Embeddings: intfloat/multilingual-e5-small (robust & fast; multilingual)
- LLM: Qwen/Qwen2.5-1.5B-Instruct (small, CPU-friendly). You can swap models via env vars.

Run:
  python products_rag_hf.py

Change the question by editing the QUERY constant.
"""

import os
import sys
import json
import glob
import time
import sqlite3
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ---- Chunking
from chonkie import RecursiveChunker  # or TokenChunker if you prefer

# ---- Embeddings (Sentence-Transformers)
from sentence_transformers import SentenceTransformer

# ---- HF LLM for answering
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

# =========================
# Configuration
# =========================

RESULTS_DIR = "result"  # root folder you described

# Embedding model (multilingual, small, solid)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))

# Generation model (light instruction-tuned)
# Alternatives: "google/gemma-2-2b-it", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GEN_MODEL_NAME = os.getenv("GEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
GEN_MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", "700"))
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.2"))

# Retrieval
TOP_K = 15
MAX_PRODUCTS_IN_CTX = 12
CHUNK_MAX_CHARS = 1800
CHUNK_OVERLAP = 180

# Cache
CACHE_DIR = ".rag_cache"
CACHE_DB_PATH = os.path.join(CACHE_DIR, "embeddings.sqlite")

# Your question (edit inline)
QUERY = (
    "From all categories, list 8 drone with the highest price "
    "best rating and highest rating counts. Include name, price, rating (avg & count), "
    "seller, and the product URL."
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
    Parse values like '৳ 172', '172.0', 'BDT 199', etc. Return float or NaN.
    """
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    for tok in ["৳", "BDT", "Tk", "tk", "৳.", "৳. "]:
        s = s.replace(tok, "")
    s = s.replace(",", " ").replace("\u00a0", " ").strip()
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

def join_unique(items, sep=", "):
    seen = []
    for x in items or []:
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

def product_to_text(prod: Dict[str, Any], category: str) -> Tuple[str, ProductMeta]:
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
    if url.startswith("//"):
        url = "https:" + url

    # Prices
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

    rating_avg = float(safe_get(detail, "rating", "average", default=float("nan")) or float("nan"))
    rating_count = int(safe_get(detail, "rating", "count", default=0) or 0)

    brand = safe_get(detail, "brand", default="No Brand") or "No Brand"
    seller_name = safe_get(detail, "seller", "name", default="") or ""
    seller_metrics = safe_get(detail, "seller", "metrics", default={}) or {}
    metrics_str = "; ".join(f"{k}: {v}" for k, v in seller_metrics.items()) if seller_metrics else ""
    seller = (seller_name + (f" ({metrics_str})" if metrics_str else "")).strip()

    sold_str = prod.get("location", "")
    colors = (safe_get(detail, "colors") or []) + [v.get("color") for v in variants if v.get("color")]
    sizes = []
    for v in variants:
        sizes.extend(v.get("sizes") or [])

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
    imgs = (safe_get(detail, "images") or [])[:4]

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

@dataclass
class BuiltChunk:
    chunk_id: str
    text: str
    meta: ProductMeta
    chunk_index: int

def chunk_product(text: str, meta: ProductMeta, chunker: RecursiveChunker) -> List[ChunkRecord]:
    chunks = chunker(text)
    out: List[ChunkRecord] = []
    for i, ch in enumerate(chunks):
        chunk_text = ch.text
        if len(chunk_text) > CHUNK_MAX_CHARS:
            chunk_text = chunk_text[:CHUNK_MAX_CHARS]
        out.append(
            ChunkRecord(
                chunk_id=f"{meta.product_id}__{i}",
                text=chunk_text,
                meta=meta,
                chunk_index=i,
            )
        )
    return out

# =========================
# Embedding cache (SQLite)
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

# =========================
# Embedding (HF ST)
# =========================

def build_embedder(model_name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model

def e5_wrap_passages(texts: List[str]) -> List[str]:
    # e5-style prefix for better retrieval quality
    return [f"passage: {t}" for t in texts]

def e5_wrap_query(q: str) -> str:
    return f"query: {q}"

def embed_texts_with_cache_hf(
    embedder: SentenceTransformer,
    texts: List[str],
    model_name: str,
    cache: EmbeddingCache | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
) -> List[np.ndarray]:
    dims = embedder.get_sentence_embedding_dimension()
    out: List[np.ndarray] = [None] * len(texts)  # type: ignore
    missing_idx: List[int] = []
    keys: List[str] = []
    for i, t in enumerate(texts):
        key = sha256(f"{model_name}:{t}")
        keys.append(key)
        if cache:
            vec = cache.get(key)
            if vec is not None:
                out[i] = vec
                continue
        missing_idx.append(i)

    if missing_idx:
        batch = [texts[i] for i in missing_idx]
        vecs = embedder.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine-ready
            show_progress_bar=False,
        )
        for local_i, i_global in enumerate(missing_idx):
            v = vecs[local_i].astype(np.float32)
            out[i_global] = v
            if cache:
                cache.set(keys[i_global], model_name, v.shape[0], v)

    return [v if isinstance(v, np.ndarray) else np.zeros((dims,), dtype=np.float32) for v in out]

# =========================
# Retrieval
# =========================

@dataclass
class BuiltIndex:
    embeddings: np.ndarray  # (N, D) cosine-ready (L2-normalized)
    chunks: List[ChunkRecord]
    dims: int

def build_index(embedder: SentenceTransformer, chunks: List[ChunkRecord], model_name: str) -> BuiltIndex:
    cache = EmbeddingCache(CACHE_DB_PATH)
    try:
        texts = [c.text for c in chunks]
        # e5 passage prefix
        texts_wrapped = e5_wrap_passages(texts)
        vecs = embed_texts_with_cache_hf(embedder, texts_wrapped, model_name, cache=cache)
    finally:
        cache.close()

    mat = np.vstack(vecs)  # already normalized
    dims = mat.shape[1]
    return BuiltIndex(embeddings=mat, chunks=chunks, dims=dims)

def search(
    embedder: SentenceTransformer,
    index: BuiltIndex,
    query: str,
    model_name: str,
    top_k: int = TOP_K,
) -> List[Tuple[float, ChunkRecord]]:
    q_vec = embed_texts_with_cache_hf(embedder, [e5_wrap_query(query)], model_name)[0]
    # cosine similarity via dot product (already normalized)
    scores = index.embeddings @ q_vec
    if top_k >= len(scores):
        top_idx = np.argsort(-scores)
    else:
        part = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = part[np.argsort(-scores[part])]
    return [(float(scores[i]), index.chunks[i]) for i in top_idx]

# =========================
# LLM Answering (HF local)
# =========================

SYSTEM_INSTRUCTIONS = """You are a product QA assistant for a Daraz-like catalog.
Answer ONLY using the provided context chunks. If the answer is not present, say "I don't know".
Prefer numeric fields like PRICE_VALUE_NUMERIC, RATING avg/count, and always include product URLs.
If the user asks for filtering (e.g., price thresholds, rating), apply those using the numbers present.
When listing multiple items, sort by the user's criteria; if ambiguous, sort by rating (avg desc, then count desc).
Return a concise answer followed by a bulleted list or simple table of the chosen products.
"""

def build_generator(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    # Load tokenizer & model
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    # dtype: bf16 on GPU if available; fall back to float16 if supported else float32 on CPU
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device != -1 else None,
        trust_remote_code=False,
    )
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device=device,
        return_full_text=False,
    )
    return tok, gen

def format_chat_prompt(tokenizer, system_msg: str, user_msg: str) -> str:
    """
    Use chat template if available; otherwise fall back to simple prompt.
    """
    try:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return (
            f"<<SYS>>\n{system_msg}\n<</SYS>>\n\n"
            f"[USER]\n{user_msg}\n[/USER]\n\n[ASSISTANT]"
        )

def make_context_block(hits: List[Tuple[float, ChunkRecord]], max_products: int = MAX_PRODUCTS_IN_CTX) -> str:
    by_pid: Dict[str, List[Tuple[float, ChunkRecord]]] = {}
    for score, rec in hits:
        by_pid.setdefault(rec.meta.product_id, []).append((score, rec))

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

def answer_with_hf(gen_pipeline, tokenizer, query: str, context: str) -> str:
    user_block = f"Question:\n{query}\n\nContext (top retrieved chunks):\n{context}\n\nAnswer:"
    prompt = format_chat_prompt(tokenizer, SYSTEM_INSTRUCTIONS, user_block)
    eos_id = tokenizer.eos_token_id
    out = gen_pipeline(
        prompt,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        temperature=GEN_TEMPERATURE,
        do_sample=True if GEN_TEMPERATURE > 0 else False,
        eos_token_id=eos_id,
        pad_token_id=tokenizer.pad_token_id or eos_id,
    )
    # pipeline returns list of dicts with 'generated_text'
    try:
        return out[0]["generated_text"]
    except Exception:
        return str(out)

# =========================
# Main
# =========================

def main():
    # 1) Load products and build chunks
    chunker = RecursiveChunker()
    all_chunks: List[ChunkRecord] = []
    count_products = 0

    print("[info] Scanning categories and loading products...")
    for category, prod in iter_products(RESULTS_DIR):
        try:
            text, meta = product_to_text(prod, category)
            chunks = chunk_product(text, meta, chunker)
            all_chunks.extend(chunks)
            count_products += 1
        except Exception as e:
            print(f"[warn] product skipped due to error: {e}", file=sys.stderr)

    if not all_chunks:
        print("[error] No chunks built. Ensure 'result/*/products.json' exist.", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Loaded {count_products} products, built {len(all_chunks)} chunks.")

    # 2) Build embedding index
    print(f"[info] Loading embedder: {EMBED_MODEL_NAME}")
    embedder = build_embedder(EMBED_MODEL_NAME)
    print("[info] Embedding chunks (cached) and building index...")
    index = build_index(embedder, all_chunks, EMBED_MODEL_NAME)

    # 3) Retrieve
    print("[info] Retrieving top chunks for the query...")
    hits = search(embedder, index, QUERY, EMBED_MODEL_NAME, top_k=TOP_K)

    # 4) Build concise context
    context = make_context_block(hits, max_products=MAX_PRODUCTS_IN_CTX)

    # 5) Generate answer with local HF model
    print(f"[info] Loading generator: {GEN_MODEL_NAME}")
    tok, gen = build_generator(GEN_MODEL_NAME)
    print("[info] Asking the model...")
    answer = answer_with_hf(gen, tok, QUERY, context)

    # 6) Output
    print("\n================ ANSWER ================\n")
    print(answer)

    print("\n================ DEBUG (Top 10 Retrieved) ================\n")
    for i, (score, rec) in enumerate(hits[:10], 1):
        m = rec.meta
        price_val = "NaN" if np.isnan(m.price_value) else f"{m.price_value:.0f}"
        print(
            f"{i:02d}. score={score:.4f} | {m.title[:80]} | ৳{price_val} | "
            f"rating={m.rating_avg} ({m.rating_count}) | {m.url}"
        )

if __name__ == "__main__":
    # Speed niceties
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
