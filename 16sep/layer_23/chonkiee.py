#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG over Daraz-scraped product JSON files (result/<category>/products.json)

- Uses Chonkie for chunking
- Embeds chunks with sentence-transformers (all-MiniLM-L6-v2)
- Searches with FAISS (or NumPy fallback)
- Generates with a free local LLM (FLAN-T5 base)
- No CLI args: set QUERY and ROOT_DIR below and run:  python rag_products.py

Swap to OpenAI later:
- Implement OpenAI in LLMProvider.generate(), or add a new provider class and switch in build_llm().

Tested with: Python 3.10+
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---- RAG CONFIG (edit as you like) ------------------------------------------
ROOT_DIR = Path("result")                         # your scraped root folder
QUERY = "Recommend 3 faucet filters under ৳250 with high ratings and many sales."
TOP_K = 8                                         # how many chunks to retrieve
CHARS_LIMIT_CONTEXT = 1800                        # flan-t5-base has ~512 token limit
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
# -----------------------------------------------------------------------------

# ---- Imports (lazy-fail with helpful messages) -------------------------------
try:
    from chonkie import RecursiveChunker   # https://github.com/chonkie-inc/chonkie
except Exception as e:
    print("ERROR: chonkie is required. Install with: pip install chonkie", file=sys.stderr)
    raise

try:
    import numpy as np
except Exception as e:
    print("ERROR: numpy is required. Install with: pip install numpy", file=sys.stderr)
    raise

# sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    print("ERROR: sentence-transformers is required. Install with: pip install sentence-transformers", file=sys.stderr)
    raise

# transformers for free local LLM (FLAN-T5)
try:
    import torch
    from transformers import pipeline
except Exception:
    print("ERROR: transformers/torch are required. Install with: pip install transformers accelerate torch", file=sys.stderr)
    raise

# FAISS for fast ANN (optional but recommended)
try:
    import faiss  # type: ignore
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False


# ---- Utility dataclasses -----------------------------------------------------
@dataclass
class ChunkRecord:
    text: str
    product_id: str
    product_title: str
    category: str
    url: str
    brand: Optional[str] = None
    price_display: Optional[str] = None
    rating_avg: Optional[float] = None
    rating_count: Optional[int] = None


# ---- I/O: Load JSON files ----------------------------------------------------
def find_products_jsons(root: Path) -> List[Path]:
    """
    Return a list of paths to products.json in result/<category>/products.json
    """
    paths: List[Path] = []
    if not root.exists():
        print(f"WARNING: {root} does not exist.", file=sys.stderr)
        return paths

    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            pj = sub / "products.json"
            if pj.exists():
                paths.append(pj)
    return paths


def load_products(products_json: Path) -> Iterable[Dict[str, Any]]:
    """
    Yield each product dict from products.json (list at top-level).
    Continue on JSON errors (robust to imperfect files).
    """
    try:
        with products_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            else:
                print(f"WARNING: {products_json} is not a list.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR reading {products_json}: {e}", file=sys.stderr)


# ---- Normalization helpers ---------------------------------------------------
_WS = re.compile(r"\s+")

def squash_ws(s: str | None) -> str:
    if not s:
        return ""
    return _WS.sub(" ", str(s)).strip()

def to_https(url: Optional[str]) -> str:
    if not url:
        return ""
    url = url.strip()
    if url.startswith("//"):
        return "https:" + url
    if not url.startswith("http"):
        # fall back to https if relative (rare in your data since detail_url is absolute)
        return "https://" + url.lstrip("/")
    return url

def safe_get(d: Dict[str, Any], *keys, default=None) -> Any:
    cur: Any = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def list_preview(values: List[Any], limit: int = 5) -> str:
    vs = [squash_ws(str(v)) for v in values if str(v).strip()]
    if len(vs) <= limit:
        return ", ".join(vs)
    return ", ".join(vs[:limit]) + f" …(+{len(vs)-limit} more)"


# ---- Product -> Document text ------------------------------------------------
def product_to_text_and_meta(prod: Dict[str, Any], category_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Build a single textual document per product that captures the important fields:
    title, brand, price (display/value/original/discount), rating, sold, seller,
    description, specs, return/warranty, images (count), and all variants (color/size/price).
    """
    item_id = str(prod.get("data_item_id") or prod.get("data_sku_simple") or "")
    title = squash_ws(prod.get("product_title") or safe_get(prod, "detail", "name") or "")
    url = to_https(prod.get("detail_url") or prod.get("product_detail_url") or safe_get(prod, "detail", "url") or "")
    brand = squash_ws(safe_get(prod, "detail", "brand"))
    price_display = squash_ws(safe_get(prod, "detail", "price", "display") or prod.get("product_price"))
    price_value = safe_get(prod, "detail", "price", "value")
    price_original = safe_get(prod, "detail", "price", "original_value")
    discount_pct = safe_get(prod, "detail", "price", "discount_percent")
    rating_avg = safe_get(prod, "detail", "rating", "average")
    rating_cnt = safe_get(prod, "detail", "rating", "count")
    sold_str = squash_ws(prod.get("location"))  # e.g., "536 sold"
    seller_name = squash_ws(safe_get(prod, "detail", "seller", "name"))
    seller_metrics = safe_get(prod, "detail", "seller", "metrics") or {}
    description = squash_ws(safe_get(prod, "detail", "details", "description_text"))
    highlights = safe_get(prod, "detail", "details", "highlights") or []
    return_and_warranty = safe_get(prod, "detail", "return_and_warranty") or []
    images = safe_get(prod, "detail", "images") or []
    colors = safe_get(prod, "detail", "colors") or []
    sizes = safe_get(prod, "detail", "sizes") or []
    variants = safe_get(prod, "detail", "variants") or []

    # Variants summary
    variant_lines: List[str] = []
    if isinstance(variants, list):
        for v in variants:
            vcolor = squash_ws(v.get("color"))
            vprice_disp = squash_ws(safe_get(v, "price", "display"))
            vsizes = v.get("sizes") or []
            variant_lines.append(
                f"- variant color: {vcolor or 'N/A'} | price: {vprice_disp or 'N/A'} | sizes: {', '.join(vsizes) if vsizes else 'N/A'}"
            )

    # Seller metrics text
    seller_metrics_txt = "; ".join([f"{k}: {squash_ws(v)}" for k, v in seller_metrics.items()]) if seller_metrics else ""

    # Build document text (balanced for retrieval)
    parts = [
        f"CATEGORY: {category_name}",
        f"ITEM_ID: {item_id}",
        f"TITLE: {title}",
        f"BRAND: {brand or 'No Brand'}",
        f"PRICE: {price_display or 'N/A'} (value: {price_value}, original: {price_original}, discount%: {discount_pct})",
        f"RATING: {rating_avg if rating_avg is not None else 'N/A'} ({rating_cnt if rating_cnt is not None else 0} ratings)",
        f"SALES: {sold_str or 'N/A'}",
        f"SELLER: {seller_name or 'N/A'}; {seller_metrics_txt or ''}".strip(),
        f"COLORS: {list_preview(colors) if isinstance(colors, list) else colors or 'N/A'}",
        f"SIZES: {list_preview(sizes) if isinstance(sizes, list) and sizes else 'N/A'}",
        f"IMAGES: {len(images)} found",
    ]

    if highlights:
        parts.append("HIGHLIGHTS: " + list_preview(highlights, limit=12))
    if description:
        parts.append("DESCRIPTION: " + description)

    if variant_lines:
        parts.append("VARIANTS:\n" + "\n".join(variant_lines))

    parts.append(f"URL: {url}")

    text = "\n".join([p for p in parts if p and p.strip()])

    meta = dict(
        product_id=item_id,
        product_title=title,
        category=category_name,
        url=url,
        brand=brand or None,
        price_display=price_display or None,
        rating_avg=float(rating_avg) if isinstance(rating_avg, (int, float)) else None,
        rating_count=int(rating_cnt) if isinstance(rating_cnt, (int, float)) else None,
    )
    return text, meta


# ---- Chunking (Chonkie) ------------------------------------------------------
def chunk_product_text(text: str, chunker: RecursiveChunker) -> List[str]:
    """
    Use Chonkie's RecursiveChunker to produce chunks. We keep each chunk as a clean string.
    """
    chunks = chunker(text)  # per Chonkie README, chunkers are callable
    out: List[str] = []
    for ch in chunks:
        # ch.text is the text; optional: keep token count ch.token_count
        out.append(squash_ws(ch.text))
    return out


# ---- Embeddings --------------------------------------------------------------
class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        # We'll use cosine similarity with normalized vectors
    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

# ---- Vector index ------------------------------------------------------------
class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.use_faiss = _HAVE_FAISS
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
    def build(self, embeddings: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.embeddings)
        else:
            # Will use NumPy dot as fallback
            self.index = None
    def search(self, qvec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (scores, indices)
        - scores: shape (top_k,)
        - indices: shape (top_k,)
        """
        if qvec.ndim == 1:
            qvec = qvec[None, :]
        if self.use_faiss and self.index is not None:
            sims, idxs = self.index.search(qvec.astype(np.float32), top_k)
            return sims[0], idxs[0]
        # cosine with normalized vectors => dot product
        assert self.embeddings is not None
        sims = (self.embeddings @ qvec.T).ravel()
        idxs = np.argsort(-sims)[:top_k]
        return sims[idxs], idxs


# ---- Free LLM (local) + abstraction for later OpenAI swap --------------------
class LLMProvider:
    """
    Minimal abstraction to swap the generator later.
    """
    def __init__(self, model_name: str = GEN_MODEL_NAME):
        device = 0 if torch.cuda.is_available() else -1
        # FLAN-T5 is seq2seq; pipeline handles tokenizer/model for us
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name,
            device=device,
        )

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0 else 1.0,
            num_return_sequences=1,
            truncation=True,
        )
        return out[0]["generated_text"]

    # --- To switch to OpenAI later:
    # def generate(self, prompt: str, ...):
    #     import openai
    #     client = openai.OpenAI(api_key=...)
    #     resp = client.chat.completions.create( ... )
    #     return resp.choices[0].message.content


def build_llm() -> LLMProvider:
    return LLMProvider(GEN_MODEL_NAME)


# ---- Retrieval pipeline ------------------------------------------------------
def pack_context(chunks: List[ChunkRecord], limit_chars: int = CHARS_LIMIT_CONTEXT) -> str:
    """
    Join chunk texts with separators until we hit a safe character budget for the generator.
    """
    buf: List[str] = []
    total = 0
    for c in chunks:
        piece = c.text.strip()
        if not piece:
            continue
        # add a header to preserve product boundaries for the LLM
        header = f"\n---\nTITLE: {c.product_title}\nCATEGORY: {c.category}\nURL: {c.url}\n"
        add = header + piece
        if total + len(add) > limit_chars and buf:
            break
        buf.append(add)
        total += len(add)
    return "".join(buf).strip()


def make_prompt(query: str, context: str) -> str:
    """
    Compact instruction tuned for product Q&A with price/rating constraints.
    """
    return textwrap.dedent(f"""
    You are a helpful e-commerce product analyst. Answer the QUESTION using only the CONTEXT.
    If something isn't in the context, say you don't know. Prefer concise bullet points listing product name, price, rating, seller, and the URL.
    Use the currency symbols as shown (e.g., ৳). If asking for "top" or "best", justify briefly using rating/sales/discount.

    CONTEXT:
    {context}

    QUESTION: {query}

    ANSWER:
    """).strip()


# ---- Main build + query ------------------------------------------------------
def build_corpus(root: Path) -> Tuple[List[str], List[ChunkRecord]]:
    """
    1) For each product => text + meta
    2) Chunk with Chonkie
    3) Return chunk_texts + chunk_meta aligned
    """
    chunker = RecursiveChunker()
    chunk_texts: List[str] = []
    chunk_meta: List[ChunkRecord] = []

    product_count = 0
    for pj in find_products_jsons(root):
        category_name = pj.parent.name
        for prod in load_products(pj):
            product_count += 1
            text, meta = product_to_text_and_meta(prod, category_name)
            # Chunk
            pieces = chunk_product_text(text, chunker)
            # Capture aligned
            for p in pieces:
                chunk_texts.append(p)
                chunk_meta.append(ChunkRecord(
                    text=p,
                    product_id=meta["product_id"] or "",
                    product_title=meta["product_title"] or "",
                    category=meta["category"] or category_name,
                    url=meta["url"] or "",
                    brand=meta.get("brand"),
                    price_display=meta.get("price_display"),
                    rating_avg=meta.get("rating_avg"),
                    rating_count=meta.get("rating_count"),
                ))

    print(f"[build_corpus] categories={len(set(cm.category for cm in chunk_meta))} "
          f"products≈{product_count} chunks={len(chunk_texts)}")
    return chunk_texts, chunk_meta


def retrieve(query: str,
             embedder: Embedder,
             vindex: VectorIndex,
             chunk_texts: List[str],
             chunk_meta: List[ChunkRecord],
             top_k: int) -> List[ChunkRecord]:
    qvec = embedder.encode([query])[0]
    _, idxs = vindex.search(qvec, top_k=top_k)
    return [chunk_meta[i] for i in idxs.tolist()]


def answer_query(query: str, root: Path) -> Tuple[str, List[ChunkRecord]]:
    # Build corpus (chunks + meta)
    chunk_texts, chunk_meta = build_corpus(root)

    if not chunk_texts:
        return "No chunks found (did you put the 'result' folder next to this script?).", []

    # Embed all chunks
    embedder = Embedder(EMBED_MODEL_NAME)
    mat = embedder.encode(chunk_texts)
    dim = mat.shape[1]

    # Build index
    vindex = VectorIndex(dim)
    vindex.build(mat)

    # Retrieve
    top_chunks = retrieve(query, embedder, vindex, chunk_texts, chunk_meta, TOP_K)

    # Pack context and ask LLM
    context = pack_context(top_chunks, limit_chars=CHARS_LIMIT_CONTEXT)
    llm = build_llm()
    prompt = make_prompt(query, context)
    generated = llm.generate(prompt, max_new_tokens=320, temperature=0.2)
    return generated.strip(), top_chunks


# ---- Script entry ------------------------------------------------------------
def main():
    print(f"QUERY: {QUERY}\n")
    answer, used_chunks = answer_query(QUERY, ROOT_DIR)
    print("\n=== ANSWER ===\n")
    print(answer)
    print("\n=== SOURCES USED ===")
    seen = set()
    for c in used_chunks:
        key = (c.product_id, c.url)
        if key in seen:
            continue
        seen.add(key)
        print(f"- {c.product_title or '[No title]'}  |  {c.category}  |  {c.url}")

if __name__ == "__main__":
    main()
