#!/usr/bin/env python3
"""
RAG over scraped Daraz product JSONs (uses Chonkie for chunking)

Folder layout expected (example):
result/
  www_daraz_com_bd_kitchen_fixtures/
    products.json
  www_daraz_com_bd_shop_bedding_sets/
    products.json
  ...

Only products.json is read from each category folder. JSON is assumed to be
a list of product dicts; the schema may vary; code is defensive.

Key features
------------
- Chunking: Chonkie (TokenChunker or RecursiveChunker).
- Embeddings: sentence-transformers via Chonkie (all-MiniLM-L6-v2 by default).
- Vector index: FAISS (cosine) if available; otherwise a numpy brute-force fallback.
- Hybrid retrieval: semantic vector search + optional metadata filters.
- Answer synthesis: If OPENAI_API_KEY is set, can use OpenAI-compatible chat API.
                    Otherwise returns an evidence-based summary + top products.
- Persisted artifacts (by default under ./artifacts): index + chunk metadata.

Install (recommended minimal):
    pip install "chonkie[st]" numpy
For FAISS acceleration (optional):
    pip install faiss-cpu

If you want OpenAI synthesis (optional):
    pip install openai
    # and set env var: OPENAI_API_KEY=sk-...

Usage
-----
Build index:
    python rag_products.py build --data_dir result --artifacts_dir artifacts

Ask a question:
    python rag_products.py ask --artifacts_dir artifacts \
        --question "Cheapest kitchen faucet filter under 250 taka with 4+ rating?" \
        --top_k 8 --category kitchen_fixtures

Interactive shell:
    python rag_products.py interactive --artifacts_dir artifacts

Notes
-----
- Currency parsing is general; "৳ 172" becomes 172.0. Everything is treated as the
  same currency for filtering/sorting (no FX conversion).
- Ratings/sold are parsed if present; absent fields are defaulted.
- Category is inferred from the immediate subfolder of data_dir.

Author: (You)
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# --- Chonkie (chunking + embeddings) ---
try:
    # Chunkers
    from chonkie import TokenChunker, RecursiveChunker
    # Embeddings wrapper for sentence-transformers
    from chonkie import SentenceTransformerEmbeddings
except Exception as e:
    raise ImportError(
        "This script requires Chonkie. Install it via:\n"
        "    pip install chonkie\n"
        "For sentence-transformers embeddings:\n"
        "    pip install 'chonkie[st]'\n\n"
        f"Import error: {e}"
    )

# --- Optional FAISS for fast ANN ---
_FAISS_AVAILABLE = False
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

# --- Optional OpenAI-compatible client for synthesis ---
_OPENAI_AVAILABLE = False
try:
    # Newer OpenAI client style
    from openai import OpenAI  # type: ignore
    _OPENAI_AVAILABLE = True
except Exception:
    try:
        # Older openai import path (fallback)
        import openai  # type: ignore
        _OPENAI_AVAILABLE = True
    except Exception:
        _OPENAI_AVAILABLE = False


# ---------------------------- Utilities ---------------------------------
def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def _parse_float_from_str(value: Any) -> Optional[float]:
    """Extract a float from arbitrary display strings like '৳ 172', '4.3', 'Ratings 73' etc."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    if not nums:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None


def _safe_get(d: Dict, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _normalize_price(product: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (price_value, original_price_value, discount_percent)
    using the best available fields.
    """
    display_price = product.get("product_price")  # e.g. "৳ 172"
    detail_price = _safe_get(product, ["detail", "price", "value"], None)
    detail_orig = _safe_get(product, ["detail", "price", "original_value"], None)
    detail_disc = _safe_get(product, ["detail", "price", "discount_percent"], None)

    price_val = _coalesce(_parse_float_from_str(display_price), _parse_float_from_str(detail_price))
    orig_val = _coalesce(_parse_float_from_str(detail_orig), None)
    disc = _coalesce(_parse_float_from_str(detail_disc), None)

    # If original missing but both price+discount present, infer:
    if orig_val is None and price_val is not None and disc:
        try:
            orig_val = price_val / (1 - disc / 100.0)
        except ZeroDivisionError:
            pass

    # If discount missing but price & original present, infer:
    if disc is None and price_val is not None and orig_val and orig_val > 0:
        disc = (1.0 - (price_val / orig_val)) * 100.0

    return price_val, orig_val, disc


def _normalize_rating(product: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    avg = _safe_get(product, ["detail", "rating", "average"], None)
    count = _safe_get(product, ["detail", "rating", "count"], None)
    return _parse_float_from_str(avg), _parse_float_from_str(count)


def _normalize_sold(product: Dict[str, Any]) -> Optional[float]:
    loc = product.get("location")  # often like "536 sold"
    return _parse_float_from_str(loc)


def _first_non_empty(*vals) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _extract_text(product: Dict[str, Any]) -> str:
    # Prefer explicit description_text, fallback to whats_in_the_box or raw_text if provided
    desc = _safe_get(product, ["detail", "details", "description_text"], None)
    if not desc:
        desc = _safe_get(product, ["detail", "details", "raw_text"], None)
    if not desc:
        desc = ""
    return str(desc)


def _extract_brand(product: Dict[str, Any]) -> Optional[str]:
    brand = _safe_get(product, ["detail", "brand"], None)
    if not brand:
        brand = product.get("brand")
    return brand


def _extract_name(product: Dict[str, Any]) -> str:
    return _first_non_empty(
        product.get("product_title"),
        _safe_get(product, ["detail", "name"], None),
        "Unknown Product",
    )


def _extract_url(product: Dict[str, Any]) -> Optional[str]:
    return _first_non_empty(
        product.get("detail_url"),
        _safe_get(product, ["detail", "url"], None),
        product.get("product_detail_url"),
    )


def _extract_image(product: Dict[str, Any]) -> Optional[str]:
    img = product.get("image_url")
    if not img:
        imgs = _safe_get(product, ["detail", "images"], [])
        if isinstance(imgs, list) and imgs:
            img = imgs[0]
    return img


def _extract_colors(product: Dict[str, Any]) -> List[str]:
    cols = _safe_get(product, ["detail", "colors"], [])
    return list(cols) if isinstance(cols, list) else []


def _extract_seller(product: Dict[str, Any]) -> Optional[str]:
    return _safe_get(product, ["detail", "seller", "name"], None)


def robust_product_to_text(product: Dict[str, Any], category: str) -> Tuple[str, Dict[str, Any]]:
    """
    Create a normalized textual representation plus a rich metadata dict.
    """
    name = _extract_name(product)
    brand = _extract_brand(product)
    url = _extract_url(product)
    image = _extract_image(product)

    price, orig_price, disc_pct = _normalize_price(product)
    rating_avg, rating_count = _normalize_rating(product)
    sold = _normalize_sold(product)
    colors = _extract_colors(product)
    seller = _extract_seller(product)

    data_item_id = product.get("data_item_id") or product.get("data_sku_simple") or product.get("data_tracking")

    # Description and variants
    desc = _extract_text(product)

    var_lines = []
    variants = _safe_get(product, ["detail", "variants"], [])
    if isinstance(variants, list):
        for i, v in enumerate(variants):
            vcolor = v.get("color") or ""
            vprice = _safe_get(v, ["price", "value"], None)
            vprice_display = _safe_get(v, ["price", "display"], None)
            v_images = v.get("images", [])
            var_lines.append(f"Variant {i+1}: color={vcolor or '-'} price={vprice or vprice_display or '-'} images={len(v_images)}")

    key_specs = _safe_get(product, ["detail", "details", "specifications"], [])
    spec_lines = []
    if isinstance(key_specs, list):
        for s in key_specs:
            if isinstance(s, dict):
                k = s.get("key") or s.get("name") or ""
                v = s.get("value") or s.get("val") or ""
                if k or v:
                    spec_lines.append(f"{k}: {v}")
            else:
                spec_lines.append(str(s))

    meta = {
        "category": category,
        "product_id": str(data_item_id) if data_item_id is not None else None,
        "name": name,
        "brand": brand,
        "url": url,
        "image": image,
        "price": price,
        "original_price": orig_price,
        "discount_percent": disc_pct,
        "rating_avg": rating_avg,
        "rating_count": rating_count,
        "sold_count": sold,
        "colors": colors,
        "seller": seller,
    }

    # Create the document text for chunking
    lines = [
        f"Category: {category}",
        f"Name: {name}",
        f"Brand: {brand or 'Unknown'}",
        f"Price: {price if price is not None else 'Unknown'}",
        f"Original Price: {orig_price if orig_price is not None else 'Unknown'}",
        f"Discount %: {round(disc_pct, 1) if disc_pct is not None else 'Unknown'}",
        f"Rating: {rating_avg if rating_avg is not None else 'Unknown'} ({rating_count if rating_count is not None else '0'} reviews)",
        f"Sold: {int(sold) if sold is not None else 'Unknown'}",
        f"Seller: {seller or 'Unknown'}",
    ]

    if colors:
        lines.append(f"Colors: {', '.join(colors)}")

    if spec_lines:
        lines.append("Specifications:")
        lines.extend([f"- {x}" for x in spec_lines])

    if var_lines:
        lines.append("Variants:")
        lines.extend([f"- {x}" for x in var_lines])

    if desc.strip():
        lines.append("Description:")
        lines.append(desc.strip())

    text = "\n".join(lines)
    return text, meta


# ----------------------- Chunking + Embeddings ---------------------------

class ChunkerWrapper:
    def __init__(self, kind: str = "token", chunk_size: int = 512):
        """
        kind: 'token' (GPT2-ish tokens) or 'recursive'
        chunk_size: maximum tokens per chunk
        """
        if kind not in {"token", "recursive"}:
            raise ValueError("chunker kind must be one of: token, recursive")
        self.kind = kind
        self.chunk_size = int(chunk_size)

        if self.kind == "token":
            self._chunker = TokenChunker(chunk_size=self.chunk_size)  # defaults to GPT2 tokenizer
        else:
            self._chunker = RecursiveChunker(chunk_size=self.chunk_size)

    def chunk(self, text: str):
        # Returns a list of objects with .text, .token_count, and char indices
        return self._chunker(text)


class EmbeddingsWrapper:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Requires: pip install "chonkie[st]"
        self.model_name = model_name
        self._emb = SentenceTransformerEmbeddings(model_name)

        # Estimate dimension: embed a tiny string
        vec = self._emb.embed(" ")
        self.dim = int(len(vec))

    def embed(self, text: str) -> np.ndarray:
        v = self._emb.embed(text)
        return np.asarray(v, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        vs = self._emb.embed_batch(texts)
        return np.asarray(vs, dtype=np.float32)


# ------------------------------ Index -----------------------------------

class VectorIndex:
    """
    Cosine-similarity ANN index. Uses FAISS if available, else numpy brute-force.
    """
    def __init__(self, dim: int):
        self.dim = int(dim)
        self._use_faiss = _FAISS_AVAILABLE
        self._index = None  # faiss index or np.ndarray
        self._matrix = None

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def build(self, vecs: np.ndarray):
        vecs = vecs.astype(np.float32)
        vecs = self._l2_normalize(vecs)

        if self._use_faiss:
            index = faiss.IndexFlatIP(self.dim)  # cosine via dot on normalized vectors
            index.add(vecs)
            self._index = index
        else:
            self._matrix = vecs

    def search(self, q: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        q = q.reshape(1, -1).astype(np.float32)
        q = self._l2_normalize(q)
        if self._use_faiss:
            D, I = self._index.search(q, top_k)  # type: ignore
            return D[0], I[0]
        else:
            # cosine ~ dot on normalized vectors
            sims = (self._matrix @ q.T).ravel()
            idx = np.argpartition(-sims, range(min(top_k, len(sims))))[:top_k]
            idx = idx[np.argsort(-sims[idx])]
            return sims[idx], idx

    # Persistence
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._use_faiss:
            faiss.write_index(self._index, str(path))  # type: ignore
        else:
            np.save(str(path), self._matrix)

    @classmethod
    def load(cls, path: Path, dim: int) -> "VectorIndex":
        vi = cls(dim)
        if _FAISS_AVAILABLE and path.exists() and path.suffix in {".faiss", ".index", ".idx"}:
            try:
                vi._index = faiss.read_index(str(path))
                vi._use_faiss = True
                return vi
            except Exception:
                pass

        # Try numpy fallback
        npy_path = path if path.suffix == ".npy" else path.with_suffix(".npy")
        if npy_path.exists():
            vi._use_faiss = False
            vi._matrix = np.load(str(npy_path))
            return vi

        raise FileNotFoundError(f"Index file not found: {path} or {npy_path}")


# -------------------------- Corpus builder -------------------------------

@dataclasses.dataclass
class ChunkRecord:
    text: str
    meta: Dict[str, Any]


def build_corpus(
    data_dir: Path,
    chunker: ChunkerWrapper,
) -> List[ChunkRecord]:
    """
    Walk all categories in data_dir, read products.json where present,
    and produce chunked documents with metadata.
    """
    records: List[ChunkRecord] = []

    # any immediate child directory of data_dir
    for cat_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        category = cat_dir.name.replace("www_daraz_com_bd_", "")
        products_file = cat_dir / "products.json"
        if not products_file.exists():
            continue

        try:
            with open(products_file, "r", encoding="utf-8") as f:
                arr = json.load(f)
            if not isinstance(arr, list):
                logging.warning("Unexpected JSON (not a list): %s", products_file)
                continue
        except Exception as e:
            logging.warning("Failed to read %s: %s", products_file, e)
            continue

        for product in arr:
            try:
                text, meta = robust_product_to_text(product, category)
                # Skip empty docs
                if not text.strip():
                    continue

                chunks = chunker.chunk(text)
                for i, ch in enumerate(chunks):
                    cm = dict(meta)
                    cm["chunk_index"] = i
                    cm["char_start"] = getattr(ch, "start_index", None)
                    cm["char_end"] = getattr(ch, "end_index", None)
                    cm["token_count"] = getattr(ch, "token_count", None)
                    records.append(ChunkRecord(text=ch.text, meta=cm))
            except Exception as e:
                logging.debug("Skipping a product due to parse error: %s", e)
                continue

    return records


# ------------------------------ RAG core --------------------------------

class ProductRAG:
    def __init__(
        self,
        artifacts_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunker_kind: str = "token",
        chunk_size: int = 512,
    ):
        self.artifacts_dir = artifacts_dir
        self.embedding_model = embedding_model
        self.emb = EmbeddingsWrapper(embedding_model)
        self.chunker = ChunkerWrapper(kind=chunker_kind, chunk_size=chunk_size)

        self.index_path = self.artifacts_dir / "products.index.faiss"
        self.meta_path = self.artifacts_dir / "chunks.meta.json"
        self.dim_path = self.artifacts_dir / "emb.dim.txt"

        self.index: Optional[VectorIndex] = None
        self.metadatas: List[Dict[str, Any]] = []

    # ---------- Build ----------
    def build(self, data_dir: Path):
        logging.info("Building corpus from %s ...", data_dir)
        corpus = build_corpus(data_dir, self.chunker)
        logging.info("Found %d chunks", len(corpus))

        if not corpus:
            raise RuntimeError("No chunks found. Check your data_dir structure.")

        texts = [c.text for c in corpus]
        metas = [c.meta for c in corpus]

        logging.info("Embedding %d chunks with %s ...", len(texts), self.embedding_model)
        X = self.emb.embed_batch(texts)
        # Build ANN index
        index = VectorIndex(self.emb.dim)
        index.build(X)

        # Persist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        index.save(self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metas, f, ensure_ascii=False)

        with open(self.dim_path, "w", encoding="utf-8") as f:
            f.write(str(self.emb.dim))

        logging.info("Saved index to %s and metadata to %s", self.index_path, self.meta_path)

    # ---------- Load ----------
    def load(self):
        with open(self.dim_path, "r", encoding="utf-8") as f:
            dim = int(f.read().strip())
        self.index = VectorIndex.load(self.index_path, dim)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadatas = json.load(f)
        logging.info("Loaded index (%s) and %d metadata rows", "FAISS" if _FAISS_AVAILABLE else "numpy", len(self.metadatas))

    # ---------- Search ----------
    def _apply_filters(self, idxs: List[int], filters: Dict[str, Any]) -> List[int]:
        if not filters:
            return idxs

        def ok(meta: Dict[str, Any]) -> bool:
            # Available filters: category (substring), min_price, max_price, brand (substring),
            # min_rating, seller (substring)
            if "category" in filters and filters["category"]:
                c = str(filters["category"]).lower()
                if c not in str(meta.get("category", "")).lower():
                    return False

            if "brand" in filters and filters["brand"]:
                b = str(filters["brand"]).lower()
                if b not in str(meta.get("brand", "")).lower():
                    return False

            if "seller" in filters and filters["seller"]:
                s = str(filters["seller"]).lower()
                if s not in str(meta.get("seller", "")).lower():
                    return False

            p = meta.get("price", None)
            if p is not None:
                if filters.get("min_price") is not None and p < float(filters["min_price"]):
                    return False
                if filters.get("max_price") is not None and p > float(filters["max_price"]):
                    return False
            else:
                # If no price info and user asked for price bounds, discard
                if "min_price" in filters or "max_price" in filters:
                    return False

            r = meta.get("rating_avg", None)
            if r is not None and filters.get("min_rating") is not None and r < float(filters["min_rating"]):
                return False
            elif r is None and "min_rating" in filters:
                return False

            return True

        out = [i for i in idxs if ok(self.metadatas[i])]
        return out

    def search(self, question: str, top_k: int = 8, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[float, int]]:
        assert self.index is not None, "Index not loaded. Call .load() first."
        q = self.emb.embed(question)
        scores, idxs = self.index.search(q, top_k * 6)  # fetch more, then filter + dedupe

        idxs = idxs.tolist()
        scores = scores.tolist()

        # Filter
        filt_idxs = self._apply_filters(idxs, filters or {})

        # Keep order by score
        filt_set = set(filt_idxs)
        scored = [(s, i) for s, i in zip(scores, idxs) if i in filt_set]

        # Deduplicate by product (keep best chunk per product)
        seen = set()
        dedup: List[Tuple[float, int]] = []
        for s, i in scored:
            pid = self.metadatas[i].get("product_id") or (self.metadatas[i].get("name"), self.metadatas[i].get("url"))
            if pid in seen:
                continue
            seen.add(pid)
            dedup.append((s, i))
            if len(dedup) >= top_k:
                break

        return dedup

    # ---------- Answer synthesis ----------
    def synthesize_answer(self, question: str, hits: List[Tuple[float, int]], use_openai: bool = True, model: str = "gpt-4o-mini") -> str:
        """
        Build an answer from retrieved chunks. If OpenAI key available and use_openai==True,
        call the chat model with a grounded prompt. Otherwise return a textual summary.
        """
        # Assemble contexts
        contexts = []
        for rank, (score, idx) in enumerate(hits, 1):
            m = self.metadatas[idx]
            # Build a concise one-liner per product
            line = self._format_product_line(m)
            contexts.append(f"[{rank}] {line} (score={score:.3f})")

        context_block = "\n".join(contexts) if contexts else "No matching products found."

        if use_openai and _OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                prompt = f"""
You are a helpful shopping assistant. Use ONLY the products shown below ("Context") to answer the user question.
Prefer items that satisfy numeric constraints. Return a concise, well-structured answer with a short rationale.
Always include a final "Top picks" list with: Name | Price | Rating | Category | URL.

Question: {question}

Context:
{context_block}

Rules:
- If prices are 'Unknown', avoid those items when user asked for price filters.
- If nothing matches, say so explicitly and suggest the closest options from Context.
"""
                # New-style SDK
                if 'OpenAI' in globals():
                    client = OpenAI()
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a precise, grounded shopping assistant."},
                            {"role": "user", "content": prompt.strip()},
                        ],
                        temperature=0.2,
                        max_tokens=400,
                    )
                    return resp.choices[0].message.content.strip()
                else:
                    # Older SDK fallback
                    openai.api_key = os.getenv("OPENAI_API_KEY")  # type: ignore
                    resp = openai.ChatCompletion.create(  # type: ignore
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a precise, grounded shopping assistant."},
                            {"role": "user", "content": prompt.strip()},
                        ],
                        temperature=0.2,
                        max_tokens=400,
                    )
                    return resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logging.warning("OpenAI synthesis failed (%s). Falling back to heuristic summary.", e)

        # Fallback: heuristic summary
        if not hits:
            return f"I couldn't find matching products for: {question}\n\nTry relaxing filters or changing keywords."

        lines = ["Answer (heuristic summary):"]
        # Build a simple explanation then list the top picks
        lines.append(f"Found {len(hits)} relevant product(s). Here are the top matches:")
        for rank, (score, idx) in enumerate(hits, 1):
            m = self.metadatas[idx]
            lines.append(f"{rank}. {self._format_product_line(m)} (similarity={score:.3f})")
        return "\n".join(lines)

    @staticmethod
    def _format_product_line(m: Dict[str, Any]) -> str:
        name = m.get("name") or "Unknown"
        price = m.get("price")
        price_str = f"{price:.0f}" if isinstance(price, (int, float)) else "Unknown"
        rating = m.get("rating_avg")
        rating_str = f"{rating:.1f}" if isinstance(rating, (int, float)) else "Unknown"
        category = m.get("category") or "-"
        url = m.get("url") or "-"
        brand = m.get("brand") or "-"
        return f"{name} (brand={brand}) | ৳ {price_str} | {rating_str}★ | {category} | {url}"


# ------------------------------ CLI -------------------------------------

def cmd_build(args):
    rag = ProductRAG(
        artifacts_dir=Path(args.artifacts_dir),
        embedding_model=args.embedding_model,
        chunker_kind=args.chunker,
        chunk_size=args.chunk_size,
    )
    rag.build(Path(args.data_dir))


def cmd_ask(args):
    rag = ProductRAG(
        artifacts_dir=Path(args.artifacts_dir),
        embedding_model=args.embedding_model,
        chunker_kind=args.chunker,
        chunk_size=args.chunk_size,
    )
    rag.load()

    filters = dict(
        category=args.category,
        brand=args.brand,
        seller=args.seller,
        min_price=args.min_price,
        max_price=args.max_price,
        min_rating=args.min_rating,
    )

    hits = rag.search(args.question, top_k=args.top_k, filters=filters)
    answer = rag.synthesize_answer(args.question, hits, use_openai=not args.no_llm, model=args.model)

    print(answer)
    print("\nSources:")
    for rank, (score, idx) in enumerate(hits, 1):
        m = rag.metadatas[idx]
        url = m.get("url") or "-"
        print(f"[{rank}] {m.get('name')} — {url}")


def cmd_interactive(args):
    rag = ProductRAG(
        artifacts_dir=Path(args.artifacts_dir),
        embedding_model=args.embedding_model,
        chunker_kind=args.chunker,
        chunk_size=args.chunk_size,
    )
    if not (Path(args.artifacts_dir) / "chunks.meta.json").exists():
        print("No index found. Please run the 'build' command first.")
        return
    rag.load()

    print("Interactive Product RAG. Type your question, or 'exit' to quit.")
    while True:
        try:
            q = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q or q.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        hits = rag.search(q, top_k=args.top_k, filters=dict(
            category=args.category,
            brand=args.brand,
            seller=args.seller,
            min_price=args.min_price,
            max_price=args.max_price,
            min_rating=args.min_rating,
        ))
        ans = rag.synthesize_answer(q, hits, use_openai=not args.no_llm, model=args.model)
        print("\n" + ans)
        print("\nSources:")
        for rank, (score, idx) in enumerate(hits, 1):
            m = rag.metadatas[idx]
            print(f"[{rank}] {m.get('name')} — {m.get('url')}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="RAG QA over scraped Daraz products (uses Chonkie for chunking).")
    parser.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Build
    pb = sub.add_parser("build", help="Build the vector index from products.json files.")
    pb.add_argument("--data_dir", default="result", help="Root directory that contains category subfolders.")
    pb.add_argument("--artifacts_dir", default="artifacts", help="Where to persist index + metadata.")
    pb.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Sentence-Transformers model for embeddings.")
    pb.add_argument("--chunker", default="token", choices=["token", "recursive"], help="Chonkie chunker to use.")
    pb.add_argument("--chunk_size", type=int, default=512, help="Max tokens per chunk for Chonkie.")
    pb.set_defaults(func=cmd_build)

    # Ask
    pa = sub.add_parser("ask", help="Ask a question over the indexed products.")
    pa.add_argument("--artifacts_dir", default="artifacts", help="Where the index + metadata were saved.")
    pa.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Sentence-Transformers model name.")
    pa.add_argument("--chunker", default="token", choices=["token", "recursive"], help="Must match what was used to build.")
    pa.add_argument("--chunk_size", type=int, default=512)
    pa.add_argument("--question", required=True, help="Your question, e.g., 'best faucet filter under 250 taka'.")
    pa.add_argument("--top_k", type=int, default=8)
    pa.add_argument("--category", default=None, help="Substring match on category folder name.")
    pa.add_argument("--brand", default=None, help="Substring match on brand.")
    pa.add_argument("--seller", default=None, help="Substring match on seller name.")
    pa.add_argument("--min_price", type=float, default=None)
    pa.add_argument("--max_price", type=float, default=None)
    pa.add_argument("--min_rating", type=float, default=None)
    pa.add_argument("--no_llm", action="store_true", help="Disable LLM synthesis (heuristic summary only).")
    pa.add_argument("--model", default="gpt-4o-mini", help="OpenAI-compatible chat model name if LLM synthesis enabled.")
    pa.set_defaults(func=cmd_ask)

    # Interactive
    pi = sub.add_parser("interactive", help="Interactive QA session.")
    pi.add_argument("--artifacts_dir", default="artifacts")
    pi.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    pi.add_argument("--chunker", default="token", choices=["token", "recursive"])
    pi.add_argument("--chunk_size", type=int, default=512)
    pi.add_argument("--top_k", type=int, default=8)
    pi.add_argument("--category", default=None)
    pi.add_argument("--brand", default=None)
    pi.add_argument("--seller", default=None)
    pi.add_argument("--min_price", type=float, default=None)
    pi.add_argument("--max_price", type=float, default=None)
    pi.add_argument("--min_rating", type=float, default=None)
    pi.add_argument("--no_llm", action="store_true")
    pi.add_argument("--model", default="gpt-4o-mini")
    pi.set_defaults(func=cmd_interactive)

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()