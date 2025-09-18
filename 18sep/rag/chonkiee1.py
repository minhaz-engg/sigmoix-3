#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG over Daraz product dumps (products.json per category).

- Supports BOTH list- and dict-shaped JSON (dict: product_id -> product).
- Normalizes rich fields: prices (current/original/discount), ratings, images,
  colors, sizes, delivery options, return/warranty, seller metrics, variants.
- Uses Chonkie (RecursiveChunker) to chunk robust product "documents".
- Uses OpenAI Embeddings for retrieval and OpenAI Chat for answering.
- Flat, dependency-light, fast; no vector DB (NumPy cosine + small cache).
- Just set USER_QUERY and run:  python rag_products.py

Folder layout expected:
result/
  ├── www_daraz_com_bd_kitchen_fixtures/
  │     └── products.json
  ├── www_daraz_com_bd_shop_bedding_sets/
  │     └── products.json
  └── ... (many categories)

Notes:
- JSON can vary a lot; normalization is defensive.
- Only products.json is read in each category folder.

Author: you + hippo power 🦛
"""

import os, json, glob, re, hashlib, pickle, math, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()
import numpy as np

# ---- OpenAI SDK (v1) ----
from openai import OpenAI

# ---- Chonkie (chunking) ----
from chonkie import RecursiveChunker

# ---- Optional: token-accurate chunk sizing; falls back to char-count ----
try:
    import tiktoken
    _enc = None
    for name in ("o200k_base", "cl100k_base"):
        try:
            _enc = tiktoken.get_encoding(name)
            break
        except Exception:
            continue
    if _enc is None:
        raise RuntimeError("No tiktoken encoding found.")

    def count_tokens(s: str) -> int:
        return len(_enc.encode(s))
except Exception:
    def count_tokens(s: str) -> int:
        # rough heuristic if tiktoken not available
        return max(1, len(s) // 4)


# =================== CONFIG ===================

ROOT_DIR = os.getenv("RAG_ROOT_DIR", "../layer_23/result")  # your scraped data root

# Embedding model (small = cheap/fast; change if you want)
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Chat model for the final answer (change via env if you prefer)
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")

# Chunking: target token length and minimum chars
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "320"))
MIN_CHARS_PER_CHUNK = int(os.getenv("MIN_CHARS_PER_CHUNK", "24"))

# Retrieval
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "12"))      # retrieve top-k chunks
TOP_PRODUCTS = int(os.getenv("TOP_PRODUCTS", "6"))       # cap unique products in final context

# Embedding batching + cache
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE_PATH", ".emb_cache.pkl")

# Put your question here (or set via env)
USER_QUERY = os.getenv("RAG_QUERY", "show me some traditional laptop")

# ==============================================


# ------------------ Data Models ------------------

@dataclass
class Variant:
    color: Optional[str] = None
    price_display: Optional[str] = None
    price_value: Optional[float] = None
    original_display: Optional[str] = None
    original_value: Optional[float] = None
    discount_display: Optional[str] = None
    discount_percent: Optional[float] = None
    images: List[str] = field(default_factory=list)
    sizes: List[str] = field(default_factory=list)

@dataclass
class ProductDoc:
    # identifiers
    doc_id: str
    category: str

    # primary attributes
    title: Optional[str]
    url: Optional[str]
    brand: Optional[str]

    # price (current + original + discount)
    price_value: Optional[float]
    price_display: Optional[str]
    price_original_value: Optional[float] = None
    price_original_display: Optional[str] = None
    discount_display: Optional[str] = None
    discount_percent: Optional[float] = None

    # social proof
    rating_avg: Optional[float] = None
    rating_count: Optional[int] = None

    # marketplace dynamics
    sold_count: Optional[int] = None
    seller_name: Optional[str] = None
    seller_link: Optional[str] = None
    seller_metrics: Dict[str, Any] = field(default_factory=dict)

    # merchandising
    colors: List[str] = field(default_factory=list)
    sizes: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)

    # PDP details
    description: Optional[str] = None
    description_html: Optional[str] = None
    highlights: List[str] = field(default_factory=list)
    specifications: List[Dict[str, Any]] = field(default_factory=list)
    whats_in_the_box: Optional[str] = None

    # logistics / policy
    delivery_options: List[Dict[str, Optional[str]]] = field(default_factory=list)
    return_and_warranty: List[str] = field(default_factory=list)

    # variants
    variants: List[Variant] = field(default_factory=list)

    # bookkeeping
    source_path: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


# ------------------ Helpers ------------------

def _first(*vals):
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
        if v not in (None, "", [], {}):
            return v
    return None

def _to_https(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = str(u).strip()
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("http"):
        return u
    if u.startswith("/"):
        # site-relative; assume daraz main host
        return "https://www.daraz.com.bd" + u
    if u.startswith("www."):
        return "https://" + u
    return u

def _ensure_list(x) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _to_https_list(urls: Optional[List[str]]) -> List[str]:
    return [v for v in (_to_https(u) for u in (urls or [])) if v]

_number_re = re.compile(r"(\d[\d,\.]*)")
def _parse_number(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    m = _number_re.search(str(s))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None

def _parse_int(s: Optional[str]) -> Optional[int]:
    v = _parse_number(s)
    return int(v) if v is not None else None

def _parse_sold(location_val: Optional[str]) -> Optional[int]:
    if not location_val:
        return None
    m = re.search(r"(\d[\d,]*)\s*sold", str(location_val), flags=re.I)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:
            return None
    return _parse_int(location_val)  # fallback

def _get_nested(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _norm_price(p: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[float],
                                                      Optional[str], Optional[float],
                                                      Optional[str], Optional[float]]:
    """
    Returns:
      display, value, original_display, original_value, discount_display, discount_percent
    """
    if not isinstance(p, dict):
        return None, None, None, None, None, None
    return (
        _first(p.get("display")),
        _parse_number(_first(p.get("value"))),
        _first(p.get("original_display")),
        _parse_number(_first(p.get("original_value"))),
        _first(p.get("discount_display")),
        _parse_number(_first(p.get("discount_percent")))
    )


# --------------- Normalization ----------------

def normalize_product(
    prod: Dict[str, Any],
    category: str,
    source_path: str,
    raw_id_override: Optional[str] = None
) -> ProductDoc:
    # Robust ID (prefer override or explicit IDs; else hash)
    raw_id = _first(
        raw_id_override,
        prod.get("data_item_id"),
        prod.get("data_sku_simple"),
        prod.get("id"),
    )
    if not raw_id:
        # stable hash from title+url
        base = (_first(prod.get("product_title"), prod.get("name"), _get_nested(prod, ["detail", "name"], "")) or "") \
               + "|" + (_first(prod.get("product_detail_url"), prod.get("detail_url"), prod.get("url"), _get_nested(prod, ["detail", "url"], "")) or "")
        raw_id = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

    # Title / URL / Brand
    title = _first(prod.get("product_title"), prod.get("name"), _get_nested(prod, ["detail", "name"]))
    url = _to_https(_first(prod.get("detail_url"), prod.get("product_detail_url"), prod.get("url"), _get_nested(prod, ["detail", "url"])))
    brand = _first(prod.get("brand"), _get_nested(prod, ["detail", "brand"]))

    # Prices (current + original + discount)
    p_display, p_value, p_orig_display, p_orig_value, p_disc_display, p_disc_percent = _norm_price(
        prod.get("price") or _get_nested(prod, ["detail", "price"])
    )

    # Ratings
    rating_avg = _first(
        _get_nested(prod, ["detail", "rating", "average"]),
        _get_nested(prod, ["rating", "average"])
    )
    try:
        rating_avg = float(rating_avg) if rating_avg is not None else None
    except Exception:
        rating_avg = None

    rating_count = _first(
        _get_nested(prod, ["detail", "rating", "count"]),
        _get_nested(prod, ["rating", "count"])
    )
    if isinstance(rating_count, str):
        rating_count = _parse_int(rating_count)

    # Seller
    seller_name = _first(_get_nested(prod, ["detail", "seller", "name"]), _get_nested(prod, ["seller", "name"]))
    seller_link = _to_https(_first(_get_nested(prod, ["seller", "link"]), _get_nested(prod, ["detail", "seller", "link"])))
    seller_metrics = _get_nested(prod, ["seller", "metrics"], {}) or {}

    # Merchandising
    colors = [str(c) for c in _ensure_list(prod.get("colors"))]
    sizes = [str(s) for s in _ensure_list(prod.get("sizes"))]
    images = _to_https_list(_ensure_list(prod.get("images")))

    # Description & details
    details_node = prod.get("details") or _get_nested(prod, ["detail", "details"]) or {}
    description_text = _first(details_node.get("description_text"))
    description_html = _first(details_node.get("description_html"))
    highlights = _ensure_list(details_node.get("highlights"))
    specifications = _ensure_list(details_node.get("specifications"))
    whats_in_the_box = _first(details_node.get("whats_in_the_box"))
    raw_text = _first(details_node.get("raw_text"))
    description = description_text or raw_text

    # Logistics / policy
    delivery_options: List[Dict[str, Optional[str]]] = []
    for d in _ensure_list(prod.get("delivery_options")):
        if isinstance(d, dict):
            delivery_options.append({
                "title": _first(d.get("title")),
                "time": _first(d.get("time")),
                "fee": _first(d.get("fee")),
            })
    return_and_warranty = [str(x) for x in _ensure_list(prod.get("return_and_warranty"))]

    # Variants
    variants: List[Variant] = []
    for v in _ensure_list(prod.get("variants")):
        if not isinstance(v, dict):
            continue
        v_display, v_value, v_orig_display, v_orig_value, v_disc_display, v_disc_percent = _norm_price(v.get("price"))
        variants.append(Variant(
            color=_first(v.get("color")),
            price_display=v_display,
            price_value=float(v_value) if v_value is not None else None,
            original_display=v_orig_display,
            original_value=float(v_orig_value) if v_orig_value is not None else None,
            discount_display=v_disc_display,
            discount_percent=float(v_disc_percent) if v_disc_percent is not None else None,
            images=_to_https_list(_ensure_list(v.get("images"))),
            sizes=[str(s) for s in _ensure_list(v.get("sizes"))],
        ))

    # Sold count (not present in sample; keep robust parser)
    sold_count = _parse_sold(prod.get("location"))

    return ProductDoc(
        doc_id=str(raw_id),
        category=category,
        title=str(title) if title else None,
        url=url,
        brand=str(brand) if brand else None,

        price_value=float(p_value) if p_value is not None else None,
        price_display=p_display,
        price_original_value=float(p_orig_value) if p_orig_value is not None else None,
        price_original_display=p_orig_display,
        discount_display=p_disc_display,
        discount_percent=float(p_disc_percent) if p_disc_percent is not None else None,

        rating_avg=float(rating_avg) if rating_avg is not None else None,
        rating_count=int(rating_count) if rating_count is not None else None,

        sold_count=int(sold_count) if sold_count is not None else None,
        seller_name=str(seller_name) if seller_name else None,
        seller_link=seller_link,
        seller_metrics=seller_metrics,

        colors=colors,
        sizes=sizes,
        images=images,

        description=str(description) if description else None,
        description_html=str(description_html) if description_html else None,
        highlights=[str(h) for h in highlights],
        specifications=specifications,
        whats_in_the_box=str(whats_in_the_box) if whats_in_the_box else None,

        delivery_options=delivery_options,
        return_and_warranty=[str(r) for r in return_and_warranty],

        variants=variants,

        source_path=source_path,
        raw=prod,
    )


# --------------- Loader (list OR dict) ----------------

def iter_product_docs(root_dir: str) -> List[ProductDoc]:
    docs: List[ProductDoc] = []
    for products_json in glob.glob(os.path.join(root_dir, "*", "products.json")):
        category = os.path.basename(os.path.dirname(products_json))
        try:
            with open(products_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Skipping {products_json}: {e}")
            continue

        # Case A: list of product dicts
        if isinstance(data, list):
            for prod in data:
                if not isinstance(prod, dict):
                    continue
                doc = normalize_product(prod, category=category, source_path=products_json)
                docs.append(doc)

        # Case B: dict of {product_id: product_dict}
        elif isinstance(data, dict):
            for raw_id, prod in data.items():
                if not isinstance(prod, dict):
                    continue
                doc = normalize_product(prod, category=category, source_path=products_json, raw_id_override=str(raw_id))
                docs.append(doc)
        else:
            print(f"[WARN] Unsupported JSON shape in {products_json}: {type(data).__name__}")
            continue

    return docs


# --------------- Text Construction ----------------

def product_text(doc: ProductDoc) -> str:
    """
    Build a retrieval-friendly representation (no images inline by default).
    Keeps it compact but info-rich: prices, ratings, seller, attrs, policies, variants summary.
    """
    parts = []
    parts.append(f"PRODUCT_ID: {doc.doc_id}")
    parts.append(f"CATEGORY: {doc.category}")
    if doc.title: parts.append(f"TITLE: {doc.title}")
    if doc.brand: parts.append(f"BRAND: {doc.brand}")

    # Price summary
    if doc.price_display or doc.price_value is not None:
        price_line = f"PRICE: {doc.price_display or doc.price_value}"
        if doc.price_original_display or doc.price_original_value is not None:
            price_line += f" | ORIGINAL: {doc.price_original_display or doc.price_original_value}"
        if doc.discount_display or doc.discount_percent is not None:
            # prefer display (e.g., "-48%"), fall back to numeric percent
            pct = f"{doc.discount_percent:.0f}%" if (doc.discount_percent is not None and doc.discount_display is None) else (doc.discount_display or "")
            price_line += f" | DISCOUNT: {pct or 'N/A'}"
        parts.append(price_line)

    # Ratings / sales
    if doc.rating_avg is not None: parts.append(f"RATING_AVG: {doc.rating_avg:.2f}")
    if doc.rating_count is not None: parts.append(f"RATING_COUNT: {doc.rating_count}")
    if doc.sold_count is not None: parts.append(f"SOLD: {doc.sold_count}")

    # Seller
    if doc.seller_name: parts.append(f"SELLER: {doc.seller_name}")
    if doc.seller_metrics:
        metrics_str = "; ".join([f"{k}: {v}" for k, v in doc.seller_metrics.items()])
        parts.append(f"SELLER_METRICS: {metrics_str}")

    # Attributes
    if doc.colors: parts.append("COLORS: " + ", ".join(doc.colors))
    if doc.sizes: parts.append("SIZES: " + ", ".join(doc.sizes))
    if doc.images: parts.append(f"IMAGES_COUNT: {len(doc.images)}")

    # Delivery & policy (compact)
    if doc.delivery_options:
        deliv = " | ".join(
            [", ".join([x for x in [d.get('title'), d.get('time'), d.get('fee')] if x]) for d in doc.delivery_options]
        )
        parts.append(f"DELIVERY_OPTIONS: {deliv}")
    if doc.return_and_warranty:
        parts.append("RETURN_AND_WARRANTY: " + " | ".join(doc.return_and_warranty))

    # Description & highlights
    if doc.highlights:
        parts.append("HIGHLIGHTS: " + " | ".join(map(str, doc.highlights)))
    if doc.whats_in_the_box:
        parts.append("WHATS_IN_THE_BOX: " + str(doc.whats_in_the_box))
    if doc.description:
        parts.append("DESCRIPTION: " + doc.description)

    # Variants (summary per variant)
    if doc.variants:
        vlines = []
        for v in doc.variants:
            seg = []
            if v.color: seg.append(f"color={v.color}")
            if v.price_display or v.price_value is not None:
                seg.append(f"price={v.price_display or v.price_value}")
            if v.discount_display or v.discount_percent is not None:
                seg.append(f"discount={v.discount_display or (str(v.discount_percent)+'%')}")
            if v.sizes: seg.append(f"sizes={','.join(v.sizes)}")
            if v.images: seg.append(f"images={len(v.images)}")
            vlines.append("{" + ", ".join(seg) + "}")
        parts.append("VARIANTS: " + " ".join(vlines))

    if doc.url: parts.append(f"URL: {doc.url}")

    return "\n".join(parts)


# --------- Chunking with Chonkie (fast & simple) ---------

def chunk_docs(docs: List[ProductDoc]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns:
      chunks_texts: List[str]  (to embed)
      chunks_meta:  List[dict] (metadata per chunk with backrefs)
    """
    chunker = RecursiveChunker(
        tokenizer_or_token_counter=count_tokens,
        chunk_size=CHUNK_SIZE_TOKENS,
        # default rules are fine; min chars prevents tiny slivers
        min_characters_per_chunk=MIN_CHARS_PER_CHUNK,
    )

    chunks_texts: List[str] = []
    chunks_meta: List[Dict[str, Any]] = []

    for doc in docs:
        text = product_text(doc)
        chunks = chunker(text)  # callable interface
        for i, ch in enumerate(chunks):
            clean = ch.text.replace("\u0000", " ").strip()
            if not clean:
                continue
            chunks_texts.append(clean)
            chunks_meta.append({
                "doc_id": doc.doc_id,
                "chunk_idx": i,
                "category": doc.category,
                "title": doc.title,
                "url": doc.url,
                "price_display": doc.price_display,
                "price_value": doc.price_value,
                "price_original_display": doc.price_original_display,
                "discount_percent": doc.discount_percent,
                "rating_avg": doc.rating_avg,
                "rating_count": doc.rating_count,
                "sold_count": doc.sold_count,
                "seller_name": doc.seller_name,
                "images_count": len(doc.images) if doc.images else 0,
                "sizes": doc.sizes,
                "source_path": doc.source_path,
                "token_count": getattr(ch, "token_count", None),
            })
    return chunks_texts, chunks_meta


# --------- Embedding (OpenAI) + tiny persistent cache ---------

class EmbeddingCache:
    def __init__(self, path: str):
        self.path = path
        try:
            with open(self.path, "rb") as f:
                self._store = pickle.load(f)
        except Exception:
            self._store = {}  # sha1 -> list[float]

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
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()

def embed_texts(client: OpenAI, texts: List[str], model: str, batch_size: int, cache: Optional[EmbeddingCache] = None) -> np.ndarray:
    """
    Returns normalized embeddings (L2 unit vectors) as ndarray of shape (N, D)
    """
    all_vecs: List[Optional[List[float]]] = []
    to_embed_idx: List[int] = []
    keys: List[Optional[str]] = []

    # Cache lookup
    if cache:
        for i, t in enumerate(texts):
            key = _sha1_for_text_model(t.replace("\n", " "), model)
            keys.append(key)
            vec = cache.get(key)
            if vec is None:
                to_embed_idx.append(i)
                all_vecs.append(None)  # placeholder
            else:
                all_vecs.append(vec)
    else:
        keys = [None] * len(texts)
        to_embed_idx = list(range(len(texts)))
        all_vecs = [None] * len(texts)

    # Batch embed missing
    for start in range(0, len(to_embed_idx), batch_size):
        batch_ids = to_embed_idx[start:start + batch_size]
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

    # Convert to numpy and L2-normalize
    arr = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


# --------- Retrieval + prompt assembly ---------

def cosine_topk(query_vec: np.ndarray, mat: np.ndarray, k: int) -> List[int]:
    # mat: (N, D), query: (D,)
    scores = mat @ query_vec  # cosine since both sides are normalized
    # Argpartition for speed; then sort the small slice
    if k >= len(scores):
        idx = np.argsort(-scores)
        return idx.tolist()
    top_idx = np.argpartition(-scores, k)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx.tolist()

def build_context(retrieved_idx: List[int], chunks_texts: List[str], chunks_meta: List[Dict[str, Any]], limit_products: int) -> Tuple[str, List[Dict[str, Any]]]:
    seen = set()
    lines = []
    included_meta: List[Dict[str, Any]] = []
    for j in retrieved_idx:
        meta = chunks_meta[j]
        doc_id = meta["doc_id"]
        if doc_id in seen:
            pass
        else:
            if len(seen) >= limit_products:
                break
            seen.add(doc_id)
        # Compact header per chunk so the model can ground answers
        header = [
            f"[DOC {doc_id} | {meta.get('title') or 'Untitled'}]",
            f"Category: {meta.get('category')}",
            f"URL: {meta.get('url') or 'N/A'}",
            f"Price: {meta.get('price_display') or meta.get('price_value') or 'N/A'} "
            f"(Orig: {meta.get('price_original_display') or 'N/A'}, Disc: {meta.get('discount_percent') if meta.get('discount_percent') is not None else 'N/A'})",
            f"Rating: {meta.get('rating_avg')} ({meta.get('rating_count')} ratings) | Sold: {meta.get('sold_count')}",
            f"Seller: {meta.get('seller_name') or 'N/A'} | Images: {meta.get('images_count')} | Sizes: {', '.join(meta.get('sizes') or []) or 'N/A'}",
        ]
        lines.append("\n".join(header))
        lines.append(chunks_texts[j])
        lines.append("-" * 60)
        included_meta.append(meta)
    return "\n".join(lines), included_meta


SYSTEM_PROMPT = """You are a precise product QA assistant. Answer ONLY using the provided product context.
- Prefer items with higher rating and more ratings if the user asks for "best".
- If the user mentions budget, filter/compare prices accordingly.
- When specific attributes are requested (brand, colors, size), extract from context exactly.
- If you don't find an answer in context, say you don't know.
- Return concise bullet points per product with:
  Title, Brand, Price (current, original, discount), Rating (avg, count), Sold, Seller (name; basic metrics if present),
  Colors, Sizes, Delivery, Returns/ Warranty, Variants summary (color + price), and a URL.
- Do not hallucinate missing values; mark them as N/A.
"""

def answer_with_llm(client: OpenAI, model: str, user_query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User query:\n{user_query}\n\nContext (multiple products & chunks):\n{context}"},
    ]
    # Chat Completions are widely supported; simple + reliable.
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=700,
    )
    return resp.choices[0].message.content.strip()


# ------------------ Main ------------------

def main():
    t0 = time.time()
    client = OpenAI()

    print(f"[INFO] Loading products from: {ROOT_DIR}")
    docs = iter_product_docs(ROOT_DIR)
    if not docs:
        print("[ERROR] No products found. Is the 'result/' folder present and non-empty?")
        return
    print(f"[OK] Loaded {len(docs)} products from {ROOT_DIR}")

    print("[INFO] Chunking with Chonkie...")
    chunks_texts, chunks_meta = chunk_docs(docs)
    print(f"[OK] Produced {len(chunks_texts)} chunks (avg per product: {len(chunks_texts)/max(1,len(docs)):.2f})")

    print("[INFO] Embedding chunks (with small local cache for speed)...")
    cache = EmbeddingCache(EMBED_CACHE_PATH)
    chunk_vecs = embed_texts(client, chunks_texts, model=EMBED_MODEL, batch_size=EMBED_BATCH_SIZE, cache=cache)
    print(f"[OK] Embedded chunks => shape {chunk_vecs.shape}")

    print(f"[INFO] Query: {USER_QUERY}")
    query_vec = embed_texts(client, [USER_QUERY], model=EMBED_MODEL, batch_size=1, cache=cache)[0]

    idx = cosine_topk(query_vec, chunk_vecs, k=TOP_K_CHUNKS)
    context, included_meta = build_context(idx, chunks_texts, chunks_meta, limit_products=TOP_PRODUCTS)

    print("[INFO] Asking LLM for the final grounded answer...")
    answer = answer_with_llm(client, CHAT_MODEL, USER_QUERY, context)

    print("\n================= ANSWER =================")
    print(answer)
    print("=========================================\n")

    elapsed = time.time() - t0
    print(f"[DONE] Total time: {elapsed:.2f}s | Products: {len(docs)} | Chunks: {len(chunks_texts)}")


if __name__ == "__main__":
    main()
