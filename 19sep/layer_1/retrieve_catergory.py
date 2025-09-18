import os, re
import json
import time
import random
import asyncio
from typing import List, Dict, Iterable, Tuple, Optional
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode, urldefrag
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlResult,
    CrawlerRunConfig,
    GeolocationConfig,
    LLMConfig,
    JsonCssExtractionStrategy,
)
# from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy  # not used, but often handy

# ======== Paths / files ========
__cur_dir__ = Path(__file__).parent
RESULT_DIR = __cur_dir__ / "category_result"
RESULT_DIR.mkdir(exist_ok=True)

CATEGORY_SCHEMA_FILE = RESULT_DIR / "category_schema.json"
CATEGORIES_FILE = RESULT_DIR / "categories.json"

# ======== Tunables ========
HOMEPAGE_URL = "https://www.daraz.com.bd/"  # change if needed
MAX_REFRESHES = 40                           # total number of refresh cycles
STOP_AFTER_NO_NEW_STREAK = 8                 # stop early if this many consecutive refreshes add nothing
ADD_CACHE_BUST = True                        # add a timestamp param to bypass CDN/browser cache
WAIT_SELECTOR = "css:.card-categories-li a.pc-custom-link, .pc-custom-link.card-categories-li-content, .card-categories-li"  # robust wait
# ==========================


def atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_json_if_exists(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def normalize_path(path: str) -> str:
    """Normalize path: collapse multiple slashes, remove trailing slash (except root)."""
    if not path:
        return "/"
    path = re.sub(r"/{2,}", "/", path)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return path


def canonicalize_category_link(base: str, href: str) -> Tuple[str, str]:
    """
    Convert an href into a canonical category URL and a dedupe key.

    Strategy:
    - Absolute URL, remove fragment, lowercase host
    - For dedupe key: strip leading 'www.' from host
    - If path looks specific (e.g., '/sanders'), DROP ALL query params
    - If path looks generic (e.g., '/catalog', '/search'), keep ONLY whitelisted params:
        {'categoryId','catId','catIdLv1','cat'} — including any parsed out of a 'params' JSON
    """
    if not href:
        return "", ""

    # absolutize, handle protocol-relative
    if href.startswith("//"):
        href = "https:" + href
    absolute = urljoin(base, href)
    absolute, _ = urldefrag(absolute)

    p = urlparse(absolute)
    scheme = "https" if not p.scheme else p.scheme
    host_lower = (p.netloc or "").lower()
    path_norm = normalize_path(p.path or "/")

    # parse query
    q_pairs = parse_qsl(p.query, keep_blank_values=True)
    qmap = {k: v for k, v in q_pairs}

    # Extract categoryId from Daraz's "params" JSON if present
    # (e.g., params={"categoryId":"10000526", ...})
    cat_from_params: Optional[str] = None
    if "params" in qmap and qmap["params"]:
        try:
            payload = json.loads(qmap["params"])
            for key in ("categoryId", "catId", "catIdLv1", "cat"):
                if key in payload and str(payload[key]).strip():
                    cat_from_params = str(payload[key]).strip()
                    break
        except Exception:
            pass

    # Whitelist logic
    whitelist = {"categoryId", "catId", "catIdLv1", "cat"}
    keep: Dict[str, str] = {}
    for k, v in q_pairs:
        if k in whitelist and str(v).strip():
            keep[k] = str(v).strip()
    if cat_from_params and not any(k in keep for k in whitelist):
        keep["categoryId"] = cat_from_params

    # Decide if path is "specific" (slug) vs "generic"
    # Specific (drop all qs): paths with 1+ word chars and not typical generic endpoints
    generic_paths = {"/catalog", "/search", "/category", "/categories"}
    is_generic = path_norm in generic_paths

    # Build canonical URL (what we store) — keep host case as-is from parsed absolute
    if is_generic and keep:
        qs = urlencode(sorted(keep.items()))
    else:
        qs = ""

    canonical_url = urlunparse((scheme, host_lower, path_norm, "", qs, ""))

    # Build dedupe key — remove leading 'www.' to unify host forms
    host_key = host_lower[4:] if host_lower.startswith("www.") else host_lower
    key = f"{host_key}{path_norm}"
    if qs:
        key += "?" + qs

    return canonical_url, key


def coerce_items(parsed) -> List[dict]:
    """Robustly get list[dict] from extractor output."""
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        for key in ("items", "data", "categories", "list", "results"):
            v = parsed.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        return [parsed]
    return []


def canonical_category_record(base: str, item: dict) -> Tuple[dict, str]:
    """
    From a raw extracted item, return (record, dedupe_key).
    Record contains canonical 'category_link' and 'category_name'.
    """
    name = (
        (item.get("category_name") or item.get("name") or item.get("title") or "").strip()
    )
    href = item.get("category_link") or item.get("link") or item.get("url") or ""
    canonical, key = canonicalize_category_link(base, href)
    if not name or not canonical or not key:
        return {}, ""
    return {"category_name": name, "category_link": canonical}, key
    # If you want to keep the original href too, return this instead:
    # return {"category_name": name, "category_link": canonical, "category_link_original": href}, key


def dedupe_by_key(records: Iterable[Tuple[dict, str]]) -> List[dict]:
    """Deduplicate in-memory list of (record, key) by key."""
    seen = set()
    out = []
    for rec, key in records:
        if not rec or not key or key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


async def load_or_generate_category_schema(home_url: str, sample_html: str) -> dict:
    """Load schema or generate via LLM once and cache."""
    if CATEGORY_SCHEMA_FILE.exists():
        with open(CATEGORY_SCHEMA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    schema = JsonCssExtractionStrategy.generate_schema(
        html=sample_html,
        llm_config=LLMConfig(
            provider="gemini/gemini-2.0-flash-lite-preview-02-05",
            api_token=os.getenv("GEMINI_API_KEY", os.getenv('GEMINI_API_KEY')),
        ),
        query=(
            f"""Extract category tiles from homepage {home_url}.
Return a JSON CSS extraction schema iterating each visible category card.
For each category, extract:
- category_name: visible text (e.g., .card-categories-name .text)
- category_link: anchor href (e.g., <a class="pc-custom-link card-categories-li-content">)

Repeating container: div.card-categories-li.hp-mod-card-hover a.pc-custom-link.card-categories-li-content

Only output the schema JSON."""
        ),
    )
    atomic_write_json(CATEGORY_SCHEMA_FILE, schema)
    return schema

async def crawl_categories_once(
    crawler: AsyncWebCrawler,
    url: str,
    config: CrawlerRunConfig,
    attempt: int,
) -> List[dict]:
    """
    Run a single crawl and return canonicalized category records (deduped within attempt).
    """
    results: List[CrawlResult] = await crawler.arun(
        url=url,
        config=config,
        js_code="""
            // Allow modules to render and lazy images to load
            await new Promise(r => setTimeout(r, 1200));
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(r => setTimeout(r, 800));
            window.scrollTo(0, 0);
        """,
        wait_for=WAIT_SELECTOR,
    )

    recs_with_keys: List[Tuple[dict, str]] = []

    for res_idx, res in enumerate(results, start=1):
        # # Save HTML for debugging this attempt
        # with open(RESULT_DIR / f"categories_attempt_{attempt}_{res_idx}.html", "w", encoding="utf-8") as f:
        #     f.write(res.html)

        if not res.success:
            continue

        try:
            parsed = json.loads(res.extracted_content)
        except Exception as e:
            print(f"⚠️ JSON parse error (attempt {attempt} result {res_idx}): {e}")
            continue

        items = coerce_items(parsed)
        base = url
        for it in items:
            rec, key = canonical_category_record(base, it)
            if rec and key:
                recs_with_keys.append((rec, key))

    # Dedupe within this attempt by canonical key
    deduped = dedupe_by_key(recs_with_keys)
    return deduped


async def collect_categories_multiple_refreshes(home_url: str, refreshes: int):
    """
    Refresh the homepage multiple times (cache-busted) to discover more categories.
    Only *new (by canonical key)* categories are appended to categories.json.
    """
    # Minimal snippet from your example to guide schema generation
    
    sample_html = '''
    <div class="card-categories-li hp-mod-card-hover">
        <a class="pc-custom-link card-categories-li-content" href="..." data-spm-protocol="i" venture="BD" id="9787" name="Other Projector Accessories" data-appeared="true" data-has-appeared="true" data-before-current-y="234" data-spm-anchor-id="a2a0e.tm80335411.8881886560.23" data-has-disappeared="true">
            <div class="picture-wrapper common-img card-categories-image img-w100p">
                <img src="https://img.drz.lazcdn.com/static/bd/p/073c624b86e2e65439411dfffb3ae018.jpg_170x170q80.jpg_.webp" style="object-fit:cover" data-spm-anchor-id="a2a0e.tm80335411.8881886560.i2.631612f7oTvWyC">
            </div>
            <div class="card-categories-name">
                <p class="text two-line-clamp">Other Projector Accessories</p>
            </div>
        </a>
    </div>
    '''
    
    schema = await load_or_generate_category_schema(home_url, sample_html)
    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

    config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        cache_mode=CacheMode.DISABLED,
        geolocation=GeolocationConfig(latitude=23.8103, longitude=90.4125),
        prettiify=True,
        wait_for_images=True,
        delay_before_return_html=True,
        mean_delay=0.0000001,
        scroll_delay=0.0000001,
        verbose=True,
        # bypass_cache=True,
    )

    # Load existing categories and rebuild known dict keyed by canonical key
    existing = load_json_if_exists(CATEGORIES_FILE, default=[])
    known: Dict[str, dict] = {}
    if isinstance(existing, list):
        for rec in existing:
            link = rec.get("category_link") or ""
            name = (rec.get("category_name") or "").strip()
            if not link or not name:
                continue
            # Recompute canonical (in case older file had non-canonical links)
            canonical, key = canonicalize_category_link(home_url, link)
            if canonical and key and key not in known:
                known[key] = {"category_name": name, "category_link": canonical}

    print(f"ℹ️ Starting with {len(known)} unique categories (after canonicalization).")

    browser_config = BrowserConfig(
        # headless=False,
        # enable_stealth=True,
        # viewport_width=1920, viewport_height=1080,
    )

    new_total = 0
    no_new_streak = 0

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for attempt in range(1, refreshes + 1):
            # Cache-busted URL to encourage a new homepage mix
            if ADD_CACHE_BUST:
                sep = "&" if "?" in home_url else "?"
                url = f"{home_url}{sep}_v={int(time.time()*1000)}&r={random.randint(1000,9999)}"
            else:
                url = home_url

            print(f"\n--- Refresh {attempt}/{refreshes}: {url} ---")

            try:
                items = await crawl_categories_once(crawler, url, config, attempt)
            except Exception as e:
                print(f"❌ Crawl failed on attempt {attempt}: {e}")
                items = []

            # Attach dedupe keys and filter against known
            fresh: List[dict] = []
            for it in items:
                canonical, key = canonicalize_category_link(home_url, it["category_link"])
                if key not in known:
                    known[key] = {"category_name": it["category_name"], "category_link": canonical}
                    fresh.append(known[key])

            if fresh:
                no_new_streak = 0
                new_total += len(fresh)
                # Save per-refresh new categories
                atomic_write_json(RESULT_DIR / f"refresh_{attempt}.json", fresh)
                print(f"✅ Found {len(fresh)} new categories on refresh #{attempt}.")
            else:
                no_new_streak += 1
                print(f"ℹ️ No new categories on refresh #{attempt}. No-new streak={no_new_streak}.")

            # Update global categories file each attempt
            all_unique = list(known.values())
            atomic_write_json(CATEGORIES_FILE, all_unique)
            print(f"📦 Saved {len(all_unique)} total unique categories so far.")

            if no_new_streak >= STOP_AFTER_NO_NEW_STREAK:
                print(f"⏹️ Stopping early: no new categories for {no_new_streak} consecutive refreshes.")
                break

            await asyncio.sleep(0.8 + random.random() * 0.6)

    print(f"\n🎉 Done. Total unique categories: {len(known)}. Newly added this run: {new_total}.")
    print(f"➡️ See {CATEGORIES_FILE}")


async def main():
    print("=== Category Collector (Homepage, canonical-dedupe) ===")
    await collect_categories_multiple_refreshes(HOMEPAGE_URL, MAX_REFRESHES)
    print("\n=== Complete ===")


if __name__ == "__main__":
    asyncio.run(main())