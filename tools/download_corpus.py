"""Download a bounded, cached sample of real English Wikipedia articles.

Run from any directory: python tools/download_corpus.py --max-rows 5000
Uses only Python's standard library; existing output requires --force.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.en"
NAME = "wikipedia_20231101_en"
API_URL = "https://datasets-server.huggingface.co/rows"
CARD_URL = "https://huggingface.co/datasets/wikimedia/wikipedia"
PAGE_SIZE = 100


def fetch_page(cache_dir, offset, length, *, refresh=False):
    """Cache each complete HTTP response; retry transient fetch failures 3 times."""
    cache = cache_dir / f"{offset:09d}_{length:03d}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    url = API_URL + "?" + urlencode({
        "dataset": DATASET, "config": CONFIG, "split": "train",
        "offset": offset, "length": length,
    })
    request = Request(url, headers={"User-Agent": "LLM-Learn-CorpusDownloader/1.0"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read()
            page = json.loads(body)
            if not isinstance(page.get("rows"), list) or not isinstance(page.get("num_rows_total"), int):
                raise ValueError("Unexpected rows API response schema")
            temporary = cache.with_suffix(".tmp")
            temporary.write_bytes(body)
            temporary.replace(cache)
            return page
        except (URLError, OSError, ValueError) as error:
            delay = attempt + 1
            if isinstance(error, HTTPError) and error.code in (429, 503):
                delay = 15 * (2 ** attempt) if error.code == 429 else delay
                retry_after = (error.headers.get("Retry-After", "") if error.headers else "").strip()
                if retry_after:
                    try:
                        if retry_after.isdigit():
                            delay = int(retry_after)
                        else:
                            deadline = parsedate_to_datetime(retry_after)
                            if deadline.tzinfo is None:
                                deadline = deadline.replace(tzinfo=timezone.utc)
                            delay = max(0, math.ceil(deadline.timestamp() - time.time()))
                    except (ValueError, TypeError, OverflowError):
                        pass
                if delay > 60:
                    raise RuntimeError(f"Cannot download rows at offset {offset}: wait at least {delay} seconds before rerunning; cached pages are preserved") from error
            if attempt == 3:
                raise RuntimeError(f"Cannot download rows at offset {offset}: {error}") from error
            print(f"Retry {attempt + 1}/3 for offset {offset} in {delay} seconds: {error}", flush=True)
            time.sleep(delay)


def download(root=ROOT, *, max_rows=5000, offset=0, force=False):
    """Publish JSONL and provenance only after all requested rows are processed.

    max_rows bounds source rows examined, including skipped empty articles.
    Sampling is a contiguous prefix/offset range, not a random sample.
    """
    if max_rows < 1 or offset < 0:
        raise ValueError("max_rows must be positive and offset must be nonnegative")
    root = Path(root)
    raw = root / "data" / "raw" / f"{NAME}.jsonl"
    metadata_path = root / "data" / "sources" / f"{NAME}.json"
    cache_dir = root / "data" / "download_cache" / NAME
    for target in (raw, metadata_path):
        if target.exists() and not force:
            raise FileExistsError(f"Output exists: {target}. Use --force to replace generated output.")
    for directory in (raw.parent, metadata_path.parent, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)
    # Neither temporary file ends in .jsonl, so notebook corpus discovery ignores it.
    raw_tmp = raw.with_suffix(".jsonl.tmp")
    metadata_tmp = metadata_path.with_suffix(".json.tmp")
    written = skipped = examined = byte_count = 0
    total = None
    digest = hashlib.sha256()
    try:
        with raw_tmp.open("wb") as output:
            while examined < max_rows:
                current = offset + examined
                if total is not None and current >= total:
                    break
                length = min(PAGE_SIZE, max_rows - examined)
                if total is not None:
                    length = min(length, total - current)
                page = fetch_page(cache_dir, current, length)
                total = page["num_rows_total"]
                entries = page["rows"]
                expected = min(length, max(0, total - current))
                if len(entries) != expected:
                    raise RuntimeError(f"Incomplete page at offset {current}: expected {expected} rows, got {len(entries)}")
                for local_index, entry in enumerate(entries):
                    row_index = current + local_index
                    if entry.get("row_idx") != row_index:
                        raise RuntimeError(f"Unexpected row index at offset {row_index}")
                    if entry.get("truncated_cells"):
                        print(f"Refetch truncated row {row_index} individually", flush=True)
                        single = fetch_page(cache_dir, row_index, 1, refresh=True)["rows"]
                        if len(single) != 1 or single[0].get("row_idx") != row_index:
                            raise RuntimeError(f"Invalid single-row response at offset {row_index}")
                        entry = single[0]
                        if entry.get("truncated_cells"):
                            raise RuntimeError(f"Row {row_index} is still truncated after a length=1 request")
                    article = entry.get("row", {})
                    if not all(isinstance(article.get(field), str) for field in ("id", "url", "title", "text")):
                        raise RuntimeError(f"Invalid article schema at row {row_index}")
                    examined += 1
                    if not article["text"].strip():
                        skipped += 1
                        continue
                    record = {field: article[field] for field in ("text", "id", "url", "title")}
                    record["source"] = DATASET
                    encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                    output.write(encoded)
                    digest.update(encoded)
                    byte_count += len(encoded)
                    written += 1
                print(f"Processed {examined}/{max_rows} source rows; saved {written}; {byte_count:,} bytes", flush=True)
                if not entries:
                    break
        if written == 0:
            raise RuntimeError("No nonempty articles were downloaded; no output published")
        metadata = {
            "source": DATASET,
            "source_url": CARD_URL,
            "api_url": API_URL,
            "snapshot": "2023-11-01",
            "config": CONFIG,
            "split": "train",
            "language": "en",
            "license_card_url": CARD_URL + "#licensing-information",
            "selection": "prefix_not_random" if offset == 0 else "contiguous_offset_not_random",
            "sampling": {"offset": offset, "max_rows": max_rows, "page_size": PAGE_SIZE},
            "num_rows_total": total,
            "rows_examined": examined,
            "rows_written": written,
            "empty_rows_skipped": skipped,
            "raw_file": raw.relative_to(root).as_posix(),
            "raw_bytes": byte_count,
            "sha256": digest.hexdigest(),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raw_tmp.replace(raw)
        metadata_tmp.replace(metadata_path)
    finally:
        raw_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
    print(f"Saved: {raw}\nProvenance: {metadata_path}\nSHA256: {digest.hexdigest()}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=5000, help="Maximum source rows to inspect (default: 5000)")
    parser.add_argument("--offset", type=int, default=0, help="First source row (default: 0; selection is not random)")
    parser.add_argument("--force", action="store_true", help="Replace this script's JSONL and provenance files")
    args = parser.parse_args()
    try:
        download(max_rows=args.max_rows, offset=args.offset, force=args.force)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
