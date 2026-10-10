"""Offline checks for the real-corpus downloader (standard library only)."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "download_corpus.py"
if SCRIPT.exists():
    spec = importlib.util.spec_from_file_location("download_corpus", SCRIPT)
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
else:
    downloader = None


def row(index, text="Wikipedia is a free online encyclopedia.", truncated=False):
    return {"row_idx": index, "row": {
        "id": str(index), "url": f"https://en.wikipedia.org/wiki/{index}",
        "title": f"Article {index}", "text": text,
    }, "truncated_cells": ["text"] if truncated else []}


def response(rows):
    return io.BytesIO(json.dumps({"rows": rows, "num_rows_total": 1384748}).encode())


class DownloadCorpusTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(downloader, "The corpus downloader is not implemented")
        self.assertEqual(downloader.CONFIG, "20231101.en")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "data/raw/wikipedia_20231101_en.jsonl"
        self.meta = self.root / "data/sources/wikipedia_20231101_en.json"

    def test_success_schema_provenance_and_empty_rows(self):
        with patch.object(downloader, "urlopen", return_value=response([row(7), row(8, "  ")])):
            downloader.download(self.root, max_rows=2, offset=7)
        raw = self.raw.read_bytes()
        record = json.loads(raw)
        self.assertEqual(set(record), {"text", "id", "url", "title", "source"})
        self.assertEqual(record["id"], "7")
        meta = json.loads(self.meta.read_text(encoding="utf-8"))
        self.assertEqual(meta["rows_written"], 1)
        self.assertEqual(meta["empty_rows_skipped"], 1)
        self.assertEqual(meta["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(meta["raw_bytes"], len(raw))
        self.assertEqual(meta["selection"], "contiguous_offset_not_random")
        self.assertEqual(meta["language"], "en")
        self.assertEqual(meta["config"], "20231101.en")

    def test_truncated_row_is_refetched_alone(self):
        with patch.object(downloader, "urlopen", side_effect=[
            response([row(0, "Wikipedia is", True), row(1)]), response([row(0, "Wikipedia is a free online encyclopedia.")])
        ]) as fetch:
            downloader.download(self.root, max_rows=2)
        self.assertEqual(json.loads(self.raw.read_text(encoding="utf-8").splitlines()[0])["text"], "Wikipedia is a free online encyclopedia.")
        self.assertIn("length=1", fetch.call_args_list[1].args[0].full_url)

    def test_still_truncated_response_never_publishes(self):
        with patch.object(downloader, "urlopen", side_effect=[
            response([row(0, truncated=True), row(1)]), response([row(0, truncated=True)])
        ]):
            with self.assertRaisesRegex(RuntimeError, "truncat"):
                downloader.download(self.root, max_rows=2)
        self.assertFalse(self.raw.exists())
        self.assertFalse(self.meta.exists())

    def test_failed_download_resumes_cached_pages(self):
        with patch.object(downloader, "urlopen", side_effect=[
            response([row(i) for i in range(100)]), URLError("offline"),
            URLError("offline"), URLError("offline"), URLError("offline"),
        ]), patch.object(downloader.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "download"):
                downloader.download(self.root, max_rows=101)
        self.assertFalse(self.raw.exists())
        self.assertFalse(self.meta.exists())
        with patch.object(downloader, "urlopen", return_value=response([row(100)])) as fetch:
            downloader.download(self.root, max_rows=101)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(len(self.raw.read_text(encoding="utf-8").splitlines()), 101)

    def test_existing_target_requires_force(self):
        self.raw.parent.mkdir(parents=True)
        self.raw.write_text("keep me", encoding="utf-8")
        with patch.object(downloader, "urlopen") as fetch:
            with self.assertRaises(FileExistsError):
                downloader.download(self.root, max_rows=1)
        fetch.assert_not_called()
        self.assertEqual(self.raw.read_text(encoding="utf-8"), "keep me")
        with patch.object(downloader, "urlopen", return_value=response([row(0)])):
            downloader.download(self.root, max_rows=1, force=True)
        self.assertEqual(json.loads(self.raw.read_text(encoding="utf-8"))["id"], "0")

    def test_rate_limit_honors_retry_after_and_backoff(self):
        cases = [
            (429, "17", 17),
            (429, "Wed, 21 Oct 2015 07:28:00 GMT", 20),
            (503, "19", 19),
            (429, None, 15),
        ]
        for index, (status, retry_after, delay) in enumerate(cases):
            with self.subTest(status=status, retry_after=retry_after):
                headers = {} if retry_after is None else {"Retry-After": retry_after}
                error = HTTPError("https://example.test", status, "Rate limited", headers, None)
                with patch.object(downloader, "urlopen", side_effect=[error, response([row(0)])]), \
                     patch.object(downloader.time, "sleep") as sleep, \
                     patch.object(downloader.time, "time", return_value=1445412460):
                    downloader.download(self.root / str(index), max_rows=1)
                sleep.assert_called_once_with(delay)

    def test_long_retry_after_fails_without_retrying_early(self):
        error = HTTPError("https://example.test", 429, "Rate limited", {"Retry-After": "120"}, None)
        with patch.object(downloader, "urlopen", side_effect=error) as fetch, \
             patch.object(downloader.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "120 seconds"):
                downloader.download(self.root, max_rows=1)
        self.assertEqual(fetch.call_count, 1)
        sleep.assert_not_called()
        self.assertFalse(self.raw.exists())


if __name__ == "__main__":
    unittest.main()
