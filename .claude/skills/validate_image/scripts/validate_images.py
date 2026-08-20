#!/usr/bin/env python3
"""Validate that every externally-hosted image referenced by the course notebooks
still resolves with HTTP 200.

Reads .ipynb files (stdlib json, no nbformat dependency), extracts external image
URLs from markdown cells (markdown image syntax + HTML <img>/<source> tags) and
from code cells (bare URLs ending in an image extension), checks each one
concurrently, and writes a markdown report.

Exit code is 0 when every URL is reachable, 1 when at least one is broken, so the
script can be wired into CI or a git hook.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

IMAGE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".tif", ".tiff", ".avif", ".ico",
)

# ![alt](url "optional title")
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)>\s]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)")
# <img src="url">, <source src="url">, poster="url"
HTML_ATTR_RE = re.compile(r"<(?:img|source|embed)\b[^>]*?\b(?:src|data-src|poster)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
# Bare URLs (used to catch image links inside code cells)
BARE_URL_RE = re.compile(r"https?://[^\s\"'<>)\\\]]+")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


@dataclass
class Reference:
    """A single external image URL and every place it is referenced."""

    url: str
    occurrences: list[tuple[str, int, str]] = field(default_factory=list)  # (notebook, cell index, cell type)

    status: int | None = None
    reason: str = ""
    content_type: str = ""
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.status == 200

    @property
    def notebooks(self) -> list[str]:
        seen: dict[str, None] = {}
        for notebook, _, _ in self.occurrences:
            seen.setdefault(notebook, None)
        return list(seen)


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def is_external(url: str) -> bool:
    return urlparse(url).scheme in ("http", "https")


def looks_like_image(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(IMAGE_EXTENSIONS)


def extract_urls(cell: dict) -> set[str]:
    """Extract external image URLs from one notebook cell."""
    text = cell_source(cell)
    cell_type = cell.get("cell_type")
    urls: set[str] = set()

    if cell_type == "markdown":
        # In prose cells anything declared as an image counts, even without an
        # image extension (CDNs and query-string URLs are common).
        urls.update(MARKDOWN_IMAGE_RE.findall(text))
        urls.update(HTML_ATTR_RE.findall(text))
        # Bare links are only counted when they clearly point at an image file.
        urls.update(u for u in BARE_URL_RE.findall(text) if looks_like_image(u))
    elif cell_type == "code":
        # Code cells reference images too (download helpers, PIL/requests demos),
        # but a bare URL there is usually a doc link, so require an extension.
        urls.update(HTML_ATTR_RE.findall(text))
        urls.update(u for u in BARE_URL_RE.findall(text) if looks_like_image(u))

    return {u.strip().rstrip(".,;") for u in urls if is_external(u.strip())}


def collect_references(notebooks: list[Path], root: Path) -> dict[str, Reference]:
    refs: dict[str, Reference] = {}
    for notebook in notebooks:
        try:
            data = json.loads(notebook.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"warning: skipping unreadable notebook {notebook}: {exc}", file=sys.stderr)
            continue

        rel = notebook.relative_to(root).as_posix()
        for index, cell in enumerate(data.get("cells", [])):
            for url in sorted(extract_urls(cell)):
                refs.setdefault(url, Reference(url=url)).occurrences.append(
                    (rel, index, cell.get("cell_type", "unknown"))
                )
    return refs


def request(url: str, method: str, timeout: float) -> tuple[int, str, str]:
    req = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": USER_AGENT, "Accept": "image/*,*/*"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, "", response.headers.get("Content-Type", "")


def check(ref: Reference, timeout: float, retries: int) -> Reference:
    started = time.monotonic()
    last_status: int | None = None
    last_reason = ""

    for attempt in range(retries + 1):
        # HEAD is cheap, but plenty of CDNs reject it — fall back to GET.
        for method in ("HEAD", "GET"):
            try:
                status, _, content_type = request(ref.url, method, timeout)
                ref.status, ref.content_type, ref.reason = status, content_type, ""
                ref.elapsed_ms = int((time.monotonic() - started) * 1000)
                return ref
            except urllib.error.HTTPError as exc:
                last_status, last_reason = exc.code, f"HTTP {exc.code} {exc.reason}"
                if exc.code in (403, 405, 501) and method == "HEAD":
                    continue  # method not allowed / bot-blocked on HEAD: retry with GET
                break
            except urllib.error.URLError as exc:
                last_status, last_reason = None, f"{type(exc.reason).__name__}: {exc.reason}"
                break
            except Exception as exc:  # noqa: BLE001 - report, never crash the run
                last_status, last_reason = None, f"{type(exc).__name__}: {exc}"
                break

        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))

    ref.status, ref.reason = last_status, last_reason
    ref.elapsed_ms = int((time.monotonic() - started) * 1000)
    return ref


def render_report(refs: list[Reference], notebooks: list[Path], root: Path, elapsed: float, args) -> str:
    broken = [r for r in refs if not r.ok]
    healthy = [r for r in refs if r.ok]
    checked_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    lines = [
        "# Image validation report",
        "",
        f"- **Checked at:** {checked_at}",
        f"- **Source folder:** `{args.notebooks_dir}`",
        f"- **Notebooks scanned:** {len(notebooks)}",
        f"- **Unique external image URLs:** {len(refs)}",
        f"- **Available (200 OK):** {len(healthy)}",
        f"- **Broken:** {len(broken)}",
        f"- **Duration:** {elapsed:.1f}s",
        "",
        f"**Status: {'❌ action required' if broken else '✅ all images available'}**",
        "",
    ]

    if broken:
        lines += [
            "## ❌ Broken images",
            "",
            "| Status | Notebook | Cell | URL | Detail |",
            "| :---: | --- | :---: | --- | --- |",
        ]
        for ref in sorted(broken, key=lambda r: (r.notebooks[0], r.url)):
            for notebook, cell_index, cell_type in ref.occurrences:
                detail = ref.reason or f"HTTP {ref.status}"
                lines.append(
                    f"| ❌ | `{notebook}` | {cell_index} ({cell_type}) | <{ref.url}> | {detail} |"
                )
        lines.append("")

    lines += ["## Full results by notebook", ""]

    by_notebook: dict[str, list[tuple[Reference, int, str]]] = {}
    for ref in refs:
        for notebook, cell_index, cell_type in ref.occurrences:
            by_notebook.setdefault(notebook, []).append((ref, cell_index, cell_type))

    for notebook in sorted(by_notebook):
        entries = sorted(by_notebook[notebook], key=lambda e: (e[1], e[0].url))
        nb_broken = sum(1 for ref, _, _ in entries if not ref.ok)
        heading = "❌" if nb_broken else "✅"
        lines += [
            f"### {heading} `{notebook}`",
            "",
            f"{len(entries)} image reference(s) — {len(entries) - nb_broken} available, {nb_broken} broken.",
            "",
            "| Available | URL | Cell | HTTP | Content-Type |",
            "| :---: | --- | :---: | :---: | --- |",
        ]
        for ref, cell_index, cell_type in entries:
            mark = "✅" if ref.ok else "❌"
            http = str(ref.status) if ref.status is not None else "—"
            ctype = (ref.content_type or ref.reason or "—").split(";")[0]
            lines.append(f"| {mark} | <{ref.url}> | {cell_index} ({cell_type}) | {http} | {ctype} |")
        lines.append("")

    lines += [
        "---",
        "",
        "_Generated by the `validate_image` skill "
        "(`.claude/skills/validate_image/scripts/validate_images.py`)._",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    root = Path(__file__).resolve().parents[4]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebooks-dir", default="colab", help="folder with .ipynb files, relative to repo root (default: colab)")
    parser.add_argument("--output", default="status/image-validation.md", help="report path, relative to repo root (default: status/image-validation.md)")
    parser.add_argument("--notebook", action="append", default=[], help="check only this notebook (repeatable); path relative to repo root")
    parser.add_argument("--timeout", type=float, default=15.0, help="per-request timeout in seconds (default: 15)")
    parser.add_argument("--workers", type=int, default=12, help="concurrent requests (default: 12)")
    parser.add_argument("--retries", type=int, default=1, help="retries per URL before marking it broken (default: 1)")
    parser.add_argument("--root", default=str(root), help="repo root (default: auto-detected)")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    if args.notebook:
        notebooks = [(root / n).resolve() for n in args.notebook]
        missing = [n for n in notebooks if not n.is_file()]
        if missing:
            print(f"error: notebook(s) not found: {', '.join(str(m) for m in missing)}", file=sys.stderr)
            return 2
    else:
        notebooks_dir = root / args.notebooks_dir
        if not notebooks_dir.is_dir():
            print(f"error: notebooks folder not found: {notebooks_dir}", file=sys.stderr)
            return 2
        notebooks = sorted(notebooks_dir.rglob("*.ipynb"))
        notebooks = [n for n in notebooks if ".ipynb_checkpoints" not in n.parts]

    if not notebooks:
        print("error: no notebooks found to validate", file=sys.stderr)
        return 2

    refs = collect_references(notebooks, root)
    print(f"Scanning {len(notebooks)} notebook(s): {len(refs)} unique external image URL(s)", file=sys.stderr)

    started = time.monotonic()
    if refs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(check, ref, args.timeout, args.retries) for ref in refs.values()]
            for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                ref = future.result()
                print(f"  [{done}/{len(futures)}] {'OK ' if ref.ok else 'BAD'} {ref.url}", file=sys.stderr)
    elapsed = time.monotonic() - started

    report = render_report(list(refs.values()), notebooks, root, elapsed, args)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    broken = [r for r in refs.values() if not r.ok]
    print(f"\nReport written to {output.relative_to(root)}", file=sys.stderr)
    print(f"{len(refs) - len(broken)}/{len(refs)} images available, {len(broken)} broken", file=sys.stderr)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
