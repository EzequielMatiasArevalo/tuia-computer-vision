---
name: validate_image
description: Validate that every externally-hosted image referenced by the notebooks in colab/ still returns HTTP 200, and write a markdown report to status/ listing each image URL, its source notebook, and an availability check mark. Use when the user asks to check/validate images, find broken image links, verify notebook illustrations still load, or refresh the image status report.
---

# Validate notebook image links

The course notebooks in [colab/](../../../colab/) render most of their illustrations from
external URLs — mostly `raw.githubusercontent.com` links back into **this same repo's `main`
branch** (`media/pictures/<notebook-name>/…`), plus some third-party news and CDN images.
Those links rot silently: the notebook opens fine in Colab but shows a broken placeholder.
This skill checks them all and records the result.

## Run it

```bash
python3 .claude/skills/validate_image/scripts/validate_images.py
```

Runs from the repo root, needs no dependencies (Python 3.10+ stdlib only), and writes
`status/image-validation.md`. Takes ~15–40s for the full set.

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--notebook colab/Class_2_Modern_CNN.ipynb` | Check one notebook (repeatable) instead of the whole folder |
| `--notebooks-dir <dir>` | Scan a different folder (default `colab`) |
| `--output status/<name>.md` | Write the report somewhere else |
| `--timeout 25` | Raise the per-request timeout for slow hosts |
| `--workers 6` | Lower concurrency if a host starts rate-limiting |
| `--retries 2` | More retries before declaring a URL broken |

Exit code is `0` when every image resolves and `1` when at least one is broken, so it can
be dropped into CI or a pre-commit hook unchanged.

## What counts as an image reference

- **Markdown cells** — `![alt](url)`, and `src` / `data-src` / `poster` attributes on
  `<img>`, `<source>`, `<embed>`. Any URL declared as an image counts, even without a file
  extension (CDN and query-string URLs are common here).
- **Code cells** — `<img>` attributes, plus bare URLs whose path ends in an image extension
  (`.png .jpg .jpeg .gif .svg .webp .bmp .tif .tiff .avif .ico`). A bare non-image URL in a
  code cell is treated as a documentation link and skipped.
- Local/relative paths and `data:` URIs are ignored — this skill is only about external URLs.

Each URL is requested with `HEAD` first and retried with `GET` when the host rejects HEAD
(`403/405/501`, which several CDNs return for bots). A browser `User-Agent` is sent.

## The report

`status/image-validation.md` is overwritten on each run and contains:

1. A summary header — timestamp, notebooks scanned, unique URLs, available vs broken counts.
2. A **Broken images** table (only when something failed) with notebook, cell index, URL, and
   the failure detail (`HTTP 404`, timeout, DNS error…).
3. **Full results by notebook** — one table per notebook with an ✅/❌ check mark, the URL,
   the cell it lives in, the HTTP status, and the content type.

## After running

Report the counts to the user and link the file. If anything is broken, name the affected
notebooks and offer to fix them — the usual repairs are:

- **`raw.githubusercontent.com` 404** — these resolve to this repo's `main` branch. Either
  the file exists locally under [media/pictures/](../../../media/pictures/) but was never
  committed and pushed to `main`, or it was renamed. Compare the URL path against the local
  folder first; pushing the missing asset is usually the fix, not editing the notebook.
- **Third-party 4xx/5xx** — the outlet rotated or pulled the asset. Commit a copy under
  `media/pictures/<notebook-name>/` and repoint the notebook at the raw URL for it, so the
  slide stops depending on someone else's CDN.
- **TLS / certificate errors** — the host is misconfigured for that domain (e.g.
  `images.cocodataset.org` over HTTPS). Check whether a working variant exists (the `http://`
  form, or a different host) before treating it as dead.

Do not edit notebooks as part of validating — only when the user asks for the fix.
