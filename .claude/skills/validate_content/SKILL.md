---
name: validate_content
description: Audit the Computer Vision content of the notebooks in colab/ — outdated model, dataset, architecture or library versions, topics that are used but never explained, claims that have gone stale, deprecated APIs and missing material worth adding — and write the findings to status/content/validate_content_<timestamp>.md. Runs over the whole colab/ folder by default, or over a single notebook when one is given as an argument. Use when the user asks to validate/review notebook content, check whether the course material is up to date, find gaps or missing explanations, or refresh the content report.
---

# Validate notebook content

The `validate_image` skill checks that the illustrations still load. This one checks whether
the **teaching content itself** still holds up: a class that presents YOLOv8 as the current
release, a notebook that benchmarks against a superseded checkpoint, an acronym used in three
cells and defined in none, a `pip install` that no longer resolves. None of that breaks
loudly — the notebook runs fine and quietly teaches something out of date.

The output is a report, never an edit. Fixing is a separate, explicitly requested job.

## Scope

- **No argument → every notebook in [colab/](../../../colab/).** This is the default and the
  common case.
- **An argument → only that notebook.** Accept either a repo-relative path
  (`colab/Class_2_Modern_CNN.ipynb`) or a bare name (`Class_2_Modern_CNN`); the helper script
  resolves both. Repeat `--notebook` for several.

Confirm the scope and get the report path in one call:

```bash
python3 .claude/skills/validate_content/scripts/extract_notebook_content.py --list
```

## Step 1 — read the mechanical inventory

```bash
python3 .claude/skills/validate_content/scripts/extract_notebook_content.py --inventory
```

Stdlib only, instant, ~35k chars for the full folder. It reports, per notebook: every model /
architecture / dataset / library name **in the surface form the notebook actually wrote it**
(`YOLOv11`, `SAM 3`, `DINOv2` — the version suffix is the part that rots), installed packages
and whether they are pinned, Hugging Face ids and checkpoint files, sentences phrased as
"state of the art" / "actualmente" / "latest", years cited in prose, suspected deprecated
APIs, author `TODO`s, arXiv ids and non-image external links.

Nothing in there is a finding on its own. It is the list of things whose current state you
are obliged to check.

## Step 2 — read the content

```bash
python3 .claude/skills/validate_content/scripts/extract_notebook_content.py \
  --digest --notebook colab/Class_2_Modern_CNN.ipynb
```

The digest is the notebook with outputs, widget state and base64 images stripped — prose and
code only, each cell labelled with its index so findings can cite `cell 42`. Per notebook it
runs 20k–90k chars; the whole folder at once is ~450k, which is why full-folder runs are
split up (below).

Read the digest. The inventory tells you what to verify; the digest is where you notice what
is **missing, wrong or unexplained**, which no regex will find for you.

## Step 3 — verify before claiming

A finding that says "there is a newer version" is worthless unless it is true today. Your
training data has a cutoff and this repo is a moving target, so **use WebSearch / WebFetch to
confirm every version, release and availability claim** before it goes in the report — the
current release of a model family, whether a checkpoint is still on the Hub, whether a
package still exposes an API, whether a dataset is still distributed.

Two rules that keep this report trustworthy:

- **Verified or flagged.** Anything you could not confirm goes in with an explicit
  `unverified` confidence, not as a fact.
- **Newer is not automatically better for teaching.** ResNet and LeNet belong in a course
  regardless of their age. Flag an old model as a problem only when the notebook presents it
  as *current*, or when the newer thing changes what a student should learn. Say which of the
  two it is.

See [references/review-dimensions.md](references/review-dimensions.md) for the eight things to
look for and the severity scale.

## Step 4 — write the report

Path — take it from the script so the timestamp format stays consistent:

```bash
python3 .claude/skills/validate_content/scripts/extract_notebook_content.py --report-path
# status/content/validate_content_20260809-233144.md
```

Write it with the Write tool, following
[references/report-template.md](references/report-template.md). Reports accumulate in
[status/content/](../../../status/content/) — never overwrite an earlier one, the history of
what was flagged when is the point.

Every finding must carry: notebook, cell index, severity, what the notebook says now, what is
actually current (with a source URL), why it matters for the class, and a concrete suggested
fix. A finding without a cell index and a source is not a finding.

## Full-folder runs

Eight notebooks of digest do not fit comfortably in one context. Run the inventory for
everything at once, then handle **one notebook per subagent, all dispatched in a single
message** so they run concurrently. Give each subagent: the notebook path, the command to
produce its own digest, [references/review-dimensions.md](references/review-dimensions.md),
the instruction to verify with WebSearch, and the finding fields above — and have it return
findings as markdown table rows, not prose.

Then merge: dedupe findings that repeat across notebooks (the same stale claim often appears
in three classes — report it once with all locations), sort by severity, and write the single
report yourself.

## After running

Tell the user the counts by severity, name the notebooks with 🔴 or 🟠 findings, and link the
report file. Offer to apply the fixes — do not apply them as part of validating, and do not
edit any notebook in `colab/` during this skill.

Cross-check worth mentioning when relevant: broken *image* links are the `validate_image`
skill's job and are deliberately excluded from this report; if the content review turns up
dead paper or documentation links, those do belong here.
