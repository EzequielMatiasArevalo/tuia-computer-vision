#!/usr/bin/env python3
"""Extract the reviewable content of the course notebooks so a model can audit it.

Reading raw .ipynb JSON is impractical here: the notebooks weigh hundreds of KB
each, almost all of it cell outputs, widget state and base64 images. This script
strips that away and emits two views of a notebook:

  --digest     (default) the prose and code of every cell, outputs replaced by a
               one-line marker, ready to be read end-to-end.
  --inventory  a compact list of the mechanically detectable claims a Computer
               Vision notebook makes and that rot over time: model and dataset
               names with their version suffix, pinned pip packages, hub ids,
               weight files, papers, years, "state of the art" sentences and
               suspected deprecated APIs.

Neither mode judges anything — the judging is the model's job. `--inventory` just
makes sure nothing checkable is missed, and `--digest` is what the review is
actually based on.

Stdlib only, Python 3.10+. Run from the repo root.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# What we look for. These lists are a floor, not a ceiling: they exist so the
# obvious things are never missed. Add to them when a class gains a new topic.
# --------------------------------------------------------------------------- #

# Any of these may legitimately carry a version suffix (YOLOv8, SAM 3, DINOv2,
# ResNet50…), so the surface form actually written in the notebook is reported
# rather than the canonical name — that is the part that goes stale.
CATALOG: dict[str, list[str]] = {
    "backbones & classification": [
        r"alexnet", r"vgg", r"resnet", r"resnext", r"densenet", r"inception", r"googlenet",
        r"mobilenet", r"shufflenet", r"squeezenet", r"efficientnet", r"regnet", r"convnext",
        r"repvgg", r"hrnet", r"nfnet", r"mlp-?mixer",
    ],
    "transformers & foundation models": [
        r"vit", r"vision transformer", r"deit", r"swin", r"beit", r"mae", r"simmim",
        r"dino", r"dinov", r"ijepa", r"i-jepa", r"eva", r"siglip", r"clip", r"open ?clip",
        r"blip", r"llava", r"qwen-?vl", r"florence", r"paligemma", r"internvl", r"smolvlm",
        r"perception encoder",
    ],
    "detection & segmentation": [
        r"r-?cnn", r"fast r-?cnn", r"faster r-?cnn", r"mask r-?cnn", r"ssd", r"retinanet",
        r"fcos", r"centernet", r"yolo", r"yolov", r"yolox", r"yolos", r"efficientdet",
        r"detr", r"deformable detr", r"rt-?detr", r"d-?fine", r"deim", r"dab-?detr",
        r"owl-?vit", r"owlv2", r"grounding ?dino", r"glip", r"sam", r"segment anything",
        r"mask2former", r"maskformer", r"segformer", r"u-?net", r"deeplab", r"fpn", r"bifpn",
        r"pspnet", r"pointrend", r"oneformer",
    ],
    "generative & 3D": [
        r"gan", r"stylegan", r"cyclegan", r"pix2pix", r"vae", r"vq-?vae", r"vq-?gan",
        r"diffusion", r"stable diffusion", r"sdxl", r"sd3", r"flux", r"controlnet",
        r"dreambooth", r"lora", r"latent diffusion", r"ddpm", r"ddim", r"nerf",
        r"gaussian splatting", r"depth anything", r"midas", r"dpt", r"zoedepth",
    ],
    "faces, embeddings & retrieval": [
        r"facenet", r"arcface", r"cosface", r"sphereface", r"triplet loss", r"contrastive loss",
        r"mtcnn", r"retinaface", r"insightface", r"deepface", r"faiss", r"pgvector",
        r"hnsw", r"annoy", r"reranking",
    ],
    "tracking & video": [
        r"sort", r"deepsort", r"bytetrack", r"botsort", r"ocsort", r"kalman", r"hungarian",
        r"optical flow", r"raft", r"videomae", r"x3d", r"slowfast",
    ],
    "datasets": [
        r"imagenet", r"coco", r"ms-?coco", r"pascal voc", r"cifar", r"mnist",
        r"fashion-?mnist", r"open images", r"objects365", r"lvis", r"ade20k", r"cityscapes",
        r"kitti", r"nuscenes", r"bdd100k", r"laion", r"datacomp", r"celeba", r"lfw",
        r"wider ?face", r"vggface", r"ms1m", r"visual genome", r"flickr30k",
        r"conceptual captions", r"sa-1b", r"sa-v", r"div2k", r"oxford-?iiit",
    ],
    "libraries & runtimes": [
        r"pytorch", r"torchvision", r"torchaudio", r"timm", r"transformers", r"diffusers",
        r"accelerate", r"peft", r"datasets", r"ultralytics", r"supervision", r"detectron",
        r"mmdetection", r"mmcv", r"mmsegmentation", r"opencv", r"albumentations", r"kornia",
        r"tensorflow", r"keras", r"jax", r"onnx", r"onnxruntime", r"tensorrt", r"openvino",
        r"coreml", r"tflite", r"triton", r"fiftyone", r"roboflow", r"gradio", r"wandb",
        r"lightning", r"geoai",
    ],
    "training & evaluation concepts": [
        r"batch ?norm", r"layer ?norm", r"group ?norm", r"instance ?norm", r"dropout",
        r"data augmentation", r"mixup", r"cutmix", r"randaugment", r"label smoothing",
        r"transfer learning", r"fine-?tun\w*", r"distillation", r"quantization", r"pruning",
        r"mixed precision", r"amp", r"gradient accumulation", r"early stopping",
        r"cross-?validation", r"iou", r"map", r"mean average precision", r"nms",
        r"soft-?nms", r"anchor", r"f1", r"precision", r"recall", r"confusion matrix",
        r"panoptic quality", r"dice", r"focal loss",
    ],
}

# A version suffix written right after the name is part of the claim: "YOLOv8",
# "YOLO 11", "SAM 3", "ResNet-50", "CLIP ViT-L". Capture it with the name.
VERSION_SUFFIX = r"(?:[ \-]?v?\d+(?:\.\d+)?)?"

CATALOG_RE: dict[str, list[re.Pattern[str]]] = {
    category: [re.compile(rf"\b({pattern}{VERSION_SUFFIX})\b", re.IGNORECASE) for pattern in patterns]
    for category, patterns in CATALOG.items()
}

# Sentences that were true when written and quietly stop being true. These are
# the highest-yield lines in a course notebook, so they are pulled out verbatim.
TIME_SENSITIVE_RE = re.compile(
    r"\b("
    r"state[ -]of[ -]the[ -]art|sota|cutting[ -]edge|"
    r"latest|newest|most recent|currently|nowadays|as of|today|"
    r"actualmente|hoy en d[ií]a|en la actualidad|actualidad|reciente\w*|"
    r"[úu]ltim[oa]s?|el mejor|la mejor|los mejores|las mejores|"
    r"m[áa]s nuev[oa]|m[áa]s modern[oa]|de punta|vanguardia|"
    r"best (?:model|performing|available)|new(?:est)? (?:model|version)|"
    r"deprecat\w*|obsolet\w*|legacy|no longer|ya no"
    r")\b",
    re.IGNORECASE,
)

# Placeholders the author left behind.
TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|WIP|PENDIENTE|COMPLETAR|REVISAR)\b")

# APIs that still parse but have been superseded — cheap to flag, worth checking.
DEPRECATED_HINTS: list[tuple[str, str]] = [
    (r"pretrained\s*=\s*True", "torchvision `pretrained=True` — replaced by `weights=`"),
    (r"\btorch\.load\((?![^)]*weights_only)", "`torch.load` without `weights_only=` — default flipped in torch 2.6"),
    (r"\bVariable\s*\(", "`torch.autograd.Variable` — removed concept, use tensors"),
    (r"\bnp\.(?:float|int|bool|object|str)\b(?!\d)", "removed NumPy aliases (`np.float`, `np.int`…)"),
    (r"\btf\.Session\b|\btf\.placeholder\b", "TensorFlow 1.x graph API"),
    (r"keras\.preprocessing", "`keras.preprocessing` — deprecated, use `keras.utils` / `tf.data`"),
    (r"AutoModelWithLMHead", "`AutoModelWithLMHead` — removed from transformers"),
    (r"sklearn\.externals", "`sklearn.externals` — removed"),
    (r"\bimp\b\s*\.|\bimport imp\b", "`imp` module — removed in Python 3.12"),
    (r"scipy\.misc\.(?:imread|imresize|imsave)", "`scipy.misc` image helpers — removed"),
    (r"\.grid\(b=", "matplotlib `grid(b=…)` — removed keyword"),
    (r"cv2\.cv\b", "OpenCV 1.x `cv2.cv` namespace"),
    (r"use_auth_token", "`use_auth_token` — renamed to `token` in huggingface_hub"),
]
DEPRECATED_RE = [(re.compile(pattern), label) for pattern, label in DEPRECATED_HINTS]

PIP_RE = re.compile(r"^\s*[!%]\s*(?:uv\s+)?pip(?:3)?\s+install\s+(.+)$", re.MULTILINE)
PIP_FLAG_RE = re.compile(r"^-")
HUB_ID_RE = re.compile(r"[\"']([A-Za-z0-9][\w.\-]{1,60}/[\w.\-]{1,80})[\"']")
# `results/plot.png` looks exactly like `facebook/sam3` to the regex above.
NOT_A_HUB_ID_RE = re.compile(
    r"\.(?:png|jpe?g|gif|svg|webp|mp4|avi|mov|txt|csv|tsv|json|ya?ml|py|ipynb|zip|tar|gz|log|md|html)$",
    re.IGNORECASE,
)
WEIGHTS_RE = re.compile(r"\b([\w\-.]+\.(?:pt|pth|onnx|safetensors|engine|tflite|ckpt|pb|weights))\b", re.IGNORECASE)
ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
LINK_RE = re.compile(r"https?://[^\s\"'<>)\]]+")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".avif", ".ico")


@dataclass
class NotebookContent:
    path: str
    cells: list[dict] = field(default_factory=list)

    @property
    def n_markdown(self) -> int:
        return sum(1 for c in self.cells if c.get("cell_type") == "markdown")

    @property
    def n_code(self) -> int:
        return sum(1 for c in self.cells if c.get("cell_type") == "code")


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def output_marker(cell: dict) -> str:
    """One line standing in for the outputs we stripped."""
    outputs = cell.get("outputs") or []
    if not outputs:
        return ""

    kinds: Counter[str] = Counter()
    error = ""
    for output in outputs:
        kind = output.get("output_type", "unknown")
        if kind == "error":
            error = f"{output.get('ename', 'Error')}: {output.get('evalue', '')}".strip()
            kinds["error"] += 1
        elif kind == "stream":
            kinds["stream"] += 1
        else:
            data = output.get("data", {})
            if any(mime.startswith("image/") for mime in data):
                kinds["image"] += 1
            elif "text/html" in data:
                kinds["html"] += 1
            else:
                kinds["text"] += 1

    summary = ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))
    if error:
        summary += f" — {error[:200]}"
    return f"[outputs stripped: {summary}]"


def load(path: Path, root: Path) -> NotebookContent | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        print(f"warning: skipping unreadable notebook {path}: {exc}", file=sys.stderr)
        return None
    return NotebookContent(path=path.relative_to(root).as_posix(), cells=data.get("cells", []))


# --------------------------------------------------------------------------- #
# Digest
# --------------------------------------------------------------------------- #

def render_digest(notebook: NotebookContent, max_cell_chars: int, keep_outputs: bool) -> str:
    lines = [
        f"# Digest: `{notebook.path}`",
        "",
        f"_{len(notebook.cells)} cells — {notebook.n_markdown} markdown, {notebook.n_code} code. "
        "Cell outputs are stripped; the number in brackets is the cell index._",
        "",
    ]

    for index, cell in enumerate(notebook.cells):
        cell_type = cell.get("cell_type", "unknown")
        text = cell_source(cell).rstrip()
        if not text and not (keep_outputs and cell.get("outputs")):
            continue

        truncated = ""
        if len(text) > max_cell_chars:
            text = text[:max_cell_chars]
            truncated = f"\n… [truncated at {max_cell_chars} chars]"

        lines.append(f"## [{index}] {cell_type}")
        lines.append("")
        if cell_type == "code":
            lines += ["```python", text + truncated, "```"]
        else:
            lines.append(text + truncated)
        if keep_outputs:
            marker = output_marker(cell)
            if marker:
                lines += ["", f"`{marker}`"]
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #

def scan_catalog(notebook: NotebookContent) -> dict[str, dict[str, list[int]]]:
    """category -> canonical form -> (surface forms written, cell indices).

    Matches are grouped case-insensitively, otherwise `YOLO11`, `yolo11` and
    `Yolo11` each claim their own row. URLs are removed first: an image named
    `hungarian-2.png` is not the notebook talking about the Hungarian algorithm.
    """
    found: dict[str, dict[str, tuple[set[str], list[int]]]] = {}
    for index, cell in enumerate(notebook.cells):
        text = LINK_RE.sub(" ", cell_source(cell))
        if not text:
            continue
        for category, patterns in CATALOG_RE.items():
            for pattern in patterns:
                for match in pattern.findall(text):
                    surface = re.sub(r"\s+", " ", match).strip()
                    forms, cells = found.setdefault(category, {}).setdefault(
                        surface.casefold(), (set(), [])
                    )
                    forms.add(surface)
                    if index not in cells:
                        cells.append(index)

    return {
        category: {" / ".join(sorted(forms)): cells for forms, cells in entries.values()}
        for category, entries in found.items()
    }


def scan_lines(notebook: NotebookContent, pattern: re.Pattern[str], cell_types: tuple[str, ...]) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") not in cell_types:
            continue
        for line in cell_source(cell).splitlines():
            stripped = line.strip()
            if stripped and pattern.search(stripped):
                hits.append((index, stripped))
    return hits


def scan_pip(notebook: NotebookContent) -> list[tuple[int, str]]:
    packages: list[tuple[int, str]] = []
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        for spec_line in PIP_RE.findall(cell_source(cell)):
            for token in spec_line.replace("\\", " ").split():
                if PIP_FLAG_RE.match(token) or token in {"install", "pip", "uv"}:
                    continue
                packages.append((index, token.strip("\"'")))
    return packages


def scan_code_tokens(notebook: NotebookContent, pattern: re.Pattern[str]) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        for match in pattern.findall(cell_source(cell)):
            cells = found.setdefault(match, [])
            if index not in cells:
                cells.append(index)
    return found


def scan_deprecated(notebook: NotebookContent) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        text = cell_source(cell)
        for pattern, label in DEPRECATED_RE:
            match = pattern.search(text)
            if match:
                hits.append((index, label, match.group(0).strip()))
    return hits


def scan_links(notebook: NotebookContent) -> dict[str, list[int]]:
    """Non-image external links — papers, docs, blog posts. Images are the
    `validate_image` skill's job, so they are left out here."""
    found: dict[str, list[int]] = {}
    for index, cell in enumerate(notebook.cells):
        for url in LINK_RE.findall(cell_source(cell)):
            url = url.rstrip(".,;)")
            if url.lower().split("?")[0].endswith(IMAGE_EXT):
                continue
            cells = found.setdefault(url, [])
            if index not in cells:
                cells.append(index)
    return found


def cells_str(indices: list[int], limit: int = 8) -> str:
    shown = ", ".join(str(i) for i in indices[:limit])
    return shown + (f", +{len(indices) - limit} more" if len(indices) > limit else "")


def render_inventory(notebook: NotebookContent, max_links: int) -> str:
    lines = [
        f"# Inventory: `{notebook.path}`",
        "",
        f"_{len(notebook.cells)} cells — {notebook.n_markdown} markdown, {notebook.n_code} code._",
        "",
        "Mechanically detected claims. Nothing here is a finding on its own — it is the "
        "list of things whose current state is worth checking.",
        "",
    ]

    catalog = scan_catalog(notebook)
    lines += ["## Topics mentioned (as written in the notebook)", ""]
    if catalog:
        lines += ["| Category | Surface form | Cells |", "| --- | --- | --- |"]
        for category in CATALOG:
            forms = catalog.get(category)
            if not forms:
                continue
            for surface in sorted(forms, key=lambda s: (-len(forms[s]), s.lower())):
                lines.append(f"| {category} | `{surface}` | {cells_str(forms[surface])} |")
    else:
        lines.append("_None detected._")
    lines.append("")

    packages = scan_pip(notebook)
    lines += ["## Installed packages", ""]
    if packages:
        lines += ["| Spec | Cell | Pinned |", "| --- | :---: | :---: |"]
        for index, spec in packages:
            pinned = any(op in spec for op in ("==", ">=", "<=", "~=", "<", ">"))
            lines.append(f"| `{spec}` | {index} | {'yes' if pinned else 'no'} |")
    else:
        lines.append("_No `pip install` cells._")
    lines.append("")

    hub_ids = {identifier: cells for identifier, cells in scan_code_tokens(notebook, HUB_ID_RE).items()
               if not NOT_A_HUB_ID_RE.search(identifier)}
    weights = scan_code_tokens(notebook, WEIGHTS_RE)
    lines += ["## Model identifiers and checkpoints", ""]
    if hub_ids or weights:
        lines += ["| Identifier | Kind | Cells |", "| --- | --- | --- |"]
        for identifier in sorted(hub_ids):
            lines.append(f"| `{identifier}` | hub-style id | {cells_str(hub_ids[identifier])} |")
        for identifier in sorted(weights):
            lines.append(f"| `{identifier}` | weights file | {cells_str(weights[identifier])} |")
    else:
        lines.append("_None detected._")
    lines.append("")

    time_sensitive = scan_lines(notebook, TIME_SENSITIVE_RE, ("markdown", "code"))
    lines += ["## Time-sensitive statements", "",
              "_Claims phrased as \"current\", \"latest\", \"state of the art\"… — verify each one._", ""]
    if time_sensitive:
        for index, line in time_sensitive:
            lines.append(f"- **[{index}]** {line[:400]}")
    else:
        lines.append("_None detected._")
    lines.append("")

    years = Counter()
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") == "markdown":
            years.update(YEAR_RE.findall(cell_source(cell)))
    lines += ["## Years cited in prose", ""]
    lines.append(", ".join(f"{year} ({count})" for year, count in sorted(years.items())) if years else "_None._")
    lines.append("")

    deprecated = scan_deprecated(notebook)
    lines += ["## Possibly deprecated APIs", ""]
    if deprecated:
        lines += ["| Cell | Match | Note |", "| :---: | --- | --- |"]
        for index, label, match in deprecated:
            lines.append(f"| {index} | `{match[:60]}` | {label} |")
    else:
        lines.append("_None detected._")
    lines.append("")

    todos = scan_lines(notebook, TODO_RE, ("markdown", "code"))
    lines += ["## Author placeholders", ""]
    if todos:
        for index, line in todos:
            lines.append(f"- **[{index}]** {line[:300]}")
    else:
        lines.append("_None._")
    lines.append("")

    papers = scan_code_tokens(notebook, ARXIV_RE)
    arxiv_all: dict[str, list[int]] = dict(papers)
    for index, cell in enumerate(notebook.cells):
        for paper_id in ARXIV_RE.findall(cell_source(cell)):
            cells = arxiv_all.setdefault(paper_id, [])
            if index not in cells:
                cells.append(index)
    lines += ["## arXiv papers referenced", ""]
    if arxiv_all:
        for paper_id in sorted(arxiv_all):
            lines.append(f"- `{paper_id}` (cells {cells_str(arxiv_all[paper_id])}) — https://arxiv.org/abs/{paper_id}")
    else:
        lines.append("_None._")
    lines.append("")

    links = scan_links(notebook)
    lines += ["## Other external links (non-image)", ""]
    if links:
        for url in sorted(links)[:max_links]:
            lines.append(f"- {url} (cells {cells_str(links[url])})")
        if len(links) > max_links:
            lines.append(f"- _… {len(links) - max_links} more, re-run with a higher `--max-links`._")
    else:
        lines.append("_None._")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #

def resolve_notebooks(args, root: Path) -> list[Path]:
    if args.notebook:
        notebooks = []
        for name in args.notebook:
            candidate = (root / name).resolve()
            if not candidate.is_file():
                # Allow a bare notebook name as well as a repo-relative path.
                matches = sorted((root / args.notebooks_dir).glob(f"{name}*"))
                matches = [m for m in matches if m.suffix == ".ipynb"]
                if len(matches) == 1:
                    candidate = matches[0]
            notebooks.append(candidate)
        missing = [n for n in notebooks if not n.is_file()]
        if missing:
            print(f"error: notebook(s) not found: {', '.join(str(m) for m in missing)}", file=sys.stderr)
            sys.exit(2)
        return notebooks

    notebooks_dir = root / args.notebooks_dir
    if not notebooks_dir.is_dir():
        print(f"error: notebooks folder not found: {notebooks_dir}", file=sys.stderr)
        sys.exit(2)
    found = sorted(notebooks_dir.rglob("*.ipynb"))
    return [n for n in found if ".ipynb_checkpoints" not in n.parts]


def main() -> int:
    root = Path(__file__).resolve().parents[4]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", action="append", default=[],
                        help="only this notebook (repeatable); repo-relative path or bare filename. "
                             "Omit to cover every notebook in the folder.")
    parser.add_argument("--notebooks-dir", default="colab", help="folder with .ipynb files (default: colab)")
    parser.add_argument("--digest", action="store_true", help="emit the cell-by-cell digest (default when no mode is given)")
    parser.add_argument("--inventory", action="store_true", help="emit the mechanical inventory instead of the digest")
    parser.add_argument("--list", action="store_true", help="list the notebooks in scope and the report path to write, then exit")
    parser.add_argument("--report-path", action="store_true", help="print the timestamped report path to write, then exit")
    parser.add_argument("--max-cell-chars", type=int, default=4000, help="truncate long cells in the digest (default: 4000)")
    parser.add_argument("--max-links", type=int, default=60, help="max external links listed per notebook in the inventory (default: 60)")
    parser.add_argument("--keep-output-markers", action="store_true", help="keep the one-line `[outputs stripped: …]` marker in the digest")
    parser.add_argument("--output", help="write to this file (repo-relative) instead of stdout")
    parser.add_argument("--root", default=str(root), help="repo root (default: auto-detected)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report_path = f"status/content/validate_content_{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"

    if args.report_path:
        print(report_path)
        return 0

    notebooks = resolve_notebooks(args, root)
    if not notebooks:
        print("error: no notebooks found", file=sys.stderr)
        return 2

    if args.list:
        print(f"{len(notebooks)} notebook(s) in scope:")
        for notebook in notebooks:
            size_kb = notebook.stat().st_size // 1024
            content = load(notebook, root)
            cells = len(content.cells) if content else 0
            print(f"  {notebook.relative_to(root).as_posix()}  ({cells} cells, {size_kb} KB on disk)")
        print(f"\nWrite the report to: {report_path}")
        return 0

    chunks: list[str] = []
    for notebook in notebooks:
        content = load(notebook, root)
        if content is None:
            continue
        chunks.append(render_inventory(content, args.max_links) if args.inventory
                      else render_digest(content, args.max_cell_chars, args.keep_output_markers))

    text = "\n\n---\n\n".join(chunks)

    if args.output:
        destination = Path(args.output)
        destination = destination if destination.is_absolute() else root / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        shown = destination.relative_to(root) if destination.is_relative_to(root) else destination
        print(f"written to {shown} ({len(text)} chars)", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
