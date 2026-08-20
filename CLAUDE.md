# tuia-computer-vision

Teaching material for a university Computer Vision course (FCEIA / TUIA): Jupyter notebooks
meant to be opened in Google Colab, plus the images and helper code they depend on.

This is a **content repo, not an application** — there is no build, no test suite, and no
package to install. The deliverable is the notebooks themselves.

## Layout

| Path | What lives there |
| --- | --- |
| [colab/](colab/) | The course notebooks (`.ipynb`), one per class, plus supporting `.docx` material. This is the primary content. |
| [media/pictures/](media/pictures/) | Illustrations, one folder per notebook, named after the notebook file. Served to Colab over raw GitHub URLs. |
| [media/datasets/](media/datasets/) | Sample images used by the notebooks' demos. |
| [helpers/](helpers/) | Standalone Python used by the classes — `timm_train.py` (a patched `geoai` training class) and `compose.yaml` (a pgvector Postgres for the embeddings class). |
| [status/](status/) | Generated health reports; do not hand-edit. `image-validation.md` is overwritten each run, `content/` accumulates one timestamped report per run. |
| [.claude/skills/](.claude/skills/) | Project skills (see below). |

## How images work — read this before touching a notebook

Notebooks are run in Colab, which has no access to this checkout. So illustrations are **not**
referenced by relative path; they are embedded as absolute URLs pointing back at this repo's
`main` branch:

```
https://raw.githubusercontent.com/EzequielMatiasArevalo/tuia-computer-vision/refs/heads/main/media/pictures/<Notebook_Name>/<image>
```

Consequences that matter:

- A new image is invisible in Colab until it is **committed and pushed to `main`**. Adding the
  file locally is not enough.
- Renaming a notebook folder under `media/pictures/` breaks every notebook referencing it.
- Third-party image URLs (news sites, CDNs) rot on their own schedule. Prefer committing a
  copy under `media/pictures/` over linking someone else's asset.

Run the `validate_image` skill after any change to images or image links.

## Working on notebooks

- Edit notebook cells with the notebook-aware editing tool, not by hand-patching the JSON.
- Do **not** strip or regenerate cell outputs unless asked — some outputs are the lesson
  (rendered figures, timing comparisons, model predictions).
- Notebooks are written to run top-to-bottom on a fresh Colab runtime: `!pip install` cells
  belong at the top, and anything GPU-dependent should degrade or say so.
- Prose is a mix of Spanish and English depending on the class. Match the surrounding cell's
  language rather than normalizing it.

## Skills

| Skill | Use it for |
| --- | --- |
| `validate_image` | Check every external image URL in `colab/` returns 200 and write the report to `status/image-validation.md`. |
| `validate_content` | Audit the CV content of the notebooks — outdated model/dataset/library versions, stale "state of the art" claims, unexplained terms, deprecated APIs, missing topics — into `status/content/validate_content_<timestamp>.md`. Takes a notebook as argument, or covers all of `colab/`. Reports only; it never edits a notebook. |

## Repo hygiene

- `.env` is listed in `.gitignore` but is also tracked in git history — do not add secrets to
  it, and flag it if the user wants it cleaned up.
- Large artifacts (`data/`, `models/`, `*.pt`, `media/outputs/`) are gitignored on purpose.
