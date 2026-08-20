# What to look for

Eight dimensions. Go through them per notebook; the inventory feeds 1, 5 and 6, the rest come
from reading the digest.

## 1. Superseded versions

A model, dataset, library or technique presented at a generation that has been replaced.
Families that move fast and are already in this course: YOLO, SAM, DINO, ViT/DeiT/Swin, DETR
and its descendants (RT-DETR, D-FINE, DEIM), CLIP/SigLIP, Stable Diffusion, Depth Anything,
the trackers (SORT → ByteTrack → BoT-SORT), `transformers` / `ultralytics` / `timm` APIs.

Report it when: the notebook calls it the current one, the newer generation changes the
lesson, or a student copying the code would get a deprecation warning or worse results.
Do **not** report it when an older architecture is being taught as history or as a building
block — that is the curriculum working correctly.

## 2. Stale time-bound claims

"State of the art", "actualmente el mejor", "el modelo más reciente", "as of 2024", benchmark
numbers and leaderboard positions. Each one had an expiry date the author did not write down.
Check it and either confirm it or propose the corrected sentence.

## 3. Content used but never explained

A term, acronym, loss, metric or architectural block that the notebook relies on without ever
defining — `mAP@0.5:0.95`, NMS, focal loss, query tokens, CFG scale, LoRA rank. Cite the cell
where it first appears unexplained. This is the most common finding in practice and the
cheapest to fix.

## 4. Missing content worth adding

Topics inside the class's own scope that are absent and that a current CV course would be
expected to cover — an evaluation step for a model that is only ever trained, a modern
alternative to the technique being taught, a failure mode the demo hides, a deployment or
quantization note in a class about inference cost. Justify each one by what the student
cannot do without it; do not propose redesigning the course.

## 5. Code that will not run as written

Deprecated or removed APIs, unpinned installs that now resolve to an incompatible major,
checkpoints or Hub ids that 404, datasets no longer distributed, GPU-only code with no note
that it needs one, cells that depend on a previous cell that was deleted, hardcoded paths to
files that are not in the repo. Colab is the target runtime — "works on my machine" is not
the bar.

## 6. Broken or rotted references

Papers, docs and blog links that no longer resolve, arXiv ids pointing at a superseded
version of a paper, links to a library page that has moved. Image URLs are **out of scope**
here — that is `validate_image`.

## 7. Incorrect or imprecise explanations

Statements that are wrong, or right-but-misleading: a formula with a wrong term, a claim that
batch norm and layer norm normalize the same axes, an architecture diagram described
incorrectly in the prose, a confusion between two metrics. These outrank everything else —
a student will carry a wrong mental model far longer than a stale version number.

## 8. Pedagogical gaps

Inconsistent language inside a single explanation (the repo mixes Spanish and English **by
class**, which is fine — mixing them mid-explanation is not), an exercise with no solution or
solution with no exercise, a section that promises something it never delivers, dangling
references to "as we saw in class 3" for material that is not there, unexplained magic
constants.

# Severity

| Mark | Meaning |
| :---: | --- |
| 🔴 | Wrong, or broken for a student running the notebook today: incorrect explanation, code that fails, dead checkpoint. |
| 🟠 | Outdated: superseded version presented as current, expired claim, deprecated API that still works. |
| 🟡 | Gap: content used without explanation, missing topic within scope, missing exercise or evaluation. |
| 🔵 | Improvement: nice-to-have addition, clearer phrasing, an extra reference. |

# Confidence

`verified` — confirmed against a source you fetched this run, URL included.
`unverified` — could not confirm; say what you could not check and why.

Never present an unverified version claim as fact. An honest "as of this run I could not
confirm whether X released" is more useful to the author than a confident wrong date.
