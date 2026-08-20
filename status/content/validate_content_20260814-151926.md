# Content validation report

- **Checked at:** 2026-08-14 15:19:26 -0300
- **Scope:** `colab/class_5_Dino_Detr_and_Swin.ipynb`
- **Notebooks reviewed:** 1 (79 cells — 56 markdown, 23 code)
- **Findings:** 29 — 🔴 6 critical · 🟠 4 outdated · 🟡 15 gaps · 🔵 4 improvements
- **Verified against sources:** 5 of the 8 findings that need an external source; the other 21 are verifiable by reading the notebook itself
- **Reviewed by:** validate_content skill

**Status: 🔴 action required**

## Summary

The DINOv3 theory section is the strongest material in the notebook — KoLeo, Gram anchoring, RoPE
and register tokens are explained at a level most course material skips, and the architecture
comparison table (cell 47) matches Meta's published numbers exactly (ViT-H+ 840M / 1280-dim /
20 heads, ConvNeXt-L ~196M). Every model id, checkpoint and dataset URL the notebook depends on
still resolves, so the notebook is not going to fail on a dead reference.

What has drifted is the *factual* layer around that theory. Two claims about DINOv2/DINOv3 are
simply wrong (the teacher model, and the training data), and both are the kind of thing a
student repeats in an exam. Three code cells contain bugs that silently produce misleading
output rather than errors — most seriously, the validation loaders inherit the training
augmentations, which makes the "this indicates overfitting" conclusion in cell 31 unsupported by
its own numbers.

Fix **D5-01** (DINOv2's teacher is ViT-g/14, not ViT-L) first: it is a one-word edit and it is
the claim most likely to be repeated. **D5-03** (validation augmentation) is the second priority
because it invalidates a written conclusion, not just a number.

The DETR half of the notebook is thin relative to the DINO half: it shows the API and the output
but never explains object queries, bipartite matching or the no-object class that the code
explicitly slices off in cell 77.

## 🔴 Critical

### Wrong facts

**D5-01 · 🔴 DINOv2's teacher is ViT-g/14, not ViT-L** — `colab/class_5_Dino_Detr_and_Swin.ipynb` cell 39
- **Now:** "DINOv2's teacher was its own largest ViT-L. DINOv3 distills from a **ViT-7B teacher**"
- **Current:** DINOv2's largest model is **ViT-g/14 (1.1B parameters, 40 blocks)**. ViT-L/14 (303M)
  is one of the *distilled students*, alongside ViT-B/14 (86M) and ViT-S/14 (21M). The paper's own
  wording: smaller models are "distilled from the largest model, the ViT-g, instead of training
  them from scratch".
- **Why:** The sentence exists to set up the DINOv2 → DINOv3 contrast (own-model teacher vs. a
  separately trained 7B teacher). With ViT-L in the slot the jump looks like 300M → 7B rather than
  1.1B → 6.7B, and it contradicts cell 11, which correctly says the notebook is talking about
  "the Dinov2-g model".
- **Fix:** Replace with: "DINOv2's teacher was its own largest model, **ViT-g/14 (1.1B params)**.
  DINOv3 distills from a **ViT-7B teacher** (6.7B params) — a model trained separately at much
  larger scale." Same edit applies to the summary block a few lines below in the same cell.
- **Source:** https://arxiv.org/pdf/2304.07193 · https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md · verified

**D5-02 · 🔴 DINOv3 was not trained on SA-V video frames** — `colab/class_5_Dino_Detr_and_Swin.ipynb` cell 33
- **Now:** "Same pipeline but massively scaled, includes video frames (SA-V dataset)"
- **Current:** LVD-1689M is entirely image-based. Per §3.1 of the DINOv3 paper it has three
  components: (1) 1,689M images from hierarchical k-means clustering over a **17-billion-image
  Instagram pool** embedded with DINOv2, (2) retrieval-curated images from seed datasets, and
  (3) public CV datasets — **ImageNet-1k, ImageNet-22k and Mapillary Street-level Sequences**.
  SA-V is the *SAM 2* video segmentation dataset and is not part of DINOv3 pretraining.
- **Why:** It plants the idea that DINOv3's dense-feature quality comes from video/temporal
  supervision. It does not — it comes from scale plus Gram anchoring, which is the actual point
  the notebook makes two cells later (cell 38).
- **Fix:** Replace the DINOv3 curation cell with: "Hierarchical k-means over a 17B-image web pool,
  plus retrieval-curated images, plus ImageNet-1k/22k and Mapillary Street-level Sequences."
  Drop the SA-V mention entirely.
- **Source:** https://arxiv.org/html/2508.10104v1 §3.1 · verified

**D5-03 · 🔴 "KNN" heading over K-Means code** — `colab/class_5_Dino_Detr_and_Swin.ipynb` cells 22, 23
- **Now:** cell 22 heading — "## Segment using Dino & KNN"; cell 23 runs
  `KMeans(n_clusters=n_clusters, n_init="auto", max_iter=1000)`
- **Current:** k-nearest-neighbours (supervised, needs labels, classifies a query point by its
  neighbours) and k-means (unsupervised clustering) are different algorithms. The cell body,
  the plot title (`"k-means segmentation (3 clusters)"`) and the prose in cell 22
  ("comparing the results with K-Means clustering") all say k-means; only the heading says KNN.
- **Why:** DINOv2's k-NN *evaluation protocol* is a real and separate thing in the literature, so
  a student reading this heading has a plausible reason to believe the notebook is demonstrating
  it. Conflating the two is the kind of error that survives the whole course.
- **Fix:** Change the heading to `## Segment using Dino & K-Means`.
- **Source:** notebook-internal · verified by reading cells 22–23

### Code that silently produces wrong output

**D5-04 · 🔴 Validation and test sets are evaluated with training augmentations** — `colab/class_5_Dino_Detr_and_Swin.ipynb` cells 13, 30, 31, 65
- **Now:** cell 13 builds `val_tfms` (Resize 256 → CenterCrop 224) and then never uses it:
  ```python
  images = datasets.ImageFolder(IMAGENET_ROOT, train_tfms)
  sub_train, sub_val, sub_test = torch.utils.data.random_split(images, [...])
  ```
- **Current:** `random_split` returns `Subset` views over the *same* `ImageFolder`, so all three
  splits share `train_tfms` — `RandomResizedCrop(224)` + `RandomHorizontalFlip()`. Validation
  accuracy is therefore measured on randomly cropped and flipped images, and differs run to run.
- **Why:** This directly undercuts a written conclusion. Cell 31 says "validation performance does
  not improve at the same pace, which indicates overfitting" — but the gap it points at is at
  least partly the augmentation, not generalization. A student learns the wrong diagnostic reflex.
  It also affects the Swin fine-tune (cell 65), which reuses the same loaders.
- **Fix:** Build two `ImageFolder` instances and split by index so the transform differs per split:
  ```python
  train_ds = datasets.ImageFolder(IMAGENET_ROOT, train_tfms)
  eval_ds  = datasets.ImageFolder(IMAGENET_ROOT, val_tfms)
  g = torch.Generator().manual_seed(SEED)
  idx = torch.randperm(len(train_ds), generator=g).tolist()
  n_tr, n_va = int(0.7*len(idx)), int(0.2*len(idx))
  sub_train = Subset(train_ds, idx[:n_tr])
  sub_val   = Subset(eval_ds,  idx[n_tr:n_tr+n_va])
  sub_test  = Subset(eval_ds,  idx[n_tr+n_va:])
  ```
  Then soften cell 31 — with a 2-epoch linear probe on a small subset, the honest statement is
  "the head has not converged yet", not "overfitting".
- **Source:** notebook-internal · verified by reading cells 13, 30, 31

**D5-05 · 🔴 Panoptic masks are all resized to the first image's dimensions** — `colab/class_5_Dino_Detr_and_Swin.ipynb` cell 71
- **Now:** inside `for inputs in processed_images:` —
  `target_sizes = [pil_images[0].size[::-1]]  # (H, W)`
- **Current:** `pil_images[0]` is hardcoded, so images 1–5 have their segmentation upsampled to
  image 0's height and width. COCO val images are a mix of landscape and portrait, so this
  stretches masks across the wrong aspect ratio for most of the batch.
- **Why:** The visual result in cell 72 still *looks* plausible, because `visualize_panoptic` does
  `image.resize((w, h))` — it distorts the photo to match the distorted mask. The class is left
  looking at squashed predictions and no one notices. Worse, it teaches that `target_sizes` is
  boilerplate rather than the per-image argument it is.
- **Fix:** Track the index and use the matching image:
  ```python
  for img, inputs in zip(pil_images, processed_images):
      ...
      target_sizes = [img.size[::-1]]   # (H, W) for THIS image
  ```
- **Source:** notebook-internal · verified by reading cells 70–72

**D5-06 · 🔴 PCA component sign is arbitrary, so the "background" threshold flips per image** — `colab/class_5_Dino_Detr_and_Swin.ipynb` cells 20, 21
- **Now:** cell 20 — "We are using PCA to extract a *single principal component* (PC1) in order to
  separate the main feature from the background."; cell 21 —
  ```python
  threshold = 0.4
  background = pca_features > threshold
  ```
- **Current:** `pca_segment` refits `PCA` **and** `MinMaxScaler` independently for every image. The
  sign of a principal component is not determined by the data — sklearn's SVD may return either
  `+PC1` or `−PC1` for the same structure. After MinMax normalisation to [0,1], whether the
  foreground lands above or below 0.4 is therefore arbitrary and can differ between the ten images
  in the loop.
- **Why:** The cell is teaching "DINOv2 patch features separate foreground from background", which
  is true and is the whole point of the section. But the demo will show the mask inverted for some
  images, and a student has no way to tell whether that is a property of DINOv2 or of PCA. The
  notebook plots both `background` and `~background` (cell 21) without ever saying why.
- **Fix:** Either orient PC1 deterministically before thresholding, e.g. assume the border patches
  are background and flip if needed:
  ```python
  border = np.concatenate([pca_features[0, :], pca_features[-1, :],
                           pca_features[:, 0], pca_features[:, -1]])
  if border.mean() < 0.5:          # border should be the "background" side
      pca_features = 1.0 - pca_features
  ```
  or keep the current code and add one sentence: "the sign of a principal component is arbitrary,
  so which side of the threshold is foreground can flip per image — that is why we plot both the
  mask and its complement."
- **Source:** notebook-internal · verified by reading cells 17, 20, 21

## 🟠 Outdated

**D5-07 · 🟠 DINOv2 does have register variants** — `colab/class_5_Dino_Detr_and_Swin.ipynb` cell 34
- **Now:** "**[Register tokens](https://arxiv.org/pdf/2309.16588)** — DINOv2 had none. DINOv3 adds 4"
- **Current:** True of the *April 2023 release*, but the registers paper (2309.16588) is by the
  same Meta team and they subsequently shipped register variants of DINOv2 itself: torch hub
  `dinov2_vits14_reg` / `dinov2_vitb14_reg` / `dinov2_vitl14_reg` / `dinov2_vitg14_reg`, and on the
  Hub as `facebook/dinov2-with-registers-{small,base,large,giant}` — all with 4 registers. The
  `transformers` library carries a dedicated `Dinov2WithRegisters` model class.
- **Why:** The notebook frames registers as *the* DINOv2→DINOv3 delta ("biggest quality boost for
  dense tasks" in the cell 39 summary). A student who wants registers is told to move to DINOv3
  when they could just swap in `dinov2_vitl14_reg` and keep everything else. It also makes the
  cell 34 hint about `features[:, 5:, :]` look DINOv3-specific when it applies to any 4-register
  model.
- **Fix:** "DINOv2 shipped without them; registers were added later in a follow-up paper and
  backported as the `*_reg` DINOv2 checkpoints. DINOv3 builds them in from the start — 4 of them,
  which is why patch extraction uses `features[:, 5:, :]`."
- **Source:** https://huggingface.co/facebook/dinov2-with-registers-large · https://huggingface.co/docs/transformers/model_doc/dinov2_with_registers · verified

**D5-08 · 🟠 Swin listed as "State-of-the-art performance"** — `colab/class_5_Dino_Detr_and_Swin.ipynb` cell 59
- **Now:** advantages table row — "| Strong Accuracy | State-of-the-art performance |"
- **Current:** Swin is a 2021 architecture (correctly dated in cell 48). It is a strong, widely
  deployed backbone, but the notebook's own cell 47 shows DINOv3 ViT-H+ and ConvNeXt-L as the
  current-generation backbones, and cell 63 already notes ConvNeXt is more efficient. An unqualified
  "state of the art" in an advantages table is a claim with no expiry date written on it.
- **Why:** Swin absolutely belongs in this course — it is the canonical hierarchical ViT and the
  bridge between CNNs and transformers. The problem is only the framing: a student picking a
  backbone for a project should not read this table as "pick Swin, it's the best".
- **Fix:** Change the cell to "| Strong Accuracy | Was state of the art on ImageNet/COCO/ADE20K at
  publication (2021); still a competitive general-purpose backbone |". Optionally add a closing
  line to §6: "Swin is taught here as the architecture that made hierarchical attention practical —
  for a new project today the DINOv3 backbones from §5 are usually the stronger starting point."
- **Source:** notebook-internal cross-reference (cells 47, 48, 63) · **unverified** — no current
  ImageNet/COCO leaderboard fetched this run

**D5-09 · 🟠 DETR presented with no note that it has been superseded in practice** — `colab/class_5_Dino_Detr_and_Swin.ipynb` cells 66, 76, 77
- **Now:** §7 and §8.3 present DETR as the transformer detector, with no mention of what came after.
- **Current:** DETR (2020) is the foundational set-prediction detector, but its known weaknesses —
  ~500-epoch convergence, weak small-object performance — were the explicit motivation for a line
  of successors (Deformable DETR, DAB/DN-DETR, DINO-DETR, RT-DETR, D-FINE, DEIM). The
  `facebook/detr-resnet-50-panoptic` checkpoint used here is alive and fine for teaching.
- **Why:** Same shape as D5-08 — DETR is correct as curriculum, wrong as a recommendation. A
  student who builds a detector on plain DETR because this class showed it will get slow training
  and worse results than the alternatives.
- **Fix:** One paragraph at the end of §7: "DETR proved detection can be framed as set prediction,
  but it converges slowly (~500 epochs) and struggles with small objects. The follow-up line —
  Deformable DETR, DN/DINO-DETR, and the real-time RT-DETR / D-FINE family — fixes both while
  keeping the NMS-free set-prediction design. We teach DETR because everything after it is a
  variation on this architecture."
- **Source:** https://arxiv.org/pdf/2005.12872 (DETR) · **unverified** — the current best-in-class
  DETR descendant as of Aug 2026 was not confirmed this run

**D5-10 · 🟠 The install cell does not cover the notebook's dependencies** — `colab/class_5_Dino_Detr_and_Swin.ipynb` cell 2
- **Now:** `!pip install torch torchvision opencv-python`
- **Current:** The notebook additionally imports `timm` (cells 41, 45), `transformers` (70, 72),
  `dotenv` (70), `sklearn` (17, 23, 45), `tqdm` (6) and `matplotlib`. Of these, `timm` and
  `python-dotenv` are the ones not reliably present on a stock Colab runtime. Meanwhile `torch`
  and `torchvision` are already installed in Colab, so listing them unpinned is at best a no-op.
- **Why:** §5.6/§5.7 — the DINOv3 hands-on, which is the newest and most valuable material in the
  class — is the part that dies on `ModuleNotFoundError` if `timm` is absent. Students hit it 40
  cells into the notebook, after two long dataset downloads.
- **Fix:** Replace cell 2 with a pinned, complete install and drop the torch reinstall:
  ```python
  !pip install -q "timm>=1.0.19" "transformers>=4.44" python-dotenv opencv-python scikit-learn tqdm
  ```
  (`timm>=1.0.19` is the floor for the `dinov3` weights used in cells 41 and 45.)
- **Source:** notebook-internal · **unverified** — Colab's current default preinstall list was not
  fetched this run, so which of these are already present is an assumption

## 🟡 Gaps — content used but not explained, or missing

- **D5-11** `class_5_Dino_Detr_and_Swin.ipynb` cells 66, 76, 77 — **DETR's core mechanism is never
  explained.** The notebook says "set prediction task with transformers" (cell 76) and shows
  "N object queries" in the architecture list (cell 67), but object queries, bipartite (Hungarian)
  matching, the set-prediction loss and the no-object class ∅ are never defined. Cell 77 then writes
  `probas = out['pred_logits'].softmax(-1)[0, :, :-1]` — that `[:-1]` *is* the no-object class being
  dropped, and it appears with no comment. This is the single largest gap in the notebook: §7–8.3
  teach the API, not the idea.
  **Fix:** Add a §7.1 before the architecture overview covering (a) 100 learned object queries as
  "slots", (b) Hungarian matching to assign each ground-truth box to exactly one query, (c) why
  this removes NMS, (d) the ∅ class. There is already a Hungarian-matching example elsewhere in
  the repo (commit `69c6bf4`) worth linking or inlining.

- **D5-12** `class_5_Dino_Detr_and_Swin.ipynb` cells 0, 10, 20, 22 — **"DINO" means two different
  things and the notebook never says so.** The class teaches self-supervised DINO/DINOv2/DINOv3
  *and* transformer detection in the same session. In detection literature "DINO" is a different
  model entirely (DETR with Improved deNoising anchOr boxes, arXiv 2203.03605). A student who
  searches "DINO detection" after this class lands on the wrong paper.
  **Fix:** One sentence in §2: "Careful with the name — in the detection literature 'DINO' also
  refers to an unrelated DETR variant (2203.03605). Here DINO always means the self-supervised
  method."

- **D5-13** `class_5_Dino_Detr_and_Swin.ipynb` cell 76 — **§8.3 promises metrics it never computes.**
  The intro says the section "Tracks simple detection metrics (detections per image and confidence
  distribution)", but cell 77 runs on a single image and prints only `Kept detections: N`. No
  per-image count, no confidence distribution. mAP is deferred to "use `pycocotools`" with no
  example.
  **Fix:** Either loop over `coco_subset` and plot a histogram of `scores_kept` (matching what the
  text promises), or edit the intro to describe what the cell actually does.

- **D5-14** `class_5_Dino_Detr_and_Swin.ipynb` cells 8, 10 — **Why self-distillation doesn't
  collapse is never explained.** The notebook correctly describes the EMA teacher and the two
  losses, but the obvious student question — "if the student just copies the teacher and the
  teacher is a copy of the student, why doesn't everything collapse to a constant?" — is left open.
  Centering and sharpening of the teacher output are the answer and are absent.
  **Fix:** Add to §2.1: teacher output centering (subtract a running mean over the batch) prevents
  one-dimension dominance, sharpening (low teacher temperature) prevents uniform collapse, and the
  two are balanced against each other.

- **D5-15** `class_5_Dino_Detr_and_Swin.ipynb` cells 66, 71 — **Panoptic Quality (PQ) is never
  mentioned.** §7 is titled "DETR for Panoptic Segmentation" and cell 71 prints per-segment scores,
  but the metric that defines the task — PQ = SQ × RQ — does not appear. Students see thresholds
  (`threshold=0.85`, `mask_threshold=0.5`) with no way to reason about what they trade off.
  **Fix:** Add a short block defining PQ and explaining what raising/lowering the two thresholds
  does to it.

- **D5-16** `class_5_Dino_Detr_and_Swin.ipynb` cell 47 vs cells 41, 45 — **The comparison table
  describes models the code never runs.** §5.8 compares `dinov3-vith16plus` (840M) against
  `dinov3-convnext-large`, but the hands-on cells load `convnext_large.dinov3_lvd1689m` (cell 41)
  and `vit_small_patch16_dinov3.lvd1689m` — **ViT-S/16, 21.6M** (cell 45). The student is told
  ViT-H+ is "your best feature extractor for accuracy-critical tasks" and then runs a model 40×
  smaller without being told they are different.
  **Fix:** Add a ViT-S/16 row to the table, or a note under it: "the hands-on below uses ViT-S/16
  (21.6M) so it fits in a free Colab session — same architecture family, same 4 registers, same
  RoPE, much smaller."

- **D5-17** `class_5_Dino_Detr_and_Swin.ipynb` cells 9, 10 and cells 39, 40 — **Duplicate section
  numbers.** Two cells are numbered "2.3" (Dense Prediction Transformer, then iBOT) and two are
  numbered "5.6" (Positional encoding: RoPE, then DinoV3 - ConvNext feature maps). The table of
  contents in cell 1 does not reach this depth so nothing catches it.
  **Fix:** Renumber iBOT to 2.4 (and shift 2.4 Dataset and Training → 2.5), and ConvNeXt feature
  maps to 5.7 (shifting the subsequent 5.7 → 5.8 and 5.8 → 5.9).

- **D5-18** `class_5_Dino_Detr_and_Swin.ipynb` cell 74 — **Two labels contradict the code.** The
  prints say "top-5 predictions" (four times) but the code is `torch.topk(..., k=3)`; and the
  subplot title says `"512x512 Image"` while `tfms_518` resizes to 518×518. 518 is not an arbitrary
  number — it is 37×14, the DINOv2 patch grid — so mislabelling it as 512 loses the lesson.
  **Fix:** `k=5` or retitle to "top-3"; and `"518x518 Image (37×14 patches)"`.

- **D5-19** `class_5_Dino_Detr_and_Swin.ipynb` cell 43 — **All four feature-map subplots get the
  same title.** `axs[j].set_title(f"Feature Map {i+1}")` uses `i` (the image index, constant across
  the inner loop) where it should use `j` (the stage index). Every stage in a row reads
  "Feature Map 1". This directly defeats cell 42, whose table is about telling stages apart.
  **Fix:** `axs[j].set_title(f"Stage {j+1}")`.

- **D5-20** `class_5_Dino_Detr_and_Swin.ipynb` cell 41 — **The "Feature maps" print shows one
  tensor, not four stages.** After the loop, `for o in output[0]:` iterates over the *first stage
  tensor* along its batch dimension, printing a single `[192, 56, 56]`. The cell 42 table then
  presents four stage shapes the student never actually saw printed.
  **Fix:** `for o in output:` — `output` is already the list of four stage tensors.

- **D5-21** `class_5_Dino_Detr_and_Swin.ipynb` cell 45 vs cell 41 — **Inconsistent preprocessing for
  the same model family.** Cell 41 correctly does `timm.data.resolve_model_data_config(model)` →
  `create_transform(...)`. Cell 45 hardcodes `Resize((224, 224))` and ImageNet mean/std, while the
  checkpoint's timm config is **256×256**. The DINOv3 ViTs use RoPE so this runs, but it drops
  resolution below the trained config and teaches the hardcoding habit right after demonstrating
  the correct one.
  **Fix:** Use `resolve_model_data_config` in cell 45 too, and if you keep 224 for speed, say why:
  "RoPE lets us feed 224 to a model configured for 256 — that flexibility is one of §5.6's points."
  Note the 14×14 grid arithmetic in cell 46 assumes 224 and would need updating for 256.
- **Source:** https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m · verified (256×256 config)

- **D5-22** `class_5_Dino_Detr_and_Swin.ipynb` cell 41 — **`transforms` is silently shadowed.**
  `transforms = timm.data.create_transform(**data_config, is_training=False)` overwrites the
  `torchvision.transforms` module imported in cell 3. Cell 45 papers over it with a re-import, but
  any student who runs cell 41 and then jumps back to cell 13, 65 or 74 gets a confusing
  `TypeError`. In a teaching notebook, out-of-order execution is the normal case.
  **Fix:** Rename to `timm_transforms` in cell 41.

- **D5-23** `class_5_Dino_Detr_and_Swin.ipynb` cell 62 — **SegFormer does not use a Swin backbone.**
  The applications table lists "Semantic Segmentation | UPerNet, SegFormer-like systems". UPerNet is
  correct — it is the standard Swin segmentation head. SegFormer uses its own hierarchical MiT
  encoder, designed as an *alternative* to Swin, not a consumer of it.
  **Fix:** "UPerNet, Mask2Former".

- **D5-24** `class_5_Dino_Detr_and_Swin.ipynb` cell 52 — **"partition into 7 × 7 windows" is
  ambiguous and reads as wrong.** M=7 is the window *size* in patches, not the number of windows. A
  56×56 feature map splits into 8×8 = 64 windows of 7×7 patches each.
  **Fix:** "partition into windows of 7 × 7 patches → 8 × 8 = 64 windows". This also makes the
  O(M²N) complexity formula immediately below legible.

- **D5-25** `class_5_Dino_Detr_and_Swin.ipynb` cell 7 — **`DEMO_MAX_SAMPLES = 256` is defined and
  never used.** The actual subsetting happens via `stop_at` defaults buried in the cell 26 function
  signatures (`stop_at=1000` train, `stop_at=300` eval) and `min(64, len(coco_val))` in cell 77.
  **Fix:** Either wire `DEMO_MAX_SAMPLES` through as the `stop_at` argument, or delete it and hoist
  the real limits into cell 7 where students look for them.

## 🔵 Improvements

- **D5-26** `class_5_Dino_Detr_and_Swin.ipynb` cell 78 — **The SwiGLU attribution reference is
  off-topic.** "Dual Path Attribution: Efficient Attribution for SwiGLU-Transformers through
  Layer-Wise Target Propagation" (arXiv 2603.19742, Mar 2026) is a real, current paper — but it is
  LLM interpretability, and SwiGLU gets exactly one line in this notebook (cell 34). The list
  already cites the actual SwiGLU source (2002.05202).
  **Fix:** Drop it, or move it to an "further reading, tangential" subsection so it does not read as
  required background.
- **Source:** https://arxiv.org/abs/2603.19742 · verified (paper exists, v1 Mar 2026, v2 Jul 2026)

- **D5-27** `class_5_Dino_Detr_and_Swin.ipynb` cell 78 — **KoLeo is listed twice.** arXiv 1806.03198
  appears as both "KoLEO - SPREADING VECTORS FOR SIMILARITY SEARCH" and "Koleo - SPREADING VECTORS
  FOR SIMILARITY SEARCH".
  **Fix:** Delete one.

- **D5-28** `class_5_Dino_Detr_and_Swin.ipynb` cell 78 — **Four papers the notebook leans on are
  missing from the references.** Original DINO (2104.14294) — the notebook says "This is the
  original *DINO self-distillation*" in cell 10 and never cites it; iBOT (2111.07832) — §2.3 is
  entirely about it; ConvNeXt (2201.03545) — cells 34, 47, 63; Panoptic Segmentation (1801.00868) —
  the source of the thing/stuff distinction in cell 66.
  **Fix:** Add all four. The registers paper (2309.16588) is currently only linked inline in cell 34
  and belongs in the list too.

- **D5-29** `class_5_Dino_Detr_and_Swin.ipynb` cells 5, 6 — **The dataset downloads have no size or
  time warning.** Cell 5 pulls the Kaggle ImageNet-256 bundle and cell 6 pulls COCO val2017 (~1 GB)
  plus annotations (~241 MB). The COCO cell does document sizes in comments, but nothing tells a
  student up front that setup is a multi-GB, multi-minute step, or that Colab's disk will hold it.
  **Fix:** A markdown cell before §1: total download size, expected time, and a note that the
  `DOWNLOAD_TRAIN` flag must stay `False` on Colab (train2017 is ~18 GB).

## Version landscape

| Family | Used in the course | Current as of this run | Where | Action |
| --- | --- | --- | --- | --- |
| DINO (self-supervised) | DINOv2 (`vitl14`, `vits14`, `vits14_lc`) + DINOv3 (`vit_small_patch16`, `convnext_large`) | DINOv3 (Aug 2025) — course is current | cells 15, 28, 41, 45, 74 | none — this is the up-to-date part of the notebook |
| DINOv2 registers | "DINOv2 had none" | `dinov2_*_reg` / `facebook/dinov2-with-registers-*` exist | cell 34 | update — see D5-07 |
| DETR | `facebook/detr-resnet-50-panoptic`, hub `detr_resnet50` | Checkpoints live; architecture superseded in practice by Deformable/DN/RT-DETR, D-FINE, DEIM | cells 66–77 | mention — see D5-09 |
| Swin | `swin_t` / `Swin_T_Weights.IMAGENET1K_V1`, SwinV2 named | API current; SwinV2 (2022) is the latest in the family | cells 3, 61, 65, 74 | reframe "state of the art" — see D5-08 |
| ConvNeXt | `convnext_large.dinov3_lvd1689m` (196.2M) | timm id correct, checkpoint live | cells 41, 47 | none |
| `timm` | not installed, not pinned | DINOv3 weights need `timm>=1.0.19` | cell 2 (absent) | add — see D5-10 |
| `transformers` DETR API | `DetrImageProcessor`, `post_process_panoptic_segmentation` | Both current; the deprecated names are `DetrFeatureExtractor` / `post_process_panoptic` | cells 70, 71 | none — notebook uses the right ones |
| torchvision `pretrained=` | flagged by the inventory in cells 41, 45, 77 | False positives — 41/45 are `timm.create_model` and 77 is `torch.hub.load`, where `pretrained=True` is correct | — | none |
| COCO 2017 | `images.cocodataset.org` over HTTP | All four URLs return 200 | cell 6 | none |
| Kaggle ImageNet-256 | `dimensi0n/imagenet-256` | Live; the API endpoint redirects to signed GCS storage and downloads without credentials | cell 5 | none |

## Per-notebook notes

### `colab/class_5_Dino_Detr_and_Swin.ipynb`

Covers DINOv2 (theory, PCA/k-means feature segmentation, linear probing), DINOv3 (architecture
deltas, ConvNeXt and ViT hands-on), Swin Transformer (theory + fine-tuning) and DETR (panoptic
segmentation + detection inference). Written entirely in English.

State: the DINOv3 theory (§5.1–5.8, cells 32–47) is the best material here and is factually
accurate on the architecture side — register tokens, KoLeo, Gram anchoring and RoPE are all
correctly described, and the parameter table matches Meta's published figures. The DINOv2 theory
carries two hard factual errors (**D5-01**, **D5-02**). The Swin section is solid pedagogically but
needs the SOTA claim softened (**D5-08**) and two small precision fixes (**D5-23**, **D5-24**).

The DETR section (§7, §8.1, §8.3) is the weakest: three markdown cells of overview, then API calls,
with the defining ideas of the architecture absent (**D5-11**). It also carries the panoptic
`target_sizes` bug (**D5-05**).

Code health: five of the twenty-three code cells have defects that produce wrong or misleading
output without raising — cells 13 (**D5-04**), 21 (**D5-06**), 41 (**D5-20**), 43 (**D5-19**), 71
(**D5-05**). None of them error, which is what makes them worth fixing.

Ordering: the notebook runs top-to-bottom cleanly. Cell 15 loads `dinov2_vitl14`, cells 19–23 use
it, and cell 28 then rebinds `dinov2_backbone` to `dinov2_vits14` for the linear probe — correct in
sequence, but worth a comment, since re-running cell 21 after cell 28 silently switches the model
under the PCA demo. Same class of fragility as **D5-22**.

## Not verified

- **D5-08** — that Swin is no longer state of the art. No current ImageNet/COCO leaderboard was
  fetched this run. The recommendation stands on the notebook's own internal evidence (it teaches
  DINOv3 backbones two sections earlier) rather than on a benchmark table.
- **D5-09** — which DETR descendant is the current best choice as of Aug 2026. The existence of the
  successor line is well established; their present ranking was not checked.
- **D5-10** — Colab's default preinstalled package set. `timm` and `python-dotenv` are assumed
  absent; `transformers`, `sklearn`, `tqdm` and `matplotlib` are assumed present. Adding all of them
  to the install cell is safe either way, which is why the fix does not depend on resolving this.
- **D5-06** — the PCA sign flip is derived from how `sklearn.decomposition.PCA` and per-image
  `MinMaxScaler` refitting work, not from running the notebook. Confirming which of the ten images
  invert requires a GPU session.
- **D5-21** — that cell 45 runs at all at 224 (rather than the configured 256) rests on timm's
  DINOv3 ViTs being built on the RoPE-based EVA model class. Not executed this run.

Everything else was either confirmed against a fetched source (URL on the finding) or is verifiable
by reading the notebook.

## Method

Read: `colab/class_5_Dino_Detr_and_Swin.ipynb` in full (79 cells, digest with outputs stripped),
plus the mechanical inventory. No other notebook was reviewed — the skill was invoked with this
notebook as its argument. No findings were collapsed as duplicates, since only one notebook was in
scope; several findings here (the `pretrained=True` false positives, the unpinned install pattern)
plausibly recur in other classes and would be worth checking on a full-folder run.

Verified this run by fetching: the DINOv2 and DINOv3 papers, the DINOv2 model card, the
`facebook/dinov2-with-registers-*` and `facebook/dinov3-vith16plus-pretrain-lvd1689m` model cards,
the `timm/vit_small_patch16_dinov3.lvd1689m` and `timm/convnext_large.dinov3_lvd1689m` model cards,
`facebook/detr-resnet-50-panoptic`, the `facebookresearch/detr` and `facebookresearch/dinov2` repos,
the `dinov2_vits14_linear_head.pth` checkpoint, all four COCO URLs, the Kaggle ImageNet-256 endpoint
(including following its redirect to signed storage), the DINOv2 demo page, and arXiv 2603.19742.
**Every external non-image link in the notebook resolves — there are no rotted references.**

Image URLs are out of scope here; they were checked separately and all 13 return 200 — see
`status/image-validation.md`. Note that those URLs point at `main/content/`, and the current working
tree has those files staged as deleted; pushing that deletion would break all 13.

---

_Generated by the `validate_content` skill. Findings are proposals — no notebook was
modified._
