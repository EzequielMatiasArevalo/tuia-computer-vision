# tuia-computer-vision

Course material for **Computer Vision** at **FCEIA — Tecnicatura Universitaria en Inteligencia
Artificial (TUIA)**, Universidad Nacional de Rosario.

The course is taught entirely out of Jupyter notebooks designed to run in **Google Colab**: each
class pairs the theory of a family of models with working PyTorch code that students execute,
modify and break during the class. Nothing needs to be installed locally — open a notebook in
Colab, run it top to bottom on a fresh runtime, and it works.

## Goal

Take a student who knows Python and the basics of neural networks, and walk them from a hand-written
2D convolution all the way to the models that are actually deployed today — detection and
segmentation transformers, open-vocabulary and promptable models, face recognition pipelines, and
latent diffusion.


## Contents

For the full heading-by-heading outline of every notebook, see [INDEX.md](INDEX.md).

| # | Notebook | Topic | Open |
| --- | --- | --- | --- |
| 1 | [Class_1_Convolutions_Datasets_CNN_and_Base_Architecture](colab/Class_1_Convolutions_Datasets_CNN_and_Base_Architecture.ipynb) | Datasets, annotations, convolutions, CNNs from scratch, ResNet50, Faster R-CNN | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/Class_1_Convolutions_Datasets_CNN_and_Base_Architecture.ipynb) |
| 2 | [Class_2_Modern_CNN](colab/Class_2_Modern_CNN.ipynb) | Normalization, ConvNeXt, EfficientNetV2, NFNet, detection metrics, YOLO, Mask R-CNN, transfer learning | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/Class_2_Modern_CNN.ipynb) |
| 2b | [Class_2_Hands_on_lab](colab/Class_2_Hands_on_lab.ipynb) | Lab: train a timm classifier on EuroSAT, then break it with domain shift | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/Class_2_Hands_on_lab.ipynb) |
| 3 | [class_3_Key_Concepts_for_Deep_Learning_CV](colab/class_3_Key_Concepts_for_Deep_Learning_CV.ipynb) | Anchors, RoI Align, NMS, evaluation metrics, classification and box losses | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/class_3_Key_Concepts_for_Deep_Learning_CV.ipynb) |
| 4 | [class_4_ViT_and_FD](colab/class_4_ViT_and_FD.ipynb) | Vector databases, self-attention, ViT, face detection/recognition, ArcFace | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/class_4_ViT_and_FD.ipynb) |
| 5 | [class_5_Dino_Detr_and_Swin](colab/class_5_Dino_Detr_and_Swin.ipynb) | DINOv2/DINOv3, Swin, DETR and set prediction, panoptic segmentation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/class_5_Dino_Detr_and_Swin.ipynb) |
| 6 | [class_6_Clip_and_SAM](colab/class_6_Clip_and_SAM.ipynb) | SAM 3, CLIP, Grounding DINO — promptable and open-vocabulary vision | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/class_6_Clip_and_SAM.ipynb) |
| 7 | [class_7_CPU_vs_GPU_YOLOV11_vs_SAM3](colab/class_7_CPU_vs_GPU_YOLOV11_vs_SAM3.ipynb) | Tracking (ByteTrack, BoT-SORT) and a CPU/GPU latency benchmark: YOLO11-seg vs SAM 3 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/class_7_CPU_vs_GPU_YOLOV11_vs_SAM3.ipynb) |
| 8 | [class8_stable_diffusion_course](colab/class8_stable_diffusion_course.ipynb) | Diffusion theory, latent diffusion, CLIP conditioning, samplers, LoRA, ControlNet | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/class8_stable_diffusion_course.ipynb) |

---

## Detailed index

### Class 1 — Datasets and CNNs

[`colab/Class_1_Convolutions_Datasets_CNN_and_Base_Architecture.ipynb`](colab/Class_1_Convolutions_Datasets_CNN_and_Base_Architecture.ipynb)

The foundation class: what a vision dataset is, what a convolution does, and how the two combine
into a detector you build yourself before ever loading a pretrained one.

**1. Datasets** — the role of datasets in CV and the types that exist; challenges and
considerations (bias, imbalance, label noise). Annotation and labeling types, one by one:
bounding boxes · polygonal segmentation · semantic, instance and panoptic segmentation ·
3D cuboids · keypoints and landmarks · lines and splines. A worked example downloading COCO and
visualizing its annotations. Tour of the public datasets that matter:
[ImageNet](https://ieeexplore.ieee.org/document/5206848/) · [MS COCO](https://arxiv.org/pdf/1405.0312v3) ·
[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) · [KITTI](http://www.cvlibs.net/datasets/kitti/) ·
[nuScenes](https://www.nuscenes.org/).

**2. Convolutions** — loading and displaying images with matplotlib, color space conversions, how
convolutions actually work (kernels, stride, padding, receptive field), the distinction between
*convolution* and *cross-correlation* that every framework quietly glosses over, and a 2D
convolution implemented from scratch.

**3. Convolutional Neural Networks** — the three-part anatomy of a CNN: feature extraction
(backbone/encoder), feature aggregation (neck), task-specific prediction (head/decoder). Then
building one: activation functions, capturing intermediate activations with forward hooks, dataset
normalization (what `std` is doing and why it matters), running the network and **visualizing the
feature maps after every layer**. Section 3.6 assembles a complete detector — custom backbone,
custom neck, classification head, then anchors, box offsets and NMS feeding a detection head.

**4. ResNet50** — the degradation problem that motivated residual learning
([He et al., 2015](https://arxiv.org/abs/1512.03385)), skip connections, the bottleneck block, the
ResNet50 layer-by-layer breakdown, running the ImageNet-pretrained model, and a side-by-side of
**pretrained kernels vs. the randomly-initialized kernels** from section 3.

**5. Faster R-CNN on COCO** — loading the pretrained detector, converting COCO `x,y,w,h` boxes to
`x1,y1,x2,y2`, drawing predictions, and then answering *how good is this actually?* with IoU,
precision/recall and mAP — including how to read the numbers.

---

### Class 2 — Modern CNN Architectures: From Classic to SOTA

[`colab/Class_2_Modern_CNN.ipynb`](colab/Class_2_Modern_CNN.ipynb)

**1. Key concepts** — *internal covariate shift*, presented as the historical motivation for
BatchNorm alongside a warning that it is **not** the current explanation. Normalization layers
compared: Batch · Layer · Group Norm, with the grouped-channel illustration and a summary table.
Depthwise and pointwise convolutions built up step by step into depthwise-separable convolution,
with the parameter/FLOP trade-off made explicit. Activation functions and why the field moved off
ReLU (GELU, SiLU) — the design point that sets up ConvNeXt.

**2. Top-performing CNN architectures** — **ConvNeXt** and its block design ·
**ConvNeXt V2** with Global Response Normalization, and how GRN differs from the other
normalizations · a hands-on run of pretrained `convnext_xlarge` on ImageNet-1k including
denormalizing images for display · **EfficientNetV2** (what actually changed from V1) ·
**NFNet**, normalizer-free networks and the three things that replace BatchNorm ·
**ResNeXt**, Wide ResNets, the three axes of scaling and grouped convolutions.

**3. SOTA detection & segmentation** — detection metrics from first principles: IoU → precision
and recall → AP → mAP, including the thing everybody gets wrong about mAP, plus how to compute it
yourself with `torchmetrics`. **Faster R-CNN**: the Region Proposal Network, anchor boxes, the two
RPN heads, the training signal, NMS on proposals, and RoI Pooling — with the note that
`fasterrcnn_resnet50_fpn` really runs **RoIAlign**, plus a hands-on COCO-style detection run.
**YOLO (v5 → YOLO26)**: why "only looks once", what detection looked like before it, the
single-pass approach and the YOLOv8 architecture. **Mask R-CNN** and where it sits today.
**EfficientDet** and BiFPN vs. classic FPN.

**4. Key trends** — attention inside CNNs: Squeeze-and-Excitation (channel attention) and CBAM
(channel + spatial). Efficiency vs. performance through a FLOPs analysis. 1D-CNNs for temporal
signals.

**5. Transfer learning** — fine-tuning strategies, then actually fine-tuning ConvNeXt-Tiny on
CIFAR-10.

---

### Class 2 (lab) — Train an image classifier with TIMM / GeoAI

[`colab/Class_2_Hands_on_lab.ipynb`](colab/Class_2_Hands_on_lab.ipynb)

A full training lab that follows the [GeoAI *Train timm classifier*](https://opengeoai.org/examples/train_timm_classifier/)
walkthrough, using `geoai.timm_train` over PyTorch Lightning. The second half is the interesting
part: the model that scores well on its own test split falls apart on real imagery, and the notebook
diagnoses why.

- **Setup & data** — key ideas (model zoo, multi-channel inputs, Lightning training loop, transfer
  learning vs. fine-tuning, and *always read input size and mean/std off the checkpoint*). Exploring
  the timm model zoo, downloading **EuroSAT RGB**, building file lists and labels, checking class
  distribution, train/val/test splits, one example per class, and constructing
  `RemoteSensingDataset` objects **with the preprocessing the backbone expects**.
- **Training** — three runs to compare: ResNet-50 fine-tuned, EfficientNet-B3 fine-tuned, and a
  frozen backbone (pure feature extraction). Includes why `last.ckpt` is not the best checkpoint.
- **Evaluation** — loading the trained models, inference, visualizing predictions for all three runs
  on the same 20 validation images, quantitative evaluation on the test split, first conclusions.
- **Out of distribution** — the same models on high-resolution aerial imagery. Evaluating
  performance *without ground truth*; the diagnosis (**domain shift, not class imbalance**);
  testing the "is it the scale?" hypothesis; and how to fix it — match the resolution first, then
  data-centric improvements, label quality and sampling, and model/training changes. Closes with
  four ways to generate ground truth (manual, semi-automatic, external sources, weak supervision).

---

### Class 3 — Key concepts for deep learning CV

[`colab/class_3_Key_Concepts_for_Deep_Learning_CV.ipynb`](colab/class_3_Key_Concepts_for_Deep_Learning_CV.ipynb)

The mechanics class. Everything a detector does between "features" and "boxes on screen", plus the
losses that train it.

- **Anchor boxes & region proposals** — IoU and the assignment rules Faster R-CNN's RPN uses; what
  problem anchors solve; what an anchor box is and how it works; the encoding/decoding math; and
  anchor-free detectors.
- **The head** — a ResNet50 backbone with an FPN neck as the worked example, the problem the head
  has to solve, RoI Pooling, and **RoI Align** (Mask R-CNN, 2017) with the misalignment it fixes.
- **NMS** — non-maximum suppression, the two things the usual diagram leaves out, and NMS
  implemented from scratch on a synthetic example.
- **Evaluation metrics** — accuracy, recall, precision (with a worked example), F1, mAP@50, and
  COCO-style mAP@[50:95], ending in a one-page mental map of which metric answers which question.
- **Loss functions** — *classification*: cross-entropy (intuition, formula, why the `-log`) and
  **Focal Loss** — the imbalance problem it solves, its focusing mechanism, who actually uses it,
  and a direct comparison against cross-entropy. *Box regression*: what the model is really
  predicting, then L2/MSE → Smooth L1 (Huber) → IoU → GIoU → DIoU → CIoU, a full comparison table,
  what "scale-invariant" really means here, how these combine in a real detector, and a hands-on
  cell **watching IoU loss die** on non-overlapping boxes.

---

### Class 4 — Transformers for Computer Vision (ViT + face recognition)

[`colab/class_4_ViT_and_FD.ipynb`](colab/class_4_ViT_and_FD.ipynb)

The longest notebook in the course: attention from scratch, then a complete face-recognition
pipeline with a real vector database behind it.

- **1. Vector databases** — what they are, core mechanics, the typical pipeline, where they're used,
  and how the field evolved (production infrastructure, hybrid retrieval as default, indexing and
  compression, LLM-stack integration, multimodal embeddings, streaming updates, security and
  governance, hardware acceleration) plus current limitations. Uses **pgvector on Postgres** — see
  [`helpers/compose.yaml`](helpers/compose.yaml).
- **2. Transformer fundamentals** — self-attention intuition, multi-head attention, positional
  information, encoder/decoder, and a minimal scaled dot-product attention implementation.
- **3–4. ViT and the model landscape** — the Vision Transformer, its training considerations and
  limitations; then DETR and variants, DETR for panoptic segmentation, Swin, DINO/DINOv2/DINOv3
  self-supervised features, and hybrid CNN–Transformer designs.
- **5. Hands-on ViT** — ImageNet inference with pretrained `vit_base_patch16_224`, a fine-tuning
  sketch for a small dataset, and plotting helpers.
- **6. Face recognition, end to end** — opens with **6.0 bias, consent and the law**: look at your
  own training set first, error rates are not equal across demographics, some uses are prohibited
  outright in the EU, and locally **Ley 25.326** and the CABA ruling — and what all of that changes
  about the rest of the section. Then: downloading and caching celebrity photos; face alignment and
  detection with **MTCNN** and **RetinaFace/InsightFace** (boxes, landmarks, aligned vs. unaligned
  crops); building a post-processing training dataset and normalization pipelines; training a head
  (or loading a checkpoint); top-k retrieval on face crops; ViT as a feature extractor; registering
  identities and embeddings in Postgres; embedding analysis with an identification/verification API;
  **PCA and t-SNE visualization** of the embedding space with what good and bad embeddings look
  like; **6.14.7 choosing the threshold — FAR, FRR, ROC and TAR@FAR**, with an exercise applying one
  threshold to several populations; query flow (new photo → detect → identify); 1:1 verification;
  and baselines for ViT *without* fine-tuning and for InsightFace.
- **7. Face recognition losses** — softmax, triplet, and **ArcFace**: motivation, mathematical
  formulation (standard softmax → normalized softmax/CosFace → ArcFace), geometric intuition,
  step-by-step derivation, PyTorch implementation, comparison with related losses, decision-boundary
  comparison, a hyperparameter guide tied to the LFW/IJB-C benchmarks, training tips (including the
  easy-to-miss point that the class centers live on the *loss* module, not the backbone), BatchNorm
  before ArcFace, and inference without ArcFace.
- **8. Training MobileFaceNet with ArcFace** as a worked example.

---

### Class 5 — DINOv2, DINOv3, Swin and DETR

[`colab/class_5_Dino_Detr_and_Swin.ipynb`](colab/class_5_Dino_Detr_and_Swin.ipynb)

- **2–4. DINOv2** — what distillation is; the three mechanisms through which DINOv2 learns depth
  implicitly from self-supervision; Dense Prediction Transformer; **iBOT** and the two losses in
  DINOv2, and why that matters for segmentation; dataset and training. Hands-on: loading pretrained
  DINOv2, helpers to preprocess/draw/run PCA over the features, ImageNet-256 images, then
  **segmenting with DINO + PCA** and **DINO + K-Means**. Section 4 fine-tunes DINOv2 with a custom
  head and evaluates it.
- **5. DINOv3** — architecture additions over v2: **register tokens**, the training objective,
  **KoLeo regularization**, **Gram anchoring**, **RoPE** positional encoding, how they work
  together, and the teacher model. Then feature maps from the ConvNeXt backbone
  (`convnext_large.dinov3_lvd1689m`), the ViT backbone (`vit_small_patch16_dinov3.lvd1689m`) step by
  step, and a backbone comparison — training, practical differences, and when to choose each.
- **6. Swin Transformer** — why it was needed; the main ideas; the architecture end to end: patch
  partition → window-based self-attention (W-MSA) with the complexity comparison against global
  ViT attention → the problem with fixed windows → the shifted-window mechanism (SW-MSA). Then the
  Swin block, hierarchical representation, patch merging, Swin vs. ViT vs. CNNs, advantages,
  disadvantages, variants, applications, Swin vs. ConvNeXt and where Swin sits today, plus a
  fine-tuning example on ImageNet classification.
- **7. DETR** — detection as **set prediction**: the problem with earlier detectors, the 100 object
  queries, the ∅ "no object" class, **bipartite matching via the Hungarian algorithm**, the set
  prediction loss, and what it costs. Architecture overview; how panoptic segmentation is scored
  with **Panoptic Quality (PQ)** and what the notebook's two thresholds do to it; and what came
  after DETR.
- **8. Hands-on** — `facebook/detr-resnet-50-panoptic` for panoptic segmentation, Swin and DINOv2
  for classification, and a DETR COCO object-detection example. Closes with a reference list split
  into core papers, supporting techniques, and where DETR went next.

---

### Class 6 — CLIP, SAM 3 and Grounding DINO

[`colab/class_6_Clip_and_SAM.ipynb`](colab/class_6_Clip_and_SAM.ipynb)

Promptable and open-vocabulary vision — models you steer with text instead of retraining.

- **0. Setup** — three installation paths (HuggingFace `transformers`, Ultralytics, or Meta's
  official repo) and Hugging Face authentication, since `facebook/sam3` is a **gated** repo you must
  request access to.
- **1. SAM 3** — introduction and the supported prompt types: Promptable Visual Segmentation (PVS),
  Promptable Concept Segmentation (PCS), exemplar-based, open-vocabulary, zero-shot generalization,
  and multi-instance retrieval with consistent IDs. Then multi-instance segmentation, pixel-level
  precision and video object tracking. **Architecture** in depth: the Perception Encoder
  (large-scale contrastive pretraining, how PE differs from CLIP, why region-level embeddings
  matter), the shared vision encoder, the decoupled detector/tracker split, the detector (fusion
  encoder, concept-aware visual features, DETR decoder + object queries, what replaces sliding
  windows and anchor boxes, the presence token, the mask head), and the tracker (temporal pipeline,
  detect-then-propagate, Masklet Detection Score, periodic re-prompting, matching detections to
  masklets) — plus why the encoder is shared and where SAM 3's limitations are. **Hands-on** in
  three parts: text-prompted concept segmentation, open-vocabulary exploration, and video tracking
  with mp4 export.
- **2. CLIP** — core idea, high-level architecture, image encoder, text encoder, the shared
  embedding space, contrastive learning and the contrastive matrix, zero-shot classification, key
  components, why it was revolutionary, and its common use cases (zero-shot classification, image
  retrieval, semantic search, content moderation, visual recommendation). Then CLIP vs. traditional
  CNN classifiers, CLIP's limitations, CLIP vs. dense vision models, and **CLIP → SigLIP**.
  Hands-on: zero-shot classification and text-to-image retrieval with `openai/clip-vit-base-patch32`.
- **3. Grounding DINO** — open-vocabulary object detection: main goal, core capabilities, and the
  architecture piece by piece — vision encoder, text encoder, feature enhancer, cross-modal fusion,
  DETR-style decoder, the grounding mechanism, open-set detection, zero-shot generalization,
  training strategy and limitations. Hands-on with `IDEA-Research/grounding-dino-base`.
- **4–5. Exercises** with a worked answer section.

---

### Class 7 — CPU vs GPU: YOLO11-seg vs SAM 3

[`colab/class_7_CPU_vs_GPU_YOLOV11_vs_SAM3.ipynb`](colab/class_7_CPU_vs_GPU_YOLOV11_vs_SAM3.ipynb)

A systematic benchmark rather than a tutorial: 200 frames of video, three YOLO11-seg sizes
(`yolo11n`, `yolo11m`, `yolo11x`) on CPU and GPU, against one SAM 3 GPU configuration prompted with
the concepts `["person", "car"]` — because SAM 3 is *concept-conditioned*, not a class-agnostic
"segment everything" model.

- **Setup & protocol** — video download, configs, GPU memory helpers, the benchmark protocol, and
  **the scope trap**: what you accidentally include or exclude when you time an inference call.
- **1. Tracking algorithms** — shared foundations of **ByteTrack** and **BoT-SORT**; the
  **Hungarian algorithm** (the assignment problem, core idea, simplified steps, why it's useful);
  the **Kalman filter**; ByteTrack's "every detection counts"; BoT-SORT's motion + camera-motion
  compensation with optional appearance; a comparison table, a note on the other four trackers, and
  when to use each.
- **2. YOLO11** — the core benchmark function, the runs, and per-device plots: inference time (CPU
  and GPU) with analysis; detected masks and active tracks per frame — including why `yolo11m`
  detects *more* than `yolo11x`, how masks and tracks diverge, and the scene structure visible in
  the signal; and a CPU vs. GPU summary bar chart covering mean latency, P95 and the tail, and FPS.
- **3. SAM 3** — getting the gated weights, per-frame metric hooks, inference, GPU inference-time
  analysis (including why the shaded band is narrow) and instance counts per frame by class.
- **4–6. Summary, conclusions and exercises** — the headline latency numbers, P95 tails, FPS, and
  **why GPU doesn't rescue SAM 3**; mean masks per frame and the shared scene structure across both
  systems. One exercise asks you to re-measure without the NMS + mask-postprocessing stage that
  **YOLO26** removes.

> Every measured figure quoted in the prose comes from one reference run (Colab T4, 200 frames of
> `shinjuku.mp4`). Tables and plots regenerate when you re-run; the narrative does not. Where they
> disagree, trust the generated table.

---

### Class 8 — Stable Diffusion: theory & practice

[`colab/class8_stable_diffusion_course.ipynb`](colab/class8_stable_diffusion_course.ipynb)

GPU recommended (T4 or better). Uses `stable-diffusion-v1-5` and
`stable-diffusion-inpainting` via `diffusers`.

**Theory**

1. **From VAEs to diffusion** — why generative models, and a comparison of VAE / GAN / normalizing
   flow / diffusion / flow matching, with the warning that *two very different things are called
   "flow"*. VAE recap and the ELBO.
2. **Diffusion core idea** — the forward process and noise schedule, the reverse (denoising)
   process, and the training objective.
3. **Latent diffusion** — the scalability problem, the LDM solution (Rombach et al., 2022), why it
   works, and the key parameters in SD 1.5.
4. **The U-Net denoiser** — timestep embedding, architecture overview, and cross-attention for text
   conditioning.
5. **CLIP text conditioning** — what CLIP is, how SD uses it, classifier-free guidance, and the
   corollary that the negative prompt *is* ∅.
6. **Sampling** — DDPM (slow) vs. DDIM (fast, deterministic), a sampler comparison for
   ε-prediction models (SD 1.x / 2.x / XL), and where the field went next with flow matching /
   rectified flow.

**Practice** — text-to-image · image-to-image · inpainting · prompt engineering (anatomy of a good
prompt, why token weighting is *not* available in this pipeline, and how to actually do it with
`diffusers`) · visualizing the denoising process step by step · **LoRA** and fine-tuning concepts
(why fine-tune, low-rank adaptation, the available methods) · **ControlNet** for structural
conditioning, including why the zero-initialized convolutions are the same trick as LoRA's **B**
matrix.

**Wrap-up** — graded exercises (beginner / intermediate / advanced, the last one building a U-Net
skeleton and a CLIPScore helper), and **"what we switched off, and what we borrowed"**: what
`safety_checker=None` actually disables, where *"by Greg Rutkowski"* gets its power from, and
disclosing generated content with C2PA and watermarking.

---

## Repository layout

| Path | What lives there |
| --- | --- |
| [colab/](colab/) | The published course notebooks — the primary content of this repo. |
| [inprogress/](inprogress/) | Working copies of the same notebooks, ahead of `colab/`. Teach from `colab/`. |
| [media/pictures/](media/pictures/) | Illustrations, one folder per notebook, served to Colab over raw GitHub URLs. |
| [media/datasets/](media/datasets/) | Sample images used by the notebook demos. |
| [helpers/](helpers/) | [`timm_train.py`](helpers/timm_train.py) — a patched `geoai` training class used by the Class 2 lab. [`compose.yaml`](helpers/compose.yaml) — a pgvector/Postgres service for the Class 4 embeddings section. |
| [status/](status/) | Generated health reports. Do not hand-edit. |
| [.claude/skills/](.claude/skills/) | Repo maintenance skills (see below). |

## Running the notebooks

Open any notebook with its **Open in Colab** badge above, or from Colab via
*File → Open notebook → GitHub → `EzequielMatiasArevalo/tuia-computer-vision`*.

- Each notebook installs its own dependencies in the first cells and is written to run
  **top to bottom on a fresh runtime**.
- Set the runtime to **GPU** for classes 4–8. Class 7 deliberately runs on both CPU and GPU.
- Classes 6 and 7 use `facebook/sam3`, a **gated** Hugging Face repo — request access and
  authenticate before running those sections.
- Class 4 expects a Postgres instance with `pgvector`; `docker compose -f helpers/compose.yaml up`
  brings one up locally.

## A note on images

Notebooks run in Colab, which has no access to this checkout, so illustrations are referenced by
absolute `raw.githubusercontent.com` URL against this repo's `main` branch — not by relative path.
The practical consequence: **a new image is invisible in Colab until it is committed and pushed to
`main`**, and renaming a folder under `media/pictures/` breaks every notebook pointing at it. See
[CLAUDE.md](CLAUDE.md) for the full rules before editing images or notebooks.

## Maintenance

Two skills keep the material honest, both writing to [status/](status/):

| Skill | What it does |
| --- | --- |
| `validate_image` | Checks that every external image URL in `colab/` still returns HTTP 200 and writes [`status/image-validation.md`](status/image-validation.md). |
| `validate_content` | Audits the CV content — outdated model/dataset/library versions, stale "state of the art" claims, unexplained terms, deprecated APIs, missing topics — into `status/content/validate_content_<timestamp>.md`. Reports only; never edits a notebook. |

## License

[MIT](LICENSE) © 2026 EzequielMatiasArevalo. Third-party datasets, model weights and papers
referenced by the notebooks keep their own licenses.
