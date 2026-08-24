# Practica

Cuadernos practicos de la materia, en paralelo a la teoria de [`colab/`](../colab/).

La numeracion **sigue la de la teoria**: `class_N_` acompana a la clase teorica N. Una clase
teorica puede tener mas de una practica (la practica se demora dentro de una clase pero no rompe
el orden); cuando eso pasa se distinguen con sufijo de letra — `class_1a`, `class_1b`, `class_1c` —
igual que la teoria hace con `Class_2_Modern_CNN` y `Class_2_Hands_on_lab`.

La `class_0` es la unica sin contraparte teorica: es el repaso previo al arranque del curso.

Los cuadernos terminados en **`_TODO.ipynb`** son huecos: la clase teorica existe pero todavia no
tiene practica. Adentro esta el tema, el link a la teorica correspondiente y el contenido previsto.

## Cuadernos

| # | Practica | Teoria que acompana | Tema |
| --- | --- | --- | --- |
| 0 | [`class_0_inicio_y_repaso.ipynb`](class_0_inicio_y_repaso.ipynb) | — | Representacion digital de imagenes, espacios de color (RGB, CMYK, HSV/HSL), histogramas, convoluciones y filtros |
| 1a | [`class_1a_tecnicas_clasicas_vision.ipynb`](class_1a_tecnicas_clasicas_vision.ipynb) | [Class 1](../colab/Class_1_Convolutions_Datasets_CNN_and_Base_Architecture.ipynb) | Deteccion de bordes: Sobel paso a paso (Gx, Gy, magnitud) y Canny; comparacion directa entre ambos |
| 1b | [`class_1b_features_y_matching.ipynb`](class_1b_features_y_matching.ipynb) | [Class 1](../colab/Class_1_Convolutions_Datasets_CNN_and_Base_Architecture.ipynb) | Esquinas de Harris, piramides gaussiana y laplaciana, descriptores locales y matching |
| 1c | [`class_1c_redes_neuronales_cnn.ipynb`](class_1c_redes_neuronales_cnn.ipynb) | [Class 1](../colab/Class_1_Convolutions_Datasets_CNN_and_Base_Architecture.ipynb) | Backbone / neck / head, pooling y ReLU, feature maps por capa de ResNet50, clasificacion con ResNet50 pre-entrenada |
| 2a | [`class_2a_cnn_modernas.ipynb`](class_2a_cnn_modernas.ipynb) | [Class 2](../colab/Class_2_Modern_CNN.ipynb) | Transfer learning sobre STL-10, data augmentation, comparacion de modelos y analisis de errores |
| 2b | [`class_2b_features_similitud_transfer_learning.ipynb`](class_2b_features_similitud_transfer_learning.ipynb) | [Class 2](../colab/Class_2_Modern_CNN.ipynb) | Del clasificador al extractor de features, similitud del coseno, top-k, PCA, fine-tuning de ResNet18 en CIFAR-10 |
| 3 | [`class_3_key_concepts_TODO.ipynb`](class_3_key_concepts_TODO.ipynb) | [Class 3](../colab/class_3_Key_Concepts_for_Deep_Learning_CV.ipynb) | **TODO** — anchors, RoI Align, NMS, metricas y funciones de perdida |
| 4 | [`class_4_reconocimiento_facial.ipynb`](class_4_reconocimiento_facial.ipynb) | [Class 4](../colab/class_4_ViT_and_FD.ipynb) | LFW, deteccion con MTCNN, embeddings y t-SNE, verificacion 1:1, identificacion 1:N, pipeline completo |
| 5 | [`class_5_dino_detr_y_swin_TODO.ipynb`](class_5_dino_detr_y_swin_TODO.ipynb) | [Class 5](../colab/class_5_Dino_Detr_and_Swin.ipynb) | **TODO** — DINOv2/v3, DETR y Swin |
| 6 | [`class_6_clip_sam_y_grounding_dino_TODO.ipynb`](class_6_clip_sam_y_grounding_dino_TODO.ipynb) | [Class 6](../colab/class_6_Clip_and_SAM.ipynb) | **TODO** — CLIP, SAM 3 y Grounding DINO |
| 7 | [`class_7_deteccion_y_segmentacion.ipynb`](class_7_deteccion_y_segmentacion.ipynb) | [Class 7](../colab/class_7_CPU_vs_GPU_YOLOV11_vs_SAM3.ipynb) | Segmentacion de instancias con YOLO11-seg, benchmark CPU vs GPU, tracking (ByteTrack / BoT-SORT) y SAM 3 |
| 8 | [`class_8_stable_diffusion_TODO.ipynb`](class_8_stable_diffusion_TODO.ipynb) | [Class 8](../colab/class8_stable_diffusion_course.ipynb) | **TODO** — Stable Diffusion |

## Donde la practica y la teoria no coinciden

Vale la pena tenerlo a mano antes de dar una clase:

- **Clase 1** — la practica va *mas alla* de la teoria. Harris, piramides de imagen, descriptores
  locales y matching (`class_1b`) no aparecen en la teorica, que salta de convoluciones a CNNs.
- **Clase 2** — la teoria tiene ademas el lab [`Class_2_Hands_on_lab`](../colab/Class_2_Hands_on_lab.ipynb)
  (timm + EuroSAT, y el fallo por *domain shift*) que no tiene equivalente en la practica. Es un lab
  autocontenido, asi que puede darse tal cual desde `colab/`.
- **Clase 4** — la practica cubre **solo la seccion 6** de la teorica (deteccion, embeddings,
  verificacion e identificacion). Quedan sin practica: ViT como extractor de features, la base de
  datos vectorial con pgvector ([`helpers/compose.yaml`](../helpers/compose.yaml)) y ArcFace.
- **Clase 7** — la practica llega a lo mismo que la teorica pero con otro recorte: agrega SAM 3 con
  *hooks* de metricas por frame.

## Cruce con el cronograma

Correspondencia entre el cronograma de practicas del cuatrimestre y estos cuadernos:

| Semana | Fecha | Tema del cronograma | A cargo | Practica |
| --- | --- | --- | --- | --- |
| 1 | mie 26-ago | Inicio y repaso | Juli | `class_0` |
| 2 | mie 2-sept | Tecnicas clasicas de vision | Tino | `class_1a`, `class_1b` |
| 3 | mie 9-sept | Redes neuronales para vision | Juli | `class_1c` |
| 4 | mie 16-sept | Redes neuronales para vision | Tino | `class_2b` |
| 5 | mie 23-sept | CNNs modernas | Juli | `class_2a` |
| 6 | mie 30-sept | CNNs modernas | Tino | `class_4` |
| 7 | mie 7-oct | Deteccion de objetos y segmentacion | Juli | `class_7` |
| 8 | mie 14-oct | Vision language models (VLMs) — TP Intermedio | Tino | `class_6` (TODO) |
| 9 | mie 21-oct | Modelos generativos I (GANs y VAEs) — TP Intermedio | Juli | sin teoria ni practica |
| 10 | mie 28-oct | Modelos generativos II (Diffusion) — TP Intermedio | Tino | `class_8` (TODO) |
| 12 | mie 11-nov | Vision en video | Juli | parcialmente `class_7` |
| 14 | mie 25-nov | Vision 3D y NeRF — TP Final | Tino | sin teoria ni practica |

La fecha de la semana 1 esta fija: el curso arranca una semana mas tarde de lo previsto
originalmente, y el resto del cronograma corre en bloque detras de ella. Semanas 11 y 13: receso.

Entrega del TP Intermedio en la semana 10: **viernes 8, hasta las 23:59** (tal como figura en el
cronograma; conviene confirmar la fecha exacta antes de anunciarla).

El contenido de la semana 1 segun el cronograma —representacion digital, espacios de color,
histogramas, convoluciones y filtros— es exactamente lo que cubre `class_0`.

**Dos temas del cronograma no tienen ni teoria ni practica todavia**: modelos generativos I
(GANs y VAEs, semana 9) y vision 3D / NeRF (semana 14). No estan como `_TODO.ipynb` porque la
numeracion de esta carpeta sigue la de la teoria, que termina en la clase 8.

## Requisitos por cuaderno

| Cuaderno | Necesita |
| --- | --- |
| `class_0` | Solo CPU. Espera un `avatar.png` subido a mano en Colab (`/content/avatar.png`) |
| `class_1a`, `class_1b` | Solo CPU. Descargan sus imagenes en la primera celda de setup (ver abajo) |
| `class_1c` | GPU recomendada. Baja las imagenes de prueba y las clases de ImageNet por URL |
| `class_2a` | GPU. Descarga STL-10 |
| `class_2b` | GPU. Descarga CIFAR-10 y STL-10 |
| `class_4` | GPU. Descarga LFW; usa `facenet-pytorch` (MTCNN) |
| `class_7` | GPU para la mitad del benchmark, y CPU para la otra — corre a proposito en los dos. Baja `shinjuku.mp4` y los pesos de YOLO11. SAM 3 (`facebook/sam3`) es un repo *gated*: hay que pedir acceso |

## Imagenes de ejemplo

`class_1a` y `class_1b` usan `data/columnas.jpg` y `data/ladrillos.jpg`. Como en Colab no existe el
checkout del repo, las imagenes viven en
[`media/datasets/tecnicas_vision/`](../media/datasets/tecnicas_vision/) y una celda de setup al
principio de cada cuaderno las descarga a `./data/` por URL cruda contra la rama `main`.

> Mientras estos cuadernos vivan en una rama sin mergear, esa celda falla: la URL apunta a `main`.
> Se arregla sola al mergear.

## Origen

Estos cuadernos vienen del repositorio de practica
[`IA5.2_Computer_Vision`](https://github.com/julilc/IA5.2_Computer_Vision), donde estan organizados
por unidad. Aca se reorganizaron para seguir el orden de la teoria. Los outputs se conservan a
proposito, siguiendo la regla de [`CLAUDE.md`](../CLAUDE.md).
