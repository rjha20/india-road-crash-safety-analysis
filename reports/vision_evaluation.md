# India Road-Damage Detection: Evaluation and Handoff

**Status:** Reproducible computer-vision pilot completed 7 October 2026. The detector is an exploratory road-condition aid for human review. It is not a crash-risk model and is not connected to crash locations.

## Scope

The vision workstream detects four RDD2022 pavement-label classes in road images:

| Code | Class |
|---|---|
| D00 | Longitudinal crack |
| D10 | Transverse crack |
| D20 | Alligator crack |
| D40 | Pothole |

These images are not geotagged in the released India subset and are not joined to the project's media-reported crash events. A detection may provide a separate road-condition cue for an engineer; it cannot support a claim that a defect caused or predicted a particular crash.

## Source, license, and data handling

- Dataset: [RDD2022 Figshare record](https://doi.org/10.6084/m9.figshare.21431547.v1), archive `RDD2022_released_through_CRDDC2022.zip`.
- Challenge metadata and country counts: [CRDDC 2022 data page](https://crddc2022.sekilab.global/data/); image-use statement: [CRDDC rules](https://crddc2022.sekilab.global/rules/).
- Dataset collection-area description: [RDD2022 data article](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/gdj3.260).
- The full Figshare archive is 13,264,172,619 bytes; the Figshare API reports MD5 `b62bd51d2ffcfaa76c60f234f0cc2bb3`. That MD5 is upstream metadata and was **not** recomputed locally because the full archive was never downloaded.
- The official country-specific India ZIP link returned HTTP 403 in this environment. The Figshare archive supports HTTP byte ranges, so the pipeline indexed the central directory and extracted only the nested `RDD2022/India.zip` → `India/train/*.jpg` and `*.xml` entries. The range reader refuses a server response that is not HTTP 206, checks every output path against the extraction root, and does not save the full archive.
- Archive enumeration found **7,706 image/XML pairs** (15,412 files; 522,941,840 expanded bytes). During extraction, 55 range requests read 446,379,596 bytes, including cached range blocks. The challenge page rounds the India training count to about 9,000; this analysis uses the exact count in the Figshare archive.
- The Figshare record lists **CC BY 4.0**, while the CRDDC challenge page/rules state **CC BY-SA 4.0** for the images. The licenses conflict at the release-page level. Keep the source attribution and license caveat; confirm image-level terms before redistributing. Source images are local-only and ignored by Git.
- The release describes India collection across local roads, state highways, and national highways, including Delhi and Gurugram and mainly Haryana. The image subset does not contain coordinates or capture routes, and its collection period does not align individual images to crash dates. It is not a national road-condition survey.

The source manifest, including URLs, license statements, API MD5 provenance, archive layout, and selective-extraction counts, is at `data/raw/rdd2022/source_manifest.json`.

## Label audit and inclusion policy

The profile parser found 7,706 paired images and XML files. It retained the four challenge classes and recorded these additional source labels:

| Out-of-scope source code | Annotation instances |
|---|---:|
| D44 | 1,062 |
| D01 | 179 |
| D11 | 45 |
| D50 | 28 |
| D43 | 57 |
| D0w0 (malformed/unknown tag) | 1 |

There are 1,056 images with at least one such out-of-scope or malformed class tag. Those images are excluded from both fitting and scoring, including images that also have one of the four target labels; this prevents known defects from becoming false background labels. The audit found 3,223 images with at least one target-class box before this exclusion. After exclusion, 6,650 scope-clean images remain: 2,729 contain at least one target box, and 3,921 have no target boxes and no out-of-scope labels.

Clean empty-label images stay in validation so the false-positive evaluation includes negative images. Treating an empty XML as background assumes the source annotations are complete for the selected classes; completeness cannot be independently verified. Empty-only batches produced a non-finite torchvision classification loss during an earlier fitting attempt, so empty-target images are excluded from fitting; positive images still provide background anchors and proposals. No metrics from the interrupted attempts are used below.

Selected-class instance counts in the full scope-clean data, before split and cap, are D00 **1,419**, D10 **63**, D20 **1,635**, and D40 **2,956**. D10 is especially sparse.

## Split and training protocol

- Group split: 79 training groups and 20 validation groups, with seed 42. The group key is the integer suffix of the image filename divided into 100-ID blocks. A deterministic SHA-256 ranking assigns whole blocks to the holdout.
- This is a **filename-sequence proxy**, not a verified geographic, route, or capture-session holdout. The release does not provide the metadata needed for a true spatial holdout, so nearby-frame independence is not established.
- Scope-clean split availability: 5,294 training images (2,192 target-positive; 3,102 empty) and 1,356 validation images (537 target-positive; 819 empty).
- Capped run: 1,200 target-positive training images and 300 validation images. The stable hash cap reserves an image for each available target class before filling the remaining slots. The selected validation sample contains 181 clean empty images and 119 target-positive images.
- Model: Torchvision Faster R-CNN with MobileNetV3-Large 320 FPN, initialized from Torchvision COCO weights and fine-tuned for the four target classes. Optimizer: AdamW, learning rate 0.00005, batch size 2, one epoch, seed 42, CPU.
- Training plus validation duration: 1,275.48 seconds; mean training loss: 0.4548.
- Per-class target boxes in the selected training sample: D00 579, D10 29, D20 714, D40 1,330. Validation boxes: D00 72, D10 3, D20 74, D40 122.

The exact source and selected sample IDs are in `data/processed/vision/grouped_split.json` and `data/processed/vision/selected_run_split.json`. Dataset and label counts are in `data/processed/vision/dataset_profile.json`.

## Held-out metrics

Average precision uses all scored predictions, with IoU thresholds from 0.50 to 0.95 in 0.05 increments. The fixed-threshold precision, recall, and F1 use score ≥ 0.50 and IoU ≥ 0.50.

| Class | Validation boxes | AP@0.50 | AP@0.50:0.95 | Precision @0.50 | Recall @0.50 | F1 @0.50 |
|---|---:|---:|---:|---:|---:|---:|
| D00 — longitudinal crack | 72 | 0.0447 | 0.0059 | 0.0000 | 0.0000 | 0.0000 |
| D10 — transverse crack | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| D20 — alligator crack | 74 | 0.3893 | 0.1271 | 0.5814 | 0.3378 | 0.4274 |
| D40 — pothole | 122 | 0.0718 | 0.0213 | 0.0000 | 0.0000 | 0.0000 |
| **Macro mean AP** | — | **0.1264** | **0.0386** | — | — | — |

Micro precision/recall/F1 at score ≥ 0.50 and IoU ≥ 0.50 are **0.5814 / 0.0923 / 0.1592**. At that threshold, only D20 has true-positive detections. D10's validation set contains only three boxes; its result is not a reliable class estimate. The low recall and AP show that this one-epoch run is a baseline pilot, not a useful operational detector.

The machine-readable metrics, per-class AP at every IoU, training history, model setup, and sample sizes are saved in `data/processed/vision/metrics.json`. A held-out D20 example prediction is saved at `data/processed/vision/predictions/val_positive_d20.json` (image `India_002107`; confidence 0.9092 and IoU 0.862 against its D20 annotation). This is one illustrative match, not a performance estimate. The fitted 76 MB checkpoint is local at `data/processed/vision/fasterrcnn_mobilenetv3_rdd2022_india.pt` and is ignored by Git.

## Reproduce

From the project root, with Python 3.11 or 3.12:

```powershell
uv sync --extra vision
uv run --extra vision python src/road_safety/vision.py remote-index
uv run --extra vision python src/road_safety/vision.py download-india --max-compressed-gb 1.5
uv run --extra vision python src/road_safety/vision.py profile --group-size 100
uv run --extra vision python src/road_safety/vision.py split --seed 42 --validation-fraction 0.2 --group-size 100
uv run --extra vision python src/road_safety/vision.py train
```

The default training command is the capped one-epoch run (up to 1,200 train and 300 validation images). It downloads Torchvision's COCO initialization weights on the first run (74.2 MB in this environment). For a larger fit over the full scope-clean grouped split, use `uv run --extra vision python src/road_safety/vision.py train --full-run --epochs 10`; this can take substantially longer on CPU. To score an image with the saved checkpoint:

```powershell
uv run --extra vision python src/road_safety/vision.py predict --checkpoint data/processed/vision/fasterrcnn_mobilenetv3_rdd2022_india.pt --output data/processed/vision/predictions/example.json data/raw/rdd2022/India/train/images/India_002107.jpg
```

## Limitations and next steps

1. The holdout is not geographic or route-disjoint. Obtain capture-route metadata or a new independently collected corridor set before claiming transfer to unseen roads.
2. A one-epoch, capped CPU run is a reproducibility baseline. Tune learning rate, duration, augmentation, thresholds, and class-aware sampling only on training/validation data; reserve a truly independent labelled set for final performance claims.
3. D10 is rare and has only three validation instances in this sample. Collect or select more examples and report uncertainty for rare classes.
4. Only four labels are modeled. Images with other recorded road-damage tags were excluded rather than treated as clean background. Revisit the task taxonomy before broader use.
5. Empty target labels are treated as background only when no out-of-scope XML tag is present; annotation completeness for the remaining images has not been independently audited.
6. The challenge test set has images but no public labels; it was not scored.
7. The RDD2022 image set is separate from the crash-event data. No crash-risk, causal, or location-level fusion claim is supported. Any field use requires geotagged imagery, domain validation, and engineer review.
8. Figshare and CRDDC release pages disagree on the image license. Confirm source-specific terms before redistributing images, weights derived from them, or an image-containing demo.

**Use decision:** retain this detector as a separate, exploratory condition-monitoring component. Do not combine it with crash counts or display image detections as evidence for a crash hotspot until an auditable geospatial/time linkage and an independent validation design exist.
