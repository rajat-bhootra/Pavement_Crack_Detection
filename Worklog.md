# Pavement Crack Detection — Research & Work Log

**Project:** `Pavement_Crack_Detection`  
**Purpose:** Detect pavement cracks and potholes from road images, then move toward segmentation and real-world measurement.  
**Current stage:** YOLO detection experiments completed; segmentation and measurement are the next major stages.

---

# Project Goal

The project was planned as:

**Phone camera image/video → YOLO detection → segmentation → camera calibration / road-plane geometry → crack measurement → severity analysis**

Target classes:

- Longitudinal crack
- Transverse crack
- Alligator crack
- Pothole

The first stage was kept focused on object detection before moving to pixel-level segmentation and physical measurement.

---

# Dataset Preparation

The project uses the **India subset of Road Damage Dataset 2022 (RDD2022)**.
RDD2022 contains road-damage images from multiple countries and includes the four damage types used in this project:
- D00 — Longitudinal crack
- D10 — Transverse crack
- D20 — Alligator crack
- D40 — Pothole

The original Pascal VOC XML annotations were converted to YOLO format.

### Dataset

| Split | Images |
|---|---:|
| Train | 6,945 |
| Validation | 763 |
| Test | 1 |

Total training instances:

| Class | Instances |
|---|---:|
| Longitudinal crack | 1,578 |
| Transverse crack | 112 |
| Alligator crack | 1,822 |
| Pothole | 2,887 |
| **Total** | **6,399** |

A major dataset issue identified early was class imbalance, especially the very small number of transverse-crack instances.

---

# Initial Dataset Analysis

Bounding-box sizes were analyzed before changing the model. A large percentage of defects occupy only a small portion of an image.

| Class | Boxes smaller than 5% of image |
|---|---:|
| Longitudinal crack | 1,329 / 1,578 |
| Transverse crack | 94 / 112 |
| Alligator crack | 475 / 1,822 |
| Pothole | 2,546 / 2,887 |

This showed that small-object detection is an important challenge, especially for longitudinal cracks and potholes.
This analysis motivated testing a higher input resolution.

---

# Experiment 1 — YOLOv8n @ 640×640

The first full model was **YOLOv8n at 640×640**.

Main setup:
- 100 epochs
- Automatic batch fitting
- Seed 0
- Deterministic training
- AMP enabled
- NVIDIA A30 on HPC

### Baseline result

| Metric | Result |
|---|---:|
| Precision | 0.505 |
| Recall | 0.617 |
| mAP50 | 0.599 |
| mAP50-95 | 0.255 |

This became the main reference model.
The model was able to detect pavement damage, but false positives and missed small defects were noticeable.

**Note:** mAP was used as a detection metric and was not treated as accuracy.

---

# Confidence Threshold Experiment

The baseline model was evaluated at different confidence thresholds.

| Confidence | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 0.25 | 0.531 | 0.618 | 0.535 | 0.229 |
| 0.40 | 0.482 | 0.275 | 0.231 | 0.110 |
| 0.50 | 0.538 | 0.217 | 0.193 | 0.094 |
| 0.60 | 0.592 | 0.170 | 0.154 | 0.079 |

The F1 analysis placed the best overall balance near confidence 0.26.

### Finding

Increasing confidence alone is not a real model improvement.
It can reduce false positives, but recall falls sharply. Therefore, the project continued with model/data improvements instead of relying on a higher confidence threshold.

---

# Confusion Matrix and Error Analysis

The baseline confusion matrix was examined to understand the errors.
The main problem was **background false positives**, rather than only confusion between the four defect classes.
At confidence 0.25, many normal road regions were incorrectly predicted as pavement damage.
There were also many missed defects, particularly:

- Longitudinal cracks
- Alligator cracks
- Potholes

This led to investigation of hard negatives and dataset characteristics.

---

# Hard-Negative Mining

There were **3,948 background-only training images**, approximately 56.8% of the training set.
The baseline detector was run over these images to find background images that generated false detections.
The first attempt to process all paths as a Python list caused GPU memory problems.

The approach was changed to:
- image list in a text file
- streaming inference
- batch size 1
- workers 0

This worked successfully.

### Result

**221 / 3,948** background-only images produced predictions at confidence 0.25.
Several contained relatively high-confidence false detections, including:

- alligator cracks
- longitudinal cracks
- potholes

This confirmed that normal road texture was an important source of false positives.

---

# Experiment 2 — Hard-Negative Oversampling

A separate experiment selected 20 visually confirmed hard-negative images.
They were repeated several times in the training list so the model would see them more often.

Setup:
- YOLOv8n
- 640×640
- 100 epochs
- Same main training settings as the baseline

### Result on the original validation set

| Metric | Baseline | Hard-negative experiment |
|---|---:|---:|
| Precision | 0.531 | 0.409 |
| Recall | 0.618 | 0.365 |
| mAP50 | 0.535 | 0.297 |
| mAP50-95 | 0.229 | 0.131 |

### Finding

The experiment reduced overall performance and was rejected.
It was still retained as a useful negative experiment: simply repeating a small set of hard negatives did not solve the false-positive problem.

---

# Experiment 3 — YOLOv8n @ 960×960

Because many defects were very small, the input resolution was increased.

Model: **YOLOv8n @ 960×960**

Training:
- 100 epochs
- Original dataset
- Seed 0
- Deterministic training
- AMP enabled
- NVIDIA A30

### Automatic validation result

| Metric | Result |
|---|---:|
| Precision | 0.614 |
| Recall | 0.518 |
| mAP50 | 0.593 |
| mAP50-95 | 0.345 |

For a fair comparison, validation was also run at confidence 0.25.

### Comparison to 640 x 640 resolution

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n @ 640 | 0.531 | 0.618 | 0.535 | 0.229 |
| YOLOv8n @ 960 | 0.609 | 0.517 | 0.501 | 0.308 |

### Finding

960×960 produced:
- higher precision
- higher mAP50-95
- lower recall
- similar mAP50

The improvement in mAP50-95 showed better strict localization performance.
This made **YOLOv8n @ 960** the current candidate.

---

# Experiment 4 — YOLOv8s @ 960×960

The next experiment tested whether a larger YOLO model would improve the results.
Model: **YOLOv8s @ 960×960**

Training:
- 100 epochs
- Original dataset
- Same main training settings

### Result

| Metric | Result |
|---|---:|
| Precision | 0.345 |
| Recall | 0.382 |
| mAP50 | 0.377 |
| mAP50-95 | 0.161 |

The larger model performed worse in this setup.

### Finding

Increasing model size from YOLOv8n to YOLOv8s did not improve the experiment.
The YOLOv8s 960 experiment was therefore rejected.

---

# Experiment 5 — RDD2022 Multi-Country Training

A controlled experiment was performed to test whether adding RDD2022 data from other countries improves detection performance on the project's India validation set.

### Dataset construction

Training data was combined from:
- India
- Czech Republic
- Japan
- United States

The same four project classes were retained:
- Longitudinal crack
- Transverse crack
- Alligator crack
- Pothole

The existing fixed **762-image India validation set** was kept unchanged so the experiment could be compared directly with the previous India-only models.

The resulting training dataset contained:

| Split | Images |
|---|---:|
| Training | 25,084 |
| Validation | 762 |

Training objects:

| Class | Instances |
|---|---:|
| Longitudinal crack | 13,365 |
| Transverse crack | 7,785 |
| Alligator crack | 9,015 |
| Pothole | 5,462 |
| **Total** | **35,627** |

Dataset integrity was verified:

- Training images: 25,084
- Training labels: 25,084
- Validation images: 762
- Validation labels: 762
- Images without labels: 0
- Labels without images: 0

The original `datasets/pavement/` dataset was kept unchanged.

### Model and training

Model: **YOLOv8n @ 960×960**

Training configuration:
- 100 epochs
- Batch size: 22
- Seed: 0
- Deterministic training
- AMP enabled
- Same YOLOv8n architecture and 960×960 resolution as the India-only 960 experiment

The completed experiment is stored at: `runs/detect/runs/pavement/yolov8n_960_multicountry/`

Important outputs:
- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `args.yaml`

### Evaluation

The model was evaluated on the same fixed **762-image India validation set containing 656 annotated objects**.

Overall result:

| Metric | Multi-country YOLOv8n @ 960 |
|---|---:|
| Precision | 0.520 |
| Recall | 0.556 |
| mAP50 | 0.490 |
| mAP50-95 | 0.208 |

Per-class results:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Longitudinal crack | 0.539 | 0.353 | 0.398 | 0.205 |
| Transverse crack | 0.423 | 0.846 | 0.497 | 0.164 |
| Alligator crack | 0.568 | 0.582 | 0.594 | 0.283 |
| Pothole | 0.548 | 0.443 | 0.470 | 0.178 |
| **Overall** | **0.520** | **0.556** | **0.490** | **0.208** |

### Comparison with previous models

Using the same India validation set:

| Experiment | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n @ 640, India | 0.531 | 0.618 | 0.535 | 0.229 |
| YOLOv8n @ 960, India | 0.609 | 0.517 | 0.501 | 0.308 |
| **YOLOv8n @ 960, India+Czech+Japan+USA** | **0.520** | **0.556** | **0.490** | **0.208** |

### Finding

Adding the Czech, Japan and United States RDD2022 training data did **not improve the overall aggregate metrics** on the fixed India validation set compared with the India-only 960 experiment.

Relative to India-only YOLOv8n @ 960:

- Precision changed from 0.609 to 0.520.
- Recall changed from 0.517 to 0.556.
- mAP50 changed from 0.501 to 0.490.
- mAP50-95 changed from 0.308 to 0.208.

Thus, the multi-country experiment showed a higher recall but lower precision and lower mAP metrics than the India-only 960 model.
This experiment is retained as a documented dataset/generalization experiment. It does not replace the existing India-only 960 result as the current detection candidate.

---


# Experiment 6 — Clean Cross-Country Validation

A second, cleaner cross-country experiment was performed to determine whether the multi-country training actually improves generalization to countries that were not represented in the training portion of each country's data.

Unlike Experiment 5, this experiment used a separate deterministic 90/10 train/validation split for each country's annotated images.

### Dataset construction

The same four RDD2022 countries and four project classes were used:

- India
- Czech Republic
- Japan
- United States

Class mapping remained:

- D00/D01 → Longitudinal crack
- D10/D11 → Transverse crack
- D20 → Alligator crack
- D40 → Pothole

Each country's annotated images were split into approximately 90% training and 10% validation data.

### Dataset sizes

| Country | Train images | Validation images |
|---|---:|---:|
| India | 6,935 | 771 |
| Czech Republic | 2,546 | 283 |
| Japan | 9,455 | 1,051 |
| United States | 4,325 | 480 |
| **Total** | **23,261** | **2,585** |

The resulting training dataset contained 23,261 images and the validation dataset contained 2,585 images. Training/validation integrity was checked and all images had corresponding labels.

### Models

Two YOLOv8n @ 960×960 models were compared.

**Model A — India-only**

- Training data: India 90% split
- Validation: each country's 10% validation split
- Checkpoint: `runs/detect/runs/pavement/yolov8n_960_india90/weights/best.pt`

**Model B — Multi-country**

- Training data: India + Czech + Japan + United States, using the 90% training portion from each country
- Validation: the same four country-specific 10% validation splits
- Checkpoint: `runs/detect/runs/pavement/yolov8n_960_crosscountry90/weights/best.pt`

Both models were evaluated using 960×960 input, batch size 16, the same four classes, the same country-specific validation sets, and the same validation procedure.

This makes the comparison a controlled test of the effect of geographically diverse training data.

### Model A — India-only results

| Validation country | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| India | 0.533 | 0.347 | 0.396 | 0.172 |
| Czech Republic | 0.070 | 0.047 | 0.028 | 0.007 |
| Japan | 0.133 | 0.100 | 0.041 | 0.015 |
| United States | 0.124 | 0.175 | 0.060 | 0.020 |

The India-only model showed a large drop in performance when evaluated on the other countries, demonstrating substantial domain/generalization differences across the RDD2022 country subsets.

### Model B — Multi-country results

| Validation country | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| India | 0.587 | 0.440 | 0.472 | 0.217 |
| Czech Republic | 0.457 | 0.405 | 0.387 | 0.146 |
| Japan | 0.616 | 0.564 | 0.592 | 0.282 |
| United States | 0.549 | 0.630 | 0.619 | 0.384 |

### Direct comparison

Using mAP50:

| Validation country | India-only | Multi-country | Absolute change |
|---|---:|---:|---:|
| India | 0.396 | **0.472** | **+0.076** |
| Czech Republic | 0.028 | **0.387** | **+0.359** |
| Japan | 0.041 | **0.592** | **+0.551** |
| United States | 0.060 | **0.619** | **+0.559** |

The multi-country model improved over the India-only model on all four validation countries.

The improvement was particularly large on the three foreign validation sets:

- Czech Republic: mAP50 0.028 → 0.387
- Japan: mAP50 0.041 → 0.592
- United States: mAP50 0.060 → 0.619

The model also improved on the Indian validation set:

- India: mAP50 0.396 → 0.472

### Finding

The clean 90/10 cross-country experiment provides strong evidence that geographically diverse RDD2022 training data improves cross-country generalization.

The India-only model performed substantially better on Indian validation images than on Czech, Japanese, or United States validation images. In contrast, the multi-country model maintained much stronger performance across all four country-specific validation sets.

Importantly, the multi-country model did not achieve this by sacrificing Indian performance: its India validation mAP50 also increased from 0.396 to 0.472.

Therefore, for the detection/generalization component of this project, the multi-country YOLOv8n @ 960 model is preferred over the India-only YOLOv8n @ 960 model.

This experiment is considered the stronger cross-country result than the earlier aggregate India-validation experiment because each country has its own held-out validation split and both models are evaluated on exactly the same country-specific validation sets.


# Current Model Decision

Current experiment status:

| Experiment | Status |
|---|---|
| YOLOv8n @ 640 | Baseline / reference |
| YOLOv8n @ 960, India-only | Strong India-only reference |
| YOLOv8s @ 960 | Rejected |
| Hard-negative oversampling | Rejected |
| RDD2022 multi-country on fixed India validation | Completed; aggregate India result did not beat India-only 960 |
| **Clean 90/10 multi-country validation** | **Preferred detection/generalization model** |

### Current detection candidate

For the detection/generalization stage, the current preferred model is:

**YOLOv8n @ 960 trained on India + Czech Republic + Japan + United States**

Checkpoint:

`runs/detect/runs/pavement/yolov8n_960_crosscountry90/weights/best.pt`

The decision is based on the clean country-specific 90/10 validation experiment. The multi-country model achieved higher mAP50 than the India-only model on every validation country:

| Validation country | India-only mAP50 | Multi-country mAP50 |
|---|---:|---:|
| India | 0.396 | **0.472** |
| Czech Republic | 0.028 | **0.387** |
| Japan | 0.041 | **0.592** |
| United States | 0.060 | **0.619** |

Thus, the multi-country model is currently preferred because it provides substantially better cross-country generalization while also improving performance on the Indian validation split.

The earlier Experiment 5 result on the fixed India validation set remains documented, but it is no longer the main basis for model selection. The clean country-specific 90/10 experiment provides the stronger evidence for the generalization decision.

The detector remains an intermediate stage and is not yet the final pavement measurement system. The next major stage is segmentation followed by camera calibration, road-plane geometry, physical measurement, and severity analysis.


# Segmentation Research

After detection, the project will move toward **pixel-level segmentation**.
A bounding box tells where the damage is, but it does not describe the exact crack shape.
Segmentation can provide a pixel-level mask, which is needed for later measurement.

Research was reviewed around:
- U-Net and related crack-segmentation models
- HRNet-based crack measurement
- SegFormer-based approaches
- SAM / SAM2
- Mask R-CNN
- crack centerline and width measurement

The research indicates that segmentation can support:
- crack length
- crack width
- crack area
- crack density
- severity-related measurements

However, a segmentation mask by itself is still measured in pixels.

---

# Camera Calibration and Homography Research

For real-world measurements, image pixels need to be related to physical dimensions.

The planned process is:
1. Camera calibration
2. Lens distortion correction
3. Road-plane estimation
4. Homography / perspective transformation
5. Conversion to physical coordinates
6. Crack length and width calculation

Homography is useful when the road surface can reasonably be treated as a plane.
However, homography alone does not provide centimetres or millimetres. A known physical scale, calibrated geometry, or reference object is still required.
A simple global `mm/px` value should not be assumed for uncontrolled phone images.

---

# Future Field Dataset

RDD2022 is suitable for the detection stage, but it does not provide the physical crack measurements required for the final measurement stage.
A dedicated phone-camera field dataset will therefore be collected later.

It should record:
- camera/phone model
- image resolution
- camera height
- camera angle/orientation
- road surface
- physical reference or scale
- segmentation ground truth
- manually measured crack dimensions

The exact phone orientation and capture setup will be finalized before measurement data collection.
Final measurement accuracy should be evaluated against real physical measurements.
