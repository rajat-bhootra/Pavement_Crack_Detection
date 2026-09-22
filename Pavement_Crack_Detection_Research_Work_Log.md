# Pavement Crack Detection — Research & Work Log

**Project:** `Pavement_Crack_Detection`  
**Purpose:** Detect pavement cracks and potholes from road images, then move toward segmentation and real-world measurement.  
**Current stage:** YOLO detection experiments completed; segmentation and measurement are the next major stages.

---

## 1. Project Goal

The project was planned as:

**Phone camera image/video → YOLO detection → segmentation → camera calibration / road-plane geometry → crack measurement → severity analysis**

Target classes:

- Longitudinal crack
- Transverse crack
- Alligator crack
- Pothole

The first stage was kept focused on object detection before moving to pixel-level segmentation and physical measurement.

---

# 2. Dataset Preparation — August 2026

The project uses the **India subset of Road Damage Dataset 2022 (RDD2022)**.

RDD2022 contains road-damage images from multiple countries and includes the four damage types used in this project:

- D00 — Longitudinal crack
- D10 — Transverse crack
- D20 — Alligator crack
- D40 — Pothole

The India data was collected using vehicle-mounted smartphones, making it relevant to the planned phone-camera field work.

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

# 3. Initial Dataset Analysis

Bounding-box sizes were analyzed before changing the model.

A large percentage of defects occupy only a small portion of an image.

| Class | Boxes smaller than 5% of image |
|---|---:|
| Longitudinal crack | 1,329 / 1,578 |
| Transverse crack | 94 / 112 |
| Alligator crack | 475 / 1,822 |
| Pothole | 2,546 / 2,887 |

This showed that small-object detection is an important challenge, especially for longitudinal cracks and potholes.

This analysis motivated testing a higher input resolution.

---

# 4. Experiment 1 — YOLOv8n @ 640×640

### August 2026

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

# 5. Baseline Visual Inspection

A fixed set of 20 validation images was selected for qualitative comparison.

The YOLOv8n 640 model was run at confidence 0.25.

Results:

- 8 / 20 images produced detections.
- Some images produced multiple pothole predictions.
- Some difficult crack regions were detected with low confidence.
- Overlapping detections and false positives were observed.

Two example images were inspected closely:

- `India_003103`
- `India_005389`

This was used for qualitative understanding rather than as another numerical benchmark.

---

# 6. Confidence Threshold Experiment

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

# 7. Confusion Matrix and Error Analysis

The baseline confusion matrix was examined to understand the errors.

The main problem was **background false positives**, rather than only confusion between the four defect classes.

At confidence 0.25, many normal road regions were incorrectly predicted as pavement damage.

There were also many missed defects, particularly:

- Longitudinal cracks
- Alligator cracks
- Potholes

This led to investigation of hard negatives and dataset characteristics.

---

# 8. Hard-Negative Mining

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

# 9. Experiment 2 — Hard-Negative Oversampling

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

# 10. Experiment 3 — YOLOv8n @ 960×960

### August 2026

Because many defects were very small, the input resolution was increased.

Model:

**YOLOv8n @ 960×960**

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

### Apples-to-apples comparison

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

# 11. YOLOv8n 960 Qualitative Test

The same fixed 20-image validation set was used for qualitative comparison.

At confidence 0.25:

- 7 / 20 images produced detections.
- 12 predicted boxes were produced.

Predictions included longitudinal cracks, alligator cracks and potholes.

Some images still produced no detections, so the recall problem was not solved.

The results were retained for later visual comparison.

---

# 12. Experiment 4 — YOLOv8s @ 960×960

The next experiment tested whether a larger YOLO model would improve the results.

Model:

**YOLOv8s @ 960×960**

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

# 13. Current Model Decision

Current experiment status:

| Experiment | Status |
|---|---|
| YOLOv8n @ 640 | Baseline / reference |
| YOLOv8n @ 960 | **Current candidate** |
| YOLOv8s @ 960 | Rejected |
| Hard-negative oversampling | Rejected |

Current candidate checkpoint:

`runs/detect/runs/pavement/yolov8n_960-2/weights/best.pt`

The detector is still an intermediate stage and is not yet the final measurement system.

---

# 14. Segmentation Research

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

# 15. Camera Calibration and Homography Research

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

# 16. Future Field Dataset

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

---

# 17. Important Findings

### Small defects are a major challenge
Most longitudinal cracks and potholes occupy a small percentage of the image.

### Background false positives are significant
Normal road texture can be interpreted as pavement damage.

### Higher resolution helped
YOLOv8n at 960 improved precision and mAP50-95 compared with the 640 baseline at the common confidence threshold.

### A larger model did not automatically help
YOLOv8s at 960 performed worse than YOLOv8n at 960 in this experiment.

### Confidence threshold is not a model solution
Higher confidence reduces detections along with false positives.

### Measurement needs a separate field dataset
RDD2022 does not provide the physical measurements needed to validate real-world crack dimensions.

---

# 18. Computing Setup

## Local machine

- NVIDIA RTX 2050 4 GB
- NVIDIA driver 610.43.02
- CUDA Toolkit 13.3
- PyTorch 2.11.0+cu128
- Ultralytics 8.4.122

The local machine was used mainly for development, testing, visualization and transferring files.

## HPC

Training was moved to the institute HPC because the local GPU is limited.

Available GPUs included NVIDIA A30 and L40.

The experiments above were trained on an NVIDIA A30.

HPC environment:

- Python 3.12.14
- PyTorch 2.5.1+cu121
- torchvision 0.20.1+cu121
- Ultralytics 8.4.122

The HPC compute nodes did not have normal external internet access, so packages and model weights were transferred when required.

---

# 19. Repository / Research Organization

Important experiment material currently retained includes:

- `datasets/pavement/`
- `experiments/hard_negative_v1/`
- baseline YOLOv8n run
- YOLOv8n 960 run
- hard-negative experiment
- YOLOv8s 960 experiment
- validation/diagnostic outputs
- qualitative prediction outputs
- dataset conversion tools
- training scripts

Temporary duplicate folders outside the project were cleaned after their useful contents had been preserved.

---

# 20. Timeline

## August 2026

### Project and dataset
- Defined the pavement crack/pothole detection problem.
- Planned the detection → segmentation → measurement pipeline.
- Researched RDD2022.
- Prepared the India dataset.
- Converted Pascal VOC annotations to YOLO format.
- Verified image/label matching.

### Baseline
- Set up YOLOv8n.
- Trained YOLOv8n at 640×640.
- Evaluated precision, recall, mAP50 and mAP50-95.
- Performed qualitative testing.
- Tested confidence thresholds.
- Generated confusion/error diagnostics.

### Error analysis
- Identified background false positives.
- Analyzed object-size distribution.
- Performed hard-negative mining.
- Found 221 / 3,948 background images producing predictions.

### Hard-negative experiment
- Selected 20 hard negatives.
- Oversampled them.
- Trained a separate YOLOv8n model.
- Compared it with the baseline.
- Rejected the approach because overall performance decreased.

### Higher-resolution experiment
- Trained YOLOv8n at 960×960.
- Compared it fairly with the 640 baseline.
- Found improved precision and mAP50-95 but lower recall.
- Selected YOLOv8n 960 as the current candidate.

### Larger-model experiment
- Trained YOLOv8s at 960×960.
- Compared results.
- Rejected it because performance was lower.

### Measurement research
- Studied segmentation-based crack measurement.
- Studied camera calibration and homography.
- Established the need for a dedicated field dataset with physical measurements.

---

# 21. Current Status

### Completed

- RDD2022 India dataset preparation
- VOC → YOLO conversion
- Dataset verification
- YOLOv8n 640 baseline
- Confidence-threshold analysis
- Confusion/error analysis
- Bounding-box size analysis
- Hard-negative mining
- Hard-negative oversampling experiment
- YOLOv8n 960 experiment
- YOLOv8s 960 experiment
- Qualitative prediction comparison
- Initial segmentation research
- Initial calibration/homography research
- HPC training environment setup
- Repository cleanup and organization

### Current model candidate

**YOLOv8n @ 960×960**

### Next major stage

**Segmentation → camera calibration → geometric measurement → severity analysis**

---

# 22. Overall Research Direction

The project is being developed step-by-step rather than jumping directly to final measurements:

**Reliable detection → accurate segmentation → calibrated geometry → physical measurement → severity analysis**

Each stage will be evaluated separately before it is used as the foundation for the next stage.
