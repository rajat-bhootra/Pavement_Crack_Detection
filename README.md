# Pavement Crack and Pothole Detection

Research/project workspace for pavement defect detection using phone-camera road images or videos.

## Current Stage

The current stage is YOLO-based pavement defect type detection.

Do not start segmentation, calibration, measurement, or severity estimation until the YOLO dataset and class definition are finalized.

## Planned Pipeline

```text
Phone camera image/video
-> YOLO defect detection and type classification
-> SAM/SAM2 segmentation
-> calibration and road-plane geometry
-> crack/pothole measurement
-> severity analysis
```

## Current Dataset Status

No dataset has been selected or downloaded yet.

Before training, compare candidate datasets and decide:

- final YOLO classes
- whether annotations are bounding boxes, masks, or both
- whether crack types and potholes are included
- whether images resemble road-level or phone-camera imagery
- whether the dataset can support future segmentation and measurement stages

## Initial Candidate Class Idea

These are only proposed classes and are not final:

```text
0 longitudinal_crack
1 transverse_crack
2 alligator_crack
3 pothole
```

The final class list must match or be carefully mapped from the selected dataset.

## Hardware Note

The available GPU is an NVIDIA GeForce RTX 2050 with 4 GB VRAM. Initial YOLO experiments should use a small model and conservative settings.
# Pavement_Crack_Detection
