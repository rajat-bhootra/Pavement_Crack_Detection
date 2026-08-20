# Pavement Crack and Pothole Detection

A research‑oriented project for pavement/road defect detection using phone‑camera images or videos.

---

## Project overview

The planned pipeline is:

```text
Phone camera image/video
  → YOLO defect detection + defect‑type classification
  → SAM / SAM2 crack segmentation
  → Camera / road‑plane calibration
  → Crack / pothole measurement
  → Severity analysis
```

**Current completed stage:** YOLO dataset preparation (bounding‑box YOLO‑v8 data).

**Stages not yet implemented:** SAM segmentation, calibration, measurement, severity estimation.

---

## Repository structure (relative to the cloned root)

```
Pavement_Crack_Detection/
├── README.md                # this file
├── .gitignore               # git ignore patterns
├── .python-version          # pins Python 3.12.14
├── datasets/
│   └── pavement/            # processed YOLO dataset – generated locally, ignored by Git
│       ├── data.yaml
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           └── val/
├── models/
│   └── .gitkeep             # placeholder – model weights not committed
├── scripts/
│   └── .gitkeep             # placeholder – utility scripts
└── tools/
    └── voc_to_yolo.py       # VOC → YOLO conversion script
```

*The raw RDD2022 dataset is **not** part of this repository (see “Dataset acquisition” below).*

---

## Required software

| Component | Minimum version / note |
|-----------|------------------------|
| **OS** | Ubuntu / Kubuntu (amd64) |
| **Python** | 3.12.14 (managed by pyenv) |
| **pyenv** | used to select Python 3.12.14 |
| **virtual environment** | `.venv` (created with `python -m venv .venv`) |
| **PyTorch** | 2.11.0 + cu128 (runtime CUDA 12.8) |
| **CUDA‑enabled NVIDIA GPU** | RTX 2050 (4 GB VRAM) |
| **NVIDIA driver** | 595.84 |
| **Ultralytics YOLO** | 8.4.122 |

---

## Environment setup from a fresh clone

```bash
# 1. Clone the repository
git clone <repository-url>
cd Pavement_Crack_Detection

# 2. Set the Python version
pyenv local 3.12.14       # or rely on the .python-version file

# 3. Create and activate the virtual environment
python -m venv .venv
source .venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install the required Python packages
pip install ultralytics==8.4.122          # YOLOv5/v8 interface
pip install torch==2.11.0+cu128 torchvision --extra-index-url https://download.pytorch.org/whl/cu128

# 5. Verify the environment
python -c "import ultralytics; print(ultralytics.__version__)"
# Should print “8.4.122”

# 6. (Optional) Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> **Important:** The local machine currently has a PyTorch / CUDA runtime problem – importing PyTorch can produce a SIGBUS crash involving `libcusparseLt` / `libcusparse`. If you encounter the same issue, the recommended next step is to use the institute HPC / GPU cluster. The machine can still be used for development, dataset inspection, preprocessing, Git, visualization, and inference / testing.

---

## Dataset acquisition

- **RDD2022** (Road Damage Dataset 2022) is the source dataset.
  - Official Figshare location: <https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547>
  - The **full archive** is **≈ 13.26 GB** and **is intentionally NOT committed to Git**.
- The **India subset** was extracted from the archive and is referenced only for preprocessing; it lives outside the repository.
  - Generic layout (chosen by the user):
    ```
    <workspace>/
    ├── Pavement_Crack_Detection/          # this Git repo
    └── datasets/
        └── RDD2022/
            ├── RDD2022_released_through_CRDDC2022.zip   # ≈13.26 GB
            ├── India.zip                               # extracted subset
            └── extracted/
                └── India/
                    ├── train/
                    │   ├── images/
                    │   └── annotations/
                    │       └── xmls/
                    └── test/
                        └── images/
  ```
- The user must place the downloaded archive under a `datasets/RDD2022/` directory **outside** the cloned repository and then run the conversion script, pointing it at that location.

> **Why the raw dataset is outside Git:** committing a ∼13 GB ZIP would make the repository unwieldy and violate typical Git best practices. Users should keep it in a sibling directory and adjust script paths accordingly.

---

## Dataset preprocessing

1. **Pascal VOC XML annotations** – the India subset provides XML files in Pascal‑VOC format.
2. **Run the conversion script** (requires the raw RDD2022 data to be available locally):

   ```bash
   source .venv/bin/activate
   python tools/voc_to_yolo.py \
       --rdd2022-root <path-to-datasets/RDD2022>
   ```

   The script:
   - Reads each XML file, maps the RDD2022 damage codes to the four YOLO classes,
   - Converts bounding boxes to normalised YOLO format (`class_id x_center y_center width height`),
   - Performs a class‑aware train/validation split (ensuring at least one transverse‑crack image in validation),
   - Writes `.txt` label files into `datasets/pavement/labels/train/` and `.../val/`,
   - Generates `datasets/pavement/data.yaml`.

3. **Resulting dataset layout** (tracked in Git):

   ```
   datasets/pavement/
       data.yaml
       images/
           train/   (6 944 images)
           val/     (762 images)
       labels/
           train/   (6 944 .txt files)
           val/     (762 .txt files)
   ```

4. **data.yaml** (portable – no absolute paths):

   ```yaml
   path: datasets/pavement          # relative to repo root
   train: images/train
   val: images/val

   nc: 4

   names:
     0: longitudinal_crack
     1: transverse_crack
     2: alligator_crack
     3: pothole
   ```

---

## Current YOLO classes

| YOLO class ID | Class name |
|---------------|------------|
| 0 | `longitudinal_crack` |
| 1 | `transverse_crack` |
| 2 | `alligator_crack` |
| 3 | `pothole` |

Other RDD2022 labels (D01, D11, D43, D44, D50, D0w0) are **intentionally ignored** for the current four‑class detector.

---

## Dataset validation results (India subset)

| Metric | Value |
|--------|-------|
| XML files | 7 706 |
| Total annotated objects (before filtering) | 8 203 |
| Valid target objects (the four YOLO classes) | 7 055 |
| Ignored objects (D43, D44, D50, D0w0) | 1 148 |
| Validation errors | 0 |
| **Processed train images** | **6 944** |
| **Processed validation images** | **762** |
| Class‑wise object counts (valid) | pothole: 3 187, longitudinal_crack: 1 734, alligator_crack: 2 021, transverse_crack: 113 |
| **Note** | Transverse‑crack is strongly under‑represented; class‑imbalance should be kept in mind. |

The processed train/validation counts come from running the conversion script on the India subset; the raw numbers (8 203 objects, 1 148 ignored) come from the original XML annotations.

---

## Git / repository rules

**Tracked (commit)**:

- `README.md`
- `.gitignore`
- `.python-version`
- `datasets/pavement/data.yaml`
- Source scripts and tools (`scripts/`, `tools/`, `.gitkeep` files)
- Model configuration / code

**Do NOT commit**:

- The 13 + GB RDD2022 ZIP archive
- Extracted raw RDD2022 dataset
- Generated image datasets that make the repo unnecessarily large
- `.venv/`
- Python cache files (`__pycache__/`, `.pyc`)
- `runs/` (training outputs)
- Large pretrained/model weight files (`*.pt`, `*.onnx`)
- `.env` files
- IDE/editor temporary files
- OS temporary files (`*.swp`, `*.swo`, `*~`)

The dataset must be downloaded separately following the instructions above.

---

## Intended future pipeline

```text
Phone camera image/video
  → YOLO defect detection + defect‑type classification
  → SAM / SAM2 crack segmentation
  → Camera / road‑plane calibration
  → Crack / pothole measurement
  → Severity analysis
  → Final testing on phone‑video
```

Each stage will depend on the successful completion of the previous one. At present only the YOLO detection stage (bounding‑box dataset) is ready.

---

## Training

**TRAINING HAS NOT BEEN SUCCESSFULLY COMPLETED YET.**

The first local training attempt caused system freezes / resource pressure, and a subsequent PyTorch import produced a SIGBUS crash involving CUDA libraries (`libcusparseLt`, `libcusparse`). Therefore:

- Do **not** claim that GPU training works on this machine.
- The recommended next step is to use the institute HPC / GPU cluster if access is granted, because the laptop has only 4 GB VRAM and ~8 GB system RAM.

If you still want a minimal smoke test on the local GPU, use the following **example command** (label it as a smoke test, not a completed result):

```bash
yolo detect train \
    data=datasets/pavement/data.yaml \
    model=yolov8n.pt \
    epochs=1 \
    imgsz=320 \
    batch=1 \
    workers=0 \
    device=0 \
    project=runs/pavement \
    name=yolov8n_smoke
```

- Start conservatively (small model, small input size, batch = 1, `workers=0`).
- Only increase image size, batch size, epochs, or model size after you confirm the smoke test runs without crashes.
- Final training parameters should be selected based on the HPC GPU’s available VRAM.

After a successful training run, Ultralytics creates a run directory under `runs/pavement/` and checkpoint files such as `weights/best.pt` and `weights/last.pt`.

---

## Inference

Once a trained checkpoint exists (e.g. `runs/pavement/<run-name>/weights/best.pt`), it can be used for inference:

```python
from ultralytics import YOLO
model = YOLO("runs/pavement/<run-name>/weights/best.pt")
results = model('path/to/image.jpg')   # or a video file / directory
```

The project currently does **not** have a trained checkpoint; inference can be set up after training finishes.

---

## Current hardware

- **GPU:** NVIDIA GeForce RTX 2050 (4 GB VRAM)
- **System RAM:** approximately 8 GB
- **OS:** Kubuntu / Ubuntu‑based Linux (amd64)

---

## Important project limitations

- The current YOLO dataset uses **bounding boxes**, not segmentation masks.
- **Transverse‑crack** is strongly under‑represented compared with the other classes.
- RDD2022 is useful for initial defect detection but does **not** by itself provide all information required for physical crack measurement or severity estimation.
- SAM / SAM2 will require additional segmentation‑oriented data / validation.
- Real‑world measurement / calibration will eventually require our own phone‑camera data and a dedicated calibration methodology.
- The local RTX 2050 has only **4 GB VRAM**.
- Local PyTorch / CUDA currently has a SIGBUS issue; an HPC GPU cluster is the preferred training environment.

---

## Style

- Clear headings, concise explanations, code blocks, and tables where useful.
- Checklists for project status.
- Do **not** make the README excessively long.
- Do **not** claim that training has completed, that SAM / measurement / severity has been implemented, or that the four classes are the only classes in the original RDD2022 India annotations.
- Do **not** expose large dataset files as Git‑tracked requirements.
- Make sure the README describes the **actual current state** of the repository.

--- 

*End of README*