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

**Current completed stages:**
- YOLO dataset preparation (bounding-box YOLO data).
- Local YOLO environment validation.
- A 1-epoch YOLO smoke test on the local RTX 2050, including validation and run artifacts.

**Stages not yet implemented:** Full YOLO training, SAM/SAM2 segmentation, calibration, measurement, severity estimation.

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
| **Python** | 3.12.14 (managed by pyenv) |
| **pyenv** | used to select Python 3.12.14 |
| **virtual environment** | `.venv` (created with `python -m venv .venv`) |
| **PyTorch** | 2.11.0 + cu128 (runtime CUDA 12.8) |
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

Each stage will depend on the successful completion of the previous one. At present the YOLO detection pipeline is ready for experimentation: the dataset has been prepared and the local training/validation pipeline has passed a 1-epoch smoke test. Full training and the later segmentation/measurement stages remain to be completed.

---

## Training

### Local environment status

The local CUDA/PyTorch issue has been resolved sufficiently for YOLO execution.

Verified locally:

- **GPU:** NVIDIA GeForce RTX 2050 (4 GB VRAM)
- **NVIDIA driver:** 610.43.02
- **CUDA UMD version reported by `nvidia-smi`:** 13.3
- **PyTorch:** 2.11.0+cu128
- **PyTorch CUDA runtime:** 12.8
- **Torchvision:** 0.26.0+cu128
- **Ultralytics:** 8.4.122
- `torch.cuda.is_available()` → `True`
- YOLO detects the RTX 2050 successfully.

The earlier SIGBUS problem was traced to a corrupted/incomplete `libcusparseLt.so.0` in the Python environment. Reinstalling the cuSPARSELt package replaced the damaged 42 MB library with a valid ~432 MB library. PyTorch can now be imported and CUDA is available.

### Local YOLO smoke test — completed

A conservative 1-epoch smoke test was successfully run on the RTX 2050 using:

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
    name=smoke_test-2
```

The run completed:

- Training images scanned: **6,944**
- Validation images scanned: **762**
- Corrupt images: **0**
- Epochs: **1/1**
- Training time: **~11 min 54 sec**
- Validation time: **~24.6 sec**
- GPU memory reported by Ultralytics: **~0.152 GB**
- Validation mAP50: **0.0211**
- Validation mAP50-95: **0.00641**

Class-wise validation results:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| longitudinal_crack | 112 | 156 | 0.00461 | 0.423 | 0.00832 | 0.00233 |
| transverse_crack | 1 | 1 | 0.00000 | 0.000 | 0.00000 | 0.00000 |
| alligator_crack | 168 | 199 | 0.00847 | 0.794 | 0.0678 | 0.0211 |
| pothole | 148 | 300 | 0.00464 | 0.417 | 0.00823 | 0.00219 |
| **all** | **762** | **656** | **0.00443** | **0.408** | **0.0211** | **0.00641** |

This is **only a smoke test**, not the final model. One epoch with `imgsz=320` is useful for verifying the complete training/validation pipeline, but the resulting metrics are not representative of final model performance.

### Smoke-test artifacts

The completed run was saved under:

```text
runs/detect/runs/pavement/smoke_test-2/
```

Important artifacts include:

```text
runs/detect/runs/pavement/smoke_test-2/
├── labels.jpg
├── weights/
│   ├── best.pt
│   └── last.pt
└── ...
```

`labels.jpg` contains a visualization of the **ground-truth dataset bounding boxes and labels** generated during training setup. It is not model prediction output.

To view it:

```bash
xdg-open runs/detect/runs/pavement/smoke_test-2/labels.jpg
```

The checkpoint from this smoke test can be used for a quick inference test, but it should **not** be treated as a final trained model.

### Full training status

**Full YOLO training has NOT been completed yet.**

The next step is to perform proper multi-epoch training on the institute HPC cluster. The local RTX 2050 has only 4 GB VRAM and approximately 8 GB system RAM, making the HPC GPUs much more suitable for final experiments.

---

## HPC / GPU cluster access

Institute HPC access is now available through the Slurm-managed Bhavani GPU cluster.

### Important usage rule

The login/master node must **not** be used for GPU jobs or training. GPU work must be submitted/run through Slurm on a GPU node. Jobs accidentally run on the master node may be killed by the cluster administrators.

Example interactive GPU allocation:

```bash
srun --partition=gpu01 --gres=gpu:1 --ntasks=1 --pty bash
```

After allocation, the shell runs on a compute node and GPU commands such as `nvidia-smi` are valid.

### Verified GPU partitions

The following GPU partitions were tested successfully:

| Partition | Node | GPU(s) | VRAM per GPU | Time limit |
|---|---|---|---:|---|
| `gpu01` | `node001` | 2 × NVIDIA A30 | 24 GB | 1 day |
| `gpu02` | `node002` | 2 × NVIDIA A30 | 24 GB | 1 day |
| `gpu03` | `node003` | 2 × NVIDIA A30 | 24 GB | 5 days |
| `gpu04` | `node004` | 2 × NVIDIA L40 | ~46 GB | 1 day |

The cluster currently reports NVIDIA driver **530.30.02** and CUDA **12.1** on these compute nodes.

`nvidia-smi` was successfully verified on all four GPU partitions.

The master node itself does not expose a usable NVIDIA GPU to the user, which is expected for a login/master node.

### Recommended HPC strategy

For final YOLO experiments:

1. Clone the repository on the HPC.
2. Set up the Python environment on a compute node or using the cluster's supported software modules.
3. Keep the raw RDD2022 archive and generated dataset outside Git.
4. Run preprocessing if the processed dataset is not already available.
5. Start with `gpu01`/`gpu02`/`gpu03` using an A30 (24 GB).
6. Prefer `gpu04` if an L40 allocation is available and larger experiments are required.
7. Use Slurm for all training jobs; do not train on `master`.
8. Save checkpoints and training results under `runs/`, which remains ignored by Git.

The final training configuration (model size, `imgsz`, batch size, epochs, workers, and augmentation) should be selected after a short HPC benchmark because the A30 and L40 have substantially more VRAM than the local RTX 2050.

A typical future Slurm/YOLO command will be based on:

```bash
yolo detect train \
    data=datasets/pavement/data.yaml \
    model=yolov8n.pt \
    epochs=<planned-epochs> \
    imgsz=<planned-image-size> \
    batch=<batch-size> \
    device=0 \
    project=runs/pavement \
    name=<experiment-name>
```

The exact values should be finalized after the first HPC smoke test.



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

## Current local hardware

- **CPU:** 12th Gen Intel Core i5-12450H, 12 logical CPUs
- **GPU:** NVIDIA GeForce RTX 2050 (4 GB VRAM)
- **System RAM:** approximately 8 GB
- **OS:** Kubuntu / Ubuntu-based Linux (amd64)
- **NVIDIA driver:** 610.43.02
- **System CUDA toolkit:** 13.3
- **PyTorch CUDA runtime:** 12.8

---

## Important project limitations

- The current YOLO dataset uses **bounding boxes**, not segmentation masks.
- **Transverse‑crack** is strongly under‑represented compared with the other classes.
- RDD2022 is useful for initial defect detection but does **not** by itself provide all information required for physical crack measurement or severity estimation.
- SAM / SAM2 will require additional segmentation‑oriented data / validation.
- Real‑world measurement / calibration will eventually require our own phone‑camera data and a dedicated calibration methodology.
- The local RTX 2050 has only **4 GB VRAM**.
- The previous local PyTorch SIGBUS issue has been resolved, and the local YOLO/CUDA stack is now working.
- Despite this, the HPC GPU cluster is the preferred environment for full training because its A30/L40 GPUs provide substantially more VRAM.

---

*End of README*