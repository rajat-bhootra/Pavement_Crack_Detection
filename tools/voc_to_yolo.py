#!/usr/bin/env python3
"""
Convert Pascal VOC XML annotations from the RDD2022 India training set
to Ultralytics YOLO format (txt files) and create a train/val split.

The script:
  - Reads Pascal VOC XML files.
  - Maps the original damage codes (D00, D01, D10, D11, D20, D40, D43, D44, D50, D0w0)
    to the final YOLO classes:
        0: longitudinal_crack   (D00, D01)
        1: transverse_crack     (D10, D11)
        2: alligator_crack      (D20)
        3: pothole              (D40)
    and ignores D43, D44, D50, D0w0.
  - Converts bounding boxes to normalized YOLO format:
        class_id x_center y_center width height
  - Performs a class-aware train/val split that guarantees at least one
    transverse_crack image in the validation set.
  - Copies images and creates label txt files.
  - Reports warnings for unexpected classes, missing images, malformed boxes, etc.

Run as:
    python tools/voc_to_yolo.py
"""

import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration (modify only if you move the source dataset)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # .../Pavement_Crack_Detection
SRC_IMAGES = Path("/home/rajat_bhootra/IIT_PKD/datasets/RDD2022/extracted/India/train/images")
SRC_ANNOTATIONS = Path("/home/rajat_bhootra/IIT_PKD/datasets/RDD2022/extracted/India/train/annotations/xmls")

OUT_BASE = BASE_DIR / "datasets" / "pavement"
OUT_TRAIN_IMAGES = OUT_BASE / "images" / "train"
OUT_VAL_IMAGES = OUT_BASE / "images" / "val"
OUT_TRAIN_LABELS = OUT_BASE / "labels" / "train"
OUT_VAL_LABELS = OUT_BASE / "labels" / "val"

# ---------------------------------------------------------------------------
# Class mapping (original code -> YOLO class id)
# ---------------------------------------------------------------------------
CLASS_MAP = {
    # longitudinal_crack
    "D00": 0,
    "D01": 0,
    # transverse_crack
    "D10": 1,
    "D11": 1,
    # alligator_crack
    "D20": 2,
    # pothole
    "D40": 3,
}

IGNORE_CLASSES = {"D43", "D44", "D50", "D0w0"}

# ---------------------------------------------------------------------------
def parse_xml(xml_path: Path):
    """Return (width, height, list of (class_id, xmin, ymin, xmax, ymax))."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return None, None, []
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        # Skip ignored classes
        if name in IGNORE_CLASSES:
            continue
        # Map known codes; unknown -> warn
        if name not in CLASS_MAP:
            print(f"[WARN] Unknown class '{name}' in {xml_path.name}; skipping object.")
            continue
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        # Basic box sanity
        if not (0 <= xmin < xmax <= w and 0 <= ymin < ymax <= h):
            print(f"[WARN] Malformed box in {xml_path.name}: ({xmin},{ymin})-({xmax},{ymax}) on image {w}x{h}; skipping.")
            continue

        class_id = CLASS_MAP[name]
        objects.append((class_id, xmin, ymin, xmax, ymax))

    return w, h, objects


def normalize_box(w: int, h: int, xmin, ymin, xmax, ymax):
    """Return (class_id, x_center, y_center, width, height) normalized."""
    x_center = (xmin + xmax) / 2.0 / w
    y_center = (ymin + ymax) / 2.0 / h
    box_w = (xmax - xmin) / w
    box_h = (ymax - ymin) / h
    return (0, x_center, y_center, box_w, box_h)  # placeholder class_id replaced later


def write_yolo_label(label_path: Path, entries):
    """Write a YOLO label file; entries is list of (class_id, x_center, y_center, w, h)."""
    lines = []
    for cid, cx, cy, cw, ch in entries:
        # Clip to [0,1] just in case of floating drift
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        cw = max(0.0, min(1.0, cw))
        ch = max(0.0, min(1.0, ch))
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {cw:.6f} {ch:.6f}")
    label_path.write_text("\n".join(lines) + "\n" if lines else "")


def main():
    # Ensure output directories exist
    OUT_TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_VAL_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_TRAIN_LABELS.mkdir(parents=True, exist_ok=True)
    OUT_VAL_LABELS.mkdir(parents=True, exist_ok=True)

    # Gather all XML files
    xml_files = sorted(SRC_ANNOTATIONS.glob("*.xml"))
    total_xml = len(xml_files)
    print(f"Found {total_xml} XML annotation files.")

    # ------------------------------------------------------------------
    # First pass: parse each XML, collect valid objects, and record which
    # images contain a transverse_crack (class 1).
    # ------------------------------------------------------------------
    image_info = []  # dicts: {filename, has_transverse, valid_objects}
    transverse_images = set()
    all_objects_by_class = {0: 0, 1: 0, 2: 0, 3: 0}  # counts per final class

    for xml_path in xml_files:
        img_name = xml_path.stem + ".jpg"  # assume .jpg extension matches
        img_path = SRC_IMAGES / img_name

        # Verify source image exists
        if not img_path.is_file():
            print(f"[WARN] Source image {img_name} not found; skipping XML.")
            continue

        w, h, objects = parse_xml(xml_path)
        if w is None:
            print(f"[WARN] Could not parse size in {xml_path.name}; skipping.")
            continue

        # Record whether this image has at least one transverse_crack
        has_transverse = False
        label_entries = []

        for cid, xmin, ymin, xmax, ymax in objects:
            label_entries.append((cid, xmin, ymin, xmax, ymax))
            all_objects_by_class[cid] = all_objects_by_class.get(cid, 0) + 1
            if cid == 1:  # transverse_crack
                has_transverse = True

        image_info.append({
            "filename": img_name,
            "path": img_path,
            "width": w,
            "height": h,
            "has_transverse": has_transverse,
            "objects": label_entries,  # keep raw for later writing
        })
        if has_transverse:
            transverse_images.add(img_name)

    print(f"Images processed: {len(image_info)}")
    print(f"Images with transverse_crack: {len(transverse_images)}")
    print(f"Class counts after mapping: {all_objects_by_class}")

    # ------------------------------------------------------------------
    # Create train/val split (stratified to keep transverse crack in both sets)
    # ------------------------------------------------------------------
    # Separate images that have transverse_crack from the rest
    transverse_imgs = [info for info in image_info if info["has_transverse"]]
    other_imgs = [info for info in image_info if not info["has_transverse"]]

    # Shuffle for randomness
    random.seed(42)  # reproducible
    random.shuffle(transverse_imgs)
    random.shuffle(other_imgs)

    # Reserve at least one transverse image for validation
    val_transverse = transverse_imgs[:1]
    train_transverse = transverse_imgs[1:]

    # For the remaining images, use ~10% for validation (adjust as needed)
    n_val_other = max(1, int(0.10 * len(other_imgs)))
    # Ensure we pick at most len(other_imgs)-1 so at least one stays in train
    n_val_other = min(n_val_other, len(other_imgs) - 1) if len(other_imgs) > 1 else 0
    val_other = random.sample(other_imgs, k=n_val_other)
    train_other = [info for info in other_imgs if info["filename"] not in {img["filename"] for img in val_other}]

    val_images = val_transverse + val_other
    train_images = train_transverse + train_other

    # Shuffle final lists for good measure
    random.shuffle(train_images)
    random.shuffle(val_images)

    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    print(f"  -> Val contains transverse_crack: {any(info['has_transverse'] for info in val_images)}")

    # ------------------------------------------------------------------
    # Second pass: copy images and write label files according to split
    # ------------------------------------------------------------------
    def process_image_set(images, dest_img_dir, dest_label_dir, prefix):
        """Copy images and write label files for a given split."""
        for info in images:
            # Copy image
            dest_img_path = dest_img_dir / info["filename"]
            # Use shutil.copy to preserve metadata
            import shutil
            shutil.copy2(info["path"], dest_img_path)

            # Build label entries (normalized boxes)
            w = info["width"]
            h = info["height"]
            label_entries = info["objects"]

            label_path = dest_label_dir / (Path(info["filename"]).stem + ".txt")
            yolo_lines = []
            for cid, xmin, ymin, xmax, ymax in label_entries:
                # Normalize
                cx = (xmin + xmax) / 2.0 / w
                cy = (ymin + ymax) / 2.0 / h
                cw = (xmax - xmin) / w
                ch = (ymax - ymin) / h
                yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {cw:.6f} {ch:.6f}")
            label_path.write_text("\n".join(yolo_lines) + "\n" if yolo_lines else "")

    process_image_set(train_images, OUT_TRAIN_IMAGES, OUT_TRAIN_LABELS, "train")
    process_image_set(val_images, OUT_VAL_IMAGES, OUT_VAL_LABELS, "val")

    # ------------------------------------------------------------------
    # Write data.yaml
    # ------------------------------------------------------------------
    data_yaml = OUT_BASE / "data.yaml"
    yaml_content = f"""
path: {OUT_BASE}  # dataset root
train: images/train   # train images (relative to path)
val: images/val       # val images (relative to path)

# NC (number of classes)
nc: 4

# Class names
names:
    0: longitudinal_crack
    1: transverse_crack
    2: alligator_crack
    3: pothole
"""
    data_yaml.write_text(yaml_content.strip() + "\n")
    print(f"Created {data_yaml}")

    # ------------------------------------------------------------------
    # Summary / verification
    # ------------------------------------------------------------------
    # Count label files per split
    train_lbls = list(OUT_TRAIN_LABELS.glob("*.txt"))
    val_lbls = list(OUT_VAL_LABELS.glob("*.txt"))
    print(f"\nLabel files – train: {len(train_lbls)}, val: {len(val_lbls)}")

    # Count objects per class from label files
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for lbl in train_lbls + val_lbls:
        with open(lbl, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cid = int(parts[0])
                    if cid in class_counts:
                        class_counts[cid] += 1
    print(f"Object counts per YOLO class (train+val): {class_counts}")

    # Verify that every image has a corresponding label file
    train_imgs = set(p.name for p in OUT_TRAIN_IMAGES.glob("*.jpg"))
    train_lbl_names = set(p.stem + ".txt" for p in train_lbls)  # actually label stem
    # Better: label files named same as image but .txt
    train_lbl_stems = set(p.stem for p in train_lbls)
    missing_labels = train_imgs - train_lbl_stems
    missing_images = train_lbl_stems - train_imgs
    if missing_labels:
        print(f"[WARN] {len(missing_labels)} train images without label files.")
    if missing_images:
        print(f"[WARN] {len(missing_images)} label files without corresponding images.")

    val_imgs = set(p.name for p in OUT_VAL_IMAGES.glob("*.jpg"))
    val_lbl_stems = set(p.stem for p in val_lbls)
    missing_labels_val = val_imgs - val_lbl_stems
    missing_images_val = val_lbl_stems - val_imgs
    if missing_labels_val:
        print(f"[WARN] {len(missing_labels_val)} val images without label files.")
    if missing_images_val:
        print(f"[WARN] {len(missing_images_val)} label files without corresponding images.")

    # Verify normalized coordinates are within [0,1]
    # (already ensured during writing, but double-check)
    coord_warnings = 0
    for split_dir, split_name in [(OUT_TRAIN_LABELS, "train"), (OUT_VAL_LABELS, "val")]:
        for lbl_path in split_dir.glob("*.txt"):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        coord_warnings += 1
                        continue
                    try:
                        cid, cx, cy, cw, ch = map(float, parts)
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= cw <= 1 and 0 <= ch <= 1):
                            coord_warnings += 1
                    except ValueError:
                        coord_warnings += 1
    if coord_warnings:
        print(f"[WARN] {coord_warnings} label lines had out-of-range coordinates.")
    else:
        print("All YOLO coordinates are valid (0 <= val <= 1).")

    print("\n=== Conversion complete ===")


if __name__ == "__main__":
    main()