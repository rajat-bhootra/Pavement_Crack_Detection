#!/usr/bin/env python3

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RDD_BASE = Path.home() / "datasets" / "RDD2022"

COUNTRIES = [
    "India",
    "Czech",
    "Japan",
    "United_States",
]

SEED = 0
VAL_FRACTION = 0.10

OUT_BASE = BASE_DIR / "datasets" / "pavement_crosscountry"

TRAIN_IMAGES = OUT_BASE / "images" / "train"
TRAIN_LABELS = OUT_BASE / "labels" / "train"

VAL_IMAGES = OUT_BASE / "images" / "val"
VAL_LABELS = OUT_BASE / "labels" / "val"

LIST_DIR = BASE_DIR / "experiments" / "crosscountry"


# ============================================================
# Class mapping
# ============================================================

CLASS_MAP = {
    "D00": 0,
    "D01": 0,

    "D10": 1,
    "D11": 1,

    "D20": 2,

    "D40": 3,
}

CLASS_NAMES = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "pothole",
}

IGNORE_CLASSES = {
    "D43",
    "D44",
    "D50",
    "D0w0",
}


# ============================================================
# Parse XML
# ============================================================

def parse_xml(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")

    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []

    for obj in root.findall("object"):

        name_node = obj.find("name")

        if name_node is None or name_node.text is None:
            continue

        name = name_node.text.strip()

        if name in IGNORE_CLASSES:
            continue

        if name not in CLASS_MAP:
            continue

        bnd = obj.find("bndbox")

        if bnd is None:
            continue

        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        if not (
            0 <= xmin < xmax <= width
            and
            0 <= ymin < ymax <= height
        ):
            continue

        objects.append(
            (
                CLASS_MAP[name],
                xmin,
                ymin,
                xmax,
                ymax,
            )
        )

    return width, height, objects


# ============================================================
# Convert bounding box
# ============================================================

def normalize_box(
    width,
    height,
    xmin,
    ymin,
    xmax,
    ymax,
):

    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height

    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    return (
        x_center,
        y_center,
        box_width,
        box_height,
    )


# ============================================================
# Write YOLO label
# ============================================================

def write_label(
    label_path,
    objects,
    width,
    height,
):

    lines = []

    for class_id, xmin, ymin, xmax, ymax in objects:

        cx, cy, bw, bh = normalize_box(
            width,
            height,
            xmin,
            ymin,
            xmax,
            ymax,
        )

        lines.append(
            f"{class_id} "
            f"{cx:.6f} "
            f"{cy:.6f} "
            f"{bw:.6f} "
            f"{bh:.6f}"
        )

    label_path.write_text(
        "\n".join(lines) + "\n"
        if lines
        else ""
    )


# ============================================================
# Main
# ============================================================

def main():

    random.seed(SEED)

    print("=" * 70)
    print("RDD2022 CROSS-COUNTRY DATASET BUILD")
    print("=" * 70)

    print(f"Seed: {SEED}")
    print(f"Validation fraction: {VAL_FRACTION:.0%}")

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    for directory in [
        TRAIN_IMAGES,
        TRAIN_LABELS,
        VAL_IMAGES,
        VAL_LABELS,
        LIST_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    train_counts = Counter()
    val_counts = Counter()

    train_objects = Counter()
    val_objects = Counter()

    country_train_lists = {}
    country_val_lists = {}

    # --------------------------------------------------------
    # Process each country independently
    # --------------------------------------------------------

    for country in COUNTRIES:

        print("\n" + "-" * 70)
        print(f"Processing {country}")
        print("-" * 70)

        country_base = RDD_BASE / country / "train"

        image_dir = country_base / "images"
        xml_dir = country_base / "annotations" / "xmls"

        if not image_dir.exists():
            raise RuntimeError(
                f"Image directory not found:\n{image_dir}"
            )

        if not xml_dir.exists():
            raise RuntimeError(
                f"XML directory not found:\n{xml_dir}"
            )

        xml_files = sorted(xml_dir.glob("*.xml"))

        print(f"XML files found: {len(xml_files)}")

        # ----------------------------------------------------
        # Only use XMLs with matching images
        # ----------------------------------------------------

        valid_pairs = []

        for xml_path in xml_files:

            image_name = xml_path.stem + ".jpg"
            image_path = image_dir / image_name

            if not image_path.is_file():
                continue

            valid_pairs.append(
                (xml_path, image_path)
            )

        # ----------------------------------------------------
        # Deterministic random split
        # ----------------------------------------------------

        rng = random.Random(
            SEED + COUNTRIES.index(country)
        )

        rng.shuffle(valid_pairs)

        val_count = round(
            len(valid_pairs) * VAL_FRACTION
        )

        val_pairs = valid_pairs[:val_count]
        train_pairs = valid_pairs[val_count:]

        print(f"Valid image/XML pairs: {len(valid_pairs)}")
        print(f"Training:               {len(train_pairs)}")
        print(f"Validation:             {len(val_pairs)}")

        country_train_lists[country] = []
        country_val_lists[country] = []

        # ----------------------------------------------------
        # Process training images
        # ----------------------------------------------------

        for xml_path, image_path in train_pairs:

            # Prefix country to prevent filename collisions
            output_name = (
                f"{country}_{image_path.name}"
            )

            output_image = TRAIN_IMAGES / output_name

            output_label = (
                TRAIN_LABELS /
                f"{country}_{xml_path.stem}.txt"
            )

            width, height, objects = parse_xml(
                xml_path
            )

            shutil.copy2(
                image_path,
                output_image,
            )

            write_label(
                output_label,
                objects,
                width,
                height,
            )

            country_train_lists[country].append(
                output_name
            )

            train_counts[country] += 1

            for class_id, *_ in objects:
                train_objects[class_id] += 1

        # ----------------------------------------------------
        # Process validation images
        # ----------------------------------------------------

        for xml_path, image_path in val_pairs:

            output_name = (
                f"{country}_{image_path.name}"
            )

            output_image = VAL_IMAGES / output_name

            output_label = (
                VAL_LABELS /
                f"{country}_{xml_path.stem}.txt"
            )

            width, height, objects = parse_xml(
                xml_path
            )

            shutil.copy2(
                image_path,
                output_image,
            )

            write_label(
                output_label,
                objects,
                width,
                height,
            )

            country_val_lists[country].append(
                output_name
            )

            val_counts[country] += 1

            for class_id, *_ in objects:
                val_objects[class_id] += 1

    # --------------------------------------------------------
    # Write country-specific validation lists
    # --------------------------------------------------------

    for country in COUNTRIES:

        train_list = sorted(
            country_train_lists[country]
        )

        val_list = sorted(
            country_val_lists[country]
        )

        (LIST_DIR / f"{country.lower()}_train.txt").write_text(
            "\n".join(train_list) + "\n"
        )

        (LIST_DIR / f"{country.lower()}_val.txt").write_text(
            "\n".join(val_list) + "\n"
        )

    # --------------------------------------------------------
    # Write data.yaml
    # --------------------------------------------------------

    data_yaml = OUT_BASE / "data.yaml"

    data_yaml.write_text(
        f"""path: {OUT_BASE}
train: images/train
val: images/val

nc: 4

names:
    0: longitudinal_crack
    1: transverse_crack
    2: alligator_crack
    3: pothole
"""
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    print("\nImages by country:")

    for country in COUNTRIES:

        print(
            f"{country:15s} "
            f"train={train_counts[country]:5d} "
            f"val={val_counts[country]:5d}"
        )

    print(
        f"\nTotal train images: "
        f"{sum(train_counts.values())}"
    )

    print(
        f"Total val images:   "
        f"{sum(val_counts.values())}"
    )

    print("\nTraining objects:")

    for class_id in range(4):

        print(
            f"  {class_id}: "
            f"{CLASS_NAMES[class_id]:20s} "
            f"{train_objects[class_id]}"
        )

    print("\nValidation objects:")

    for class_id in range(4):

        print(
            f"  {class_id}: "
            f"{CLASS_NAMES[class_id]:20s} "
            f"{val_objects[class_id]}"
        )

    print("\nValidation lists:")

    for country in COUNTRIES:

        print(
            f"  {LIST_DIR / (country.lower() + '_val.txt')}"
        )

    print("\nDataset:")
    print(f"  {OUT_BASE / 'data.yaml'}")

    print("\nDONE")


if __name__ == "__main__":
    main()
