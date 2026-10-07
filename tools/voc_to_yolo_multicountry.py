#!/usr/bin/env python3

"""
Build a multi-country RDD2022 dataset for pavement crack detection.

Training countries:
    India, Czech, Japan, United_States

Validation:
    The exact existing India validation set from datasets/pavement/images/val

Class mapping:
    D00, D01 -> 0 longitudinal_crack
    D10, D11 -> 1 transverse_crack
    D20      -> 2 alligator_crack
    D40      -> 3 pothole

Ignored:
    D43, D44, D50, D0w0

The existing datasets/pavement/ directory is NOT modified.
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RDD_BASE = Path.home() / "datasets" / "RDD2022"

COUNTRIES = [
    "India",
    "Czech",
    "Japan",
    "United_States",
]

OUT_BASE = BASE_DIR / "datasets" / "pavement_multicountry"

TRAIN_IMAGES = OUT_BASE / "images" / "train"
TRAIN_LABELS = OUT_BASE / "labels" / "train"

VAL_IMAGES = OUT_BASE / "images" / "val"
VAL_LABELS = OUT_BASE / "labels" / "val"

# Existing India validation set
EXISTING_VAL_IMAGES = BASE_DIR / "datasets" / "pavement" / "images" / "val"
EXISTING_VAL_LABELS = BASE_DIR / "datasets" / "pavement" / "labels" / "val"


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
# Parse VOC XML
# ============================================================

def parse_xml(xml_path):
    """
    Return:
        width, height, objects

    objects:
        [(class_id, xmin, ymin, xmax, ymax), ...]
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")

    if size is None:
        raise ValueError("Missing <size>")

    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []

    for obj in root.findall("object"):

        name_node = obj.find("name")

        if name_node is None or name_node.text is None:
            continue

        name = name_node.text.strip()

        # Ignore classes outside our four-class problem
        if name in IGNORE_CLASSES:
            continue

        # Ignore anything else that isn't explicitly mapped
        if name not in CLASS_MAP:
            continue

        bnd = obj.find("bndbox")

        if bnd is None:
            continue

        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        # Validate bounding box
        if not (
            0 <= xmin < xmax <= width
            and
            0 <= ymin < ymax <= height
        ):
            continue

        class_id = CLASS_MAP[name]

        objects.append(
            (class_id, xmin, ymin, xmax, ymax)
        )

    return width, height, objects


# ============================================================
# Convert bounding box
# ============================================================

def normalize_box(width, height, xmin, ymin, xmax, ymax):

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

def write_label(label_path, objects, width, height):

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

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not EXISTING_VAL_IMAGES.exists():
        raise RuntimeError(
            f"Existing validation directory not found:\n"
            f"{EXISTING_VAL_IMAGES}"
        )

    if not EXISTING_VAL_LABELS.exists():
        raise RuntimeError(
            f"Existing validation labels not found:\n"
            f"{EXISTING_VAL_LABELS}"
        )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    for directory in [
        TRAIN_IMAGES,
        TRAIN_LABELS,
        VAL_IMAGES,
        VAL_LABELS,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # First determine the fixed India validation filenames
    # --------------------------------------------------------

    validation_names = {
        p.name
        for p in EXISTING_VAL_IMAGES.glob("*.jpg")
    }

    print("=" * 70)
    print("MULTI-COUNTRY RDD2022 DATASET BUILD")
    print("=" * 70)

    print(f"\nExisting India validation images: {len(validation_names)}")

    if len(validation_names) != 762:
        print(
            "[WARN] Expected 762 validation images, "
            f"found {len(validation_names)}"
        )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    train_images_count = Counter()
    train_objects_count = Counter()

    val_objects_count = Counter()

    skipped_no_image = Counter()
    skipped_no_objects = Counter()

    total_xml = Counter()

    # --------------------------------------------------------
    # Process every country
    # --------------------------------------------------------

    for country in COUNTRIES:

        print("\n" + "-" * 70)
        print(f"Processing: {country}")
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

        xml_files = sorted(
            xml_dir.glob("*.xml")
        )

        total_xml[country] = len(xml_files)

        print(f"XML files: {len(xml_files)}")

        country_train = 0
        country_val = 0

        for xml_path in xml_files:

            image_name = xml_path.stem + ".jpg"
            image_path = image_dir / image_name

            # ------------------------------------------------
            # Image/XML matching
            # ------------------------------------------------

            if not image_path.is_file():
                skipped_no_image[country] += 1
                continue

            # ------------------------------------------------
            # Parse XML
            # ------------------------------------------------

            try:
                width, height, objects = parse_xml(
                    xml_path
                )
            except Exception as exc:
                print(
                    f"[WARN] Failed XML "
                    f"{xml_path.name}: {exc}"
                )
                continue

            # ------------------------------------------------
            # Determine split
            #
            # Existing India validation images stay in val.
            # Everything else goes to train.
            # Foreign countries are all training data.
            # ------------------------------------------------

            is_validation = (
                country == "India"
                and image_name in validation_names
            )

            if is_validation:

                destination_image = (
                    VAL_IMAGES / image_name
                )

                destination_label = (
                    VAL_LABELS /
                    (xml_path.stem + ".txt")
                )

                shutil.copy2(
                    image_path,
                    destination_image
                )

                write_label(
                    destination_label,
                    objects,
                    width,
                    height,
                )

                country_val += 1

                for class_id, *_ in objects:
                    val_objects_count[class_id] += 1

            else:

                destination_image = (
                    TRAIN_IMAGES / image_name
                )

                destination_label = (
                    TRAIN_LABELS /
                    (xml_path.stem + ".txt")
                )

                shutil.copy2(
                    image_path,
                    destination_image
                )

                write_label(
                    destination_label,
                    objects,
                    width,
                    height,
                )

                country_train += 1

                train_images_count[country] += 1

                for class_id, *_ in objects:
                    train_objects_count[class_id] += 1

        print(
            f"Training images added:   {country_train}"
        )

        print(
            f"Validation images added: {country_val}"
        )

    # --------------------------------------------------------
    # Verify validation set
    # --------------------------------------------------------

    actual_val = {
        p.name
        for p in VAL_IMAGES.glob("*.jpg")
    }

    missing_val = validation_names - actual_val
    extra_val = actual_val - validation_names

    print("\n" + "=" * 70)
    print("VALIDATION SET VERIFICATION")
    print("=" * 70)

    print(f"Expected validation images: {len(validation_names)}")
    print(f"Actual validation images:   {len(actual_val)}")
    print(f"Missing validation images:  {len(missing_val)}")
    print(f"Extra validation images:    {len(extra_val)}")

    if missing_val:
        print("\nFirst missing validation images:")
        for name in sorted(missing_val)[:20]:
            print(" ", name)

        raise RuntimeError(
            "Validation set is incomplete."
        )

    if extra_val:
        raise RuntimeError(
            "Unexpected extra validation images found."
        )

    # --------------------------------------------------------
    # Write data.yaml
    # --------------------------------------------------------

    data_yaml = OUT_BASE / "data.yaml"

    yaml_content = f"""path: {OUT_BASE}
train: images/train
val: images/val

nc: 4

names:
    0: longitudinal_crack
    1: transverse_crack
    2: alligator_crack
    3: pothole
"""

    data_yaml.write_text(
        yaml_content
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    total_train_images = sum(
        train_images_count.values()
    )

    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    print("\nTraining images by country:")

    for country in COUNTRIES:
        print(
            f"  {country:15s}: "
            f"{train_images_count[country]}"
        )

    print(
        f"\nTotal training images: "
        f"{total_train_images}"
    )

    print(
        f"Total validation images: "
        f"{len(actual_val)}"
    )

    print("\nTraining object counts:")

    total_train_objects = 0

    for class_id in range(4):

        count = train_objects_count[class_id]

        total_train_objects += count

        print(
            f"  {class_id} "
            f"{CLASS_NAMES[class_id]:20s}: "
            f"{count}"
        )

    print(
        f"  TOTAL: {total_train_objects}"
    )

    print("\nValidation object counts:")

    total_val_objects = 0

    for class_id in range(4):

        count = val_objects_count[class_id]

        total_val_objects += count

        print(
            f"  {class_id} "
            f"{CLASS_NAMES[class_id]:20s}: "
            f"{count}"
        )

    print(
        f"  TOTAL: {total_val_objects}"
    )

    print("\nXML files processed:")

    for country in COUNTRIES:
        print(
            f"  {country:15s}: "
            f"{total_xml[country]}"
        )

    print("\nWarnings / skipped files:")

    for country in COUNTRIES:

        if skipped_no_image[country]:
            print(
                f"  {country}: "
                f"{skipped_no_image[country]} "
                f"missing images"
            )

    print(
        f"\nCreated:\n{data_yaml}"
    )

    print("\n" + "=" * 70)
    print("DATASET BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
