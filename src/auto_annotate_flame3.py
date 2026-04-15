"""
Auto-annotate FLAME 3 fire images using a pre-trained YOLOv8 model.

This script ONLY creates .txt label files alongside the original images.
It does NOT copy any images (to save disk space).

Strategy:
- Uses pre-trained YOLOv8s (COCO) to detect fire-like regions
- For images where the model detects something, bounding boxes are saved
- For images where nothing is detected (but we know it's fire), full-image fallback bbox is used
- Label files are created in data/raw/flame3_labels/ directory

Run: python src/auto_annotate_flame3.py
"""
import os
import sys
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

FLAME3_BASE = os.path.join("data", "raw", "flame1", "FLAME 3 CV Dataset (Sycan Marsh)")
FIRE_DIR = os.path.join(FLAME3_BASE, "Fire", "RGB", "Raw")
NO_FIRE_DIR = os.path.join(FLAME3_BASE, "No Fire", "RGB", "Raw")

# Output directory for label files ONLY (no image copies)
LABELS_OUTPUT_DIR = os.path.join("data", "raw", "flame3_labels")

FIREBALL_CLASS_ID = 0  # maps to 'fireball'
CONF_THRESHOLD = 0.25  # Lower threshold to catch more fire regions

# Fallback bounding box for known fire images where model detects nothing
USE_FALLBACK_BBOX = True
FALLBACK_BBOX = (0.5, 0.5, 0.8, 0.8)  # cx, cy, w, h (normalized)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}


def auto_annotate_fire_images(model):
    """Run YOLOv8 on fire images and generate label .txt files."""
    if not os.path.exists(FIRE_DIR):
        print(f"ERROR: Fire directory not found: {FIRE_DIR}")
        return 0, 0

    fire_images = sorted([f for f in os.listdir(FIRE_DIR)
                          if os.path.splitext(f)[1] in IMAGE_EXTENSIONS])

    if not fire_images:
        print("ERROR: No fire images found!")
        return 0, 0

    print(f"\nProcessing {len(fire_images)} fire images...")

    detected_count = 0
    fallback_count = 0

    for i, img_name in enumerate(fire_images):
        img_path = os.path.join(FIRE_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]

        # Run detection
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)

        label_lines = []

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            img_h, img_w = results[0].orig_shape

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = max(0, min(1, ((x1 + x2) / 2) / img_w))
                cy = max(0, min(1, ((y1 + y2) / 2) / img_h))
                w = max(0, min(1, (x2 - x1) / img_w))
                h = max(0, min(1, (y2 - y1) / img_h))
                label_lines.append(f"{FIREBALL_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            detected_count += 1
        elif USE_FALLBACK_BBOX:
            cx, cy, w, h = FALLBACK_BBOX
            label_lines.append(f"{FIREBALL_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            fallback_count += 1

        # Write label file to labels output directory
        out_lbl_path = os.path.join(LABELS_OUTPUT_DIR, "fire", base_name + '.txt')
        with open(out_lbl_path, 'w') as f:
            f.write('\n'.join(label_lines))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(fire_images)} images "
                  f"(detected: {detected_count}, fallback: {fallback_count})")

    print(f"\n  Fire images: {len(fire_images)}")
    print(f"  Model detected fire regions: {detected_count}")
    print(f"  Fallback bounding box used: {fallback_count}")

    return detected_count, fallback_count


def create_no_fire_labels():
    """Create empty label files for no-fire images (hard negatives)."""
    if not os.path.exists(NO_FIRE_DIR):
        print(f"WARNING: No Fire directory not found: {NO_FIRE_DIR}")
        return 0

    no_fire_images = sorted([f for f in os.listdir(NO_FIRE_DIR)
                             if os.path.splitext(f)[1] in IMAGE_EXTENSIONS])

    print(f"\nCreating {len(no_fire_images)} empty labels for no-fire images...")

    for img_name in no_fire_images:
        base_name = os.path.splitext(img_name)[0]
        out_lbl_path = os.path.join(LABELS_OUTPUT_DIR, "no_fire", base_name + '.txt')
        with open(out_lbl_path, 'w') as f:
            f.write('')  # Empty = background

    print(f"  No-fire labels created: {len(no_fire_images)}")
    return len(no_fire_images)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("FLAME 3 AUTO-ANNOTATION (labels only, no image copies)")
    print("=" * 60)

    if not os.path.exists(FLAME3_BASE):
        print(f"ERROR: FLAME 3 dataset not found at: {FLAME3_BASE}")
        sys.exit(1)

    # Create output directories
    os.makedirs(os.path.join(LABELS_OUTPUT_DIR, "fire"), exist_ok=True)
    os.makedirs(os.path.join(LABELS_OUTPUT_DIR, "no_fire"), exist_ok=True)

    # Load pre-trained YOLOv8
    print("\nLoading pre-trained YOLOv8s model...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO('yolov8s.pt')
    print("Model loaded successfully.")

    # Auto-annotate fire images
    detected, fallback = auto_annotate_fire_images(model)

    # Create empty labels for no-fire images
    no_fire_count = create_no_fire_labels()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = detected + fallback + no_fire_count
    print(f"Total labels created: {total}")
    print(f"  Fire (model detected):    {detected}")
    print(f"  Fire (fallback bbox):     {fallback}")
    print(f"  No-fire (empty labels):   {no_fire_count}")
    print(f"\nLabels saved to: {LABELS_OUTPUT_DIR}")
    print(f"  fire/    — {detected + fallback} label files")
    print(f"  no_fire/ — {no_fire_count} label files")
    print("=" * 60)
