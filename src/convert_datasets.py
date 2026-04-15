"""
Convert and merge all datasets into unified YOLO format for blast detection.

KEY: Images are RESIZED to 640x640 during merge to save disk space.
     (YOLOv8 trains at 640x640 anyway, so no quality loss for training.)

Datasets:
  1. WA-YOLO (Explosion Detection) — already 640x640, 1 class → fireball
  2. DFire                          — resized, 2 classes → fireball + smoke_plume
  3. FLAME 3 (auto-annotated labels) — resized, fire images + no-fire negatives

Output: data/yolo_dataset/{train,val,test}/{images,labels}/

Run: python src/convert_datasets.py
"""
import os
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = os.path.join("data", "yolo_dataset")
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test
RANDOM_SEED = 42
TARGET_SIZE = 640  # Resize all images to 640x640 (saves disk space!)

CLASS_NAMES = {0: 'fireball', 1: 'flash', 2: 'smoke_plume', 3: 'debris_cloud'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def resize_and_save(img_path, output_path, size=TARGET_SIZE):
    """
    Resize image to size x size and save as compressed JPEG.
    Returns True if successful.
    """
    try:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        img.save(output_path, 'JPEG', quality=85)
        return True
    except Exception as e:
        print(f"  WARNING: Could not process {img_path}: {e}")
        return False


def remap_label_content(label_path, id_remap):
    """Read a YOLO label file and remap class IDs."""
    lines = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 5:
                    continue
                old_id = int(parts[0])
                if id_remap is not None:
                    if old_id in id_remap:
                        parts[0] = str(id_remap[old_id])
                    else:
                        continue
                lines.append(' '.join(parts))
    except Exception as e:
        print(f"  WARNING: Could not read {label_path}: {e}")
    return lines


def collect_standard_dataset(name, dirs, id_remap, prefix):
    """
    Collect pairs from a dataset with standard YOLO structure:
    {images_dir}/{image_file} + {labels_dir}/{label_file}
    """
    pairs = []
    counter = 0
    class_counts = defaultdict(int)
    bg_count = 0

    for dir_info in dirs:
        img_dir = dir_info['images']
        lbl_dir = dir_info['labels']

        if not os.path.exists(img_dir):
            print(f"  WARNING: {img_dir} not found, skipping")
            continue

        for img_file in sorted(os.listdir(img_dir)):
            ext = os.path.splitext(img_file)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            img_path = os.path.join(img_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            lbl_path = os.path.join(lbl_dir, base_name + '.txt')

            out_name = f"{prefix}_{counter:06d}"
            counter += 1

            if os.path.exists(lbl_path):
                label_lines = remap_label_content(lbl_path, id_remap)
                if label_lines:
                    for line in label_lines:
                        cls_id = int(line.split()[0])
                        class_counts[cls_id] += 1
                else:
                    bg_count += 1
            else:
                label_lines = []
                bg_count += 1

            pairs.append({
                'img_path': img_path,
                'label_lines': label_lines,
                'out_name': out_name,
            })

    annotated = len(pairs) - bg_count
    print(f"  {name}: {len(pairs)} images ({annotated} annotated, {bg_count} background)")
    for cls_id in sorted(class_counts.keys()):
        print(f"    Class {cls_id} ({CLASS_NAMES.get(cls_id, '?')}): {class_counts[cls_id]} annotations")

    return pairs


def collect_flame3_dataset(prefix='fl'):
    """
    Collect FLAME 3 pairs using original images + auto-generated labels.
    Fire images: originals in Fire/RGB/Raw, labels in flame3_labels/fire/
    No-fire images: originals in No Fire/RGB/Raw, labels in flame3_labels/no_fire/
    """
    flame3_base = os.path.join("data", "raw", "flame1", "FLAME 3 CV Dataset (Sycan Marsh)")
    labels_dir = os.path.join("data", "raw", "flame3_labels")

    fire_img_dir = os.path.join(flame3_base, "Fire", "RGB", "Raw")
    fire_lbl_dir = os.path.join(labels_dir, "fire")
    nofire_img_dir = os.path.join(flame3_base, "No Fire", "RGB", "Raw")
    nofire_lbl_dir = os.path.join(labels_dir, "no_fire")

    pairs = []
    counter = 0
    class_counts = defaultdict(int)
    fire_count = 0
    nofire_count = 0

    # Fire images
    if os.path.exists(fire_img_dir) and os.path.exists(fire_lbl_dir):
        for img_file in sorted(os.listdir(fire_img_dir)):
            ext = os.path.splitext(img_file)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            img_path = os.path.join(fire_img_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            lbl_path = os.path.join(fire_lbl_dir, base_name + '.txt')

            out_name = f"{prefix}_{counter:06d}"
            counter += 1

            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    label_lines = [l.strip() for l in f if l.strip()]
                for line in label_lines:
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] += 1
                fire_count += 1
            else:
                label_lines = []

            pairs.append({
                'img_path': img_path,
                'label_lines': label_lines,
                'out_name': out_name,
            })
    else:
        print(f"  WARNING: FLAME 3 fire data not found")

    # No-fire images (hard negatives)
    if os.path.exists(nofire_img_dir):
        for img_file in sorted(os.listdir(nofire_img_dir)):
            ext = os.path.splitext(img_file)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            img_path = os.path.join(nofire_img_dir, img_file)
            out_name = f"{prefix}_{counter:06d}"
            counter += 1

            pairs.append({
                'img_path': img_path,
                'label_lines': [],  # Empty = background
                'out_name': out_name,
            })
            nofire_count += 1
    
    print(f"  flame3: {len(pairs)} images ({fire_count} fire, {nofire_count} no-fire/background)")
    for cls_id in sorted(class_counts.keys()):
        print(f"    Class {cls_id} ({CLASS_NAMES.get(cls_id, '?')}): {class_counts[cls_id]} annotations")

    return pairs


def split_and_save(all_pairs, output_dir, ratios):
    """
    Shuffle, split, resize images to 640x640, and save in YOLO structure.
    """
    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)

    n = len(all_pairs)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    splits = {
        'train': all_pairs[:train_end],
        'val': all_pairs[train_end:val_end],
        'test': all_pairs[val_end:],
    }

    for split_name, pairs in splits.items():
        img_out = os.path.join(output_dir, split_name, 'images')
        lbl_out = os.path.join(output_dir, split_name, 'labels')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        success = 0
        fail = 0

        for j, pair in enumerate(pairs):
            out_name = pair['out_name']
            dst_img = os.path.join(img_out, out_name + '.jpg')
            dst_lbl = os.path.join(lbl_out, out_name + '.txt')

            # Resize and save image as compressed JPEG
            if resize_and_save(pair['img_path'], dst_img):
                # Write label file
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(pair['label_lines']))
                success += 1
            else:
                fail += 1

            if (j + 1) % 1000 == 0:
                print(f"    {split_name}: processed {j + 1}/{len(pairs)}...")

        # Stats
        split_classes = defaultdict(int)
        bg = 0
        for pair in pairs:
            if pair['label_lines']:
                for line in pair['label_lines']:
                    cls_id = int(line.split()[0])
                    split_classes[cls_id] += 1
            else:
                bg += 1

        print(f"  {split_name}: {success} images saved ({bg} background, {fail} failed)")
        for cls_id in sorted(split_classes.keys()):
            print(f"    Class {cls_id} ({CLASS_NAMES.get(cls_id, '?')}): {split_classes[cls_id]}")


def clean_output_splits(output_dir):
    """Clean existing split directories."""
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            try:
                shutil.rmtree(split_dir)
                print(f"  Cleaned: {split_dir}")
            except PermissionError:
                # On Windows, sometimes dirs are locked
                import subprocess
                subprocess.run(['cmd', '/c', 'rmdir', '/s', '/q', split_dir],
                               capture_output=True)
                print(f"  Force-cleaned: {split_dir}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("DATASET MERGE & CONVERSION")
    print(f"All images resized to {TARGET_SIZE}x{TARGET_SIZE} to save disk space")
    print("=" * 60)

    # Check FLAME 3 labels
    flame3_labels_dir = os.path.join("data", "raw", "flame3_labels")
    has_flame3 = os.path.exists(flame3_labels_dir)
    if not has_flame3:
        print("\nWARNING: FLAME 3 labels not found!")
        print("Run 'python src/auto_annotate_flame3.py' first to create them.")
        print("Continuing without FLAME 3...\n")

    # Clean output
    print("\nCleaning output directory...")
    clean_output_splits(OUTPUT_DIR)

    # Collect all pairs
    print("\nCollecting datasets:")
    all_pairs = []

    # 1. WA-YOLO
    wa_pairs = collect_standard_dataset(
        name='wa_yolo',
        dirs=[
            {'images': os.path.join("data", "raw", "wa_yolo", "data", "train", "images"),
             'labels': os.path.join("data", "raw", "wa_yolo", "data", "train", "labels")},
            {'images': os.path.join("data", "raw", "wa_yolo", "data", "valid", "images"),
             'labels': os.path.join("data", "raw", "wa_yolo", "data", "valid", "labels")},
            {'images': os.path.join("data", "raw", "wa_yolo", "data", "test", "images"),
             'labels': os.path.join("data", "raw", "wa_yolo", "data", "test", "labels")},
        ],
        id_remap={0: 0},  # Explosion → fireball
        prefix='wa',
    )
    all_pairs.extend(wa_pairs)

    # 2. DFire
    df_pairs = collect_standard_dataset(
        name='dfire',
        dirs=[
            {'images': os.path.join("data", "raw", "dfire", "data", "train", "images"),
             'labels': os.path.join("data", "raw", "dfire", "data", "train", "labels")},
            {'images': os.path.join("data", "raw", "dfire", "data", "test", "images"),
             'labels': os.path.join("data", "raw", "dfire", "data", "test", "labels")},
        ],
        id_remap={0: 0, 1: 2},  # fire→fireball, smoke→smoke_plume
        prefix='df',
    )
    all_pairs.extend(df_pairs)

    # 3. FLAME 3
    if has_flame3:
        fl_pairs = collect_flame3_dataset(prefix='fl')
        all_pairs.extend(fl_pairs)

    print(f"\nTotal dataset size: {len(all_pairs)} images")

    # Estimate output size
    # 640x640 JPEG at quality 85 ≈ ~50-80KB each
    est_size_mb = len(all_pairs) * 0.065  # ~65KB average
    print(f"Estimated output size: ~{est_size_mb:.0f} MB ({est_size_mb/1024:.1f} GB)")

    # Split and save
    print(f"\nResizing to {TARGET_SIZE}x{TARGET_SIZE} and splitting (70/15/15):")
    split_and_save(all_pairs, OUTPUT_DIR, SPLIT_RATIOS)

    # Update data.yaml
    abs_output = os.path.abspath(OUTPUT_DIR)
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    yaml_content = f"""# Blast Detection Dataset Configuration
# Auto-generated by convert_datasets.py
# Total images: {len(all_pairs)}
# Images resized to: {TARGET_SIZE}x{TARGET_SIZE}

path: {abs_output}
train: train/images
val: val/images
test: test/images

nc: 4
names:
  0: fireball
  1: flash
  2: smoke_plume
  3: debris_cloud
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nUpdated: {yaml_path}")

    # Final disk space check
    try:
        import psutil
        free_gb = psutil.disk_usage(abs_output).free / (1024**3)
        print(f"Remaining disk space: {free_gb:.2f} GB")
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print(f"✅ DATASET READY AT: {abs_output}")
    print("Next step: python src/verify_dataset.py")
    print("=" * 60)
