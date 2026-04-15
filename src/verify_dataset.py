"""
Verify the merged YOLO dataset.
Checks image/label counts, class distribution, and file integrity.

Run: python src/verify_dataset.py
"""
import os
from collections import defaultdict
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
DATASET_DIR = os.path.join("data", "yolo_dataset")
CLASS_NAMES = {0: 'fireball', 1: 'flash', 2: 'smoke_plume', 3: 'debris_cloud'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def verify_split(split_name):
    """Verify a single split (train/val/test)."""
    img_dir = os.path.join(DATASET_DIR, split_name, 'images')
    lbl_dir = os.path.join(DATASET_DIR, split_name, 'labels')
    
    if not os.path.exists(img_dir):
        print(f"  ❌ {split_name}: images directory missing!")
        return None
    if not os.path.exists(lbl_dir):
        print(f"  ❌ {split_name}: labels directory missing!")
        return None
    
    # Count images
    images = [f for f in os.listdir(img_dir)
              if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    
    # Count labels
    labels = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
    
    # Check matching
    img_bases = {os.path.splitext(f)[0] for f in images}
    lbl_bases = {os.path.splitext(f)[0] for f in labels}
    
    missing_labels = img_bases - lbl_bases
    orphan_labels = lbl_bases - img_bases
    
    # Count class annotations
    class_counts = defaultdict(int)
    empty_labels = 0
    total_annotations = 0
    
    for lbl_file in labels:
        lbl_path = os.path.join(lbl_dir, lbl_file)
        has_content = False
        with open(lbl_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    parts = stripped.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        class_counts[cls_id] += 1
                        total_annotations += 1
                        has_content = True
        if not has_content:
            empty_labels += 1
    
    # Check for corrupt images (sample first 10)
    corrupt_count = 0
    for img_file in images[:10]:
        try:
            img = Image.open(os.path.join(img_dir, img_file))
            img.verify()
        except Exception as e:
            corrupt_count += 1
    
    # Report
    print(f"\n  📁 {split_name.upper()}")
    print(f"  {'─' * 40}")
    print(f"  Images:            {len(images)}")
    print(f"  Labels:            {len(labels)}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Background (empty): {empty_labels}")
    
    if missing_labels:
        print(f"  ⚠️  Missing labels: {len(missing_labels)}")
    if orphan_labels:
        print(f"  ⚠️  Orphan labels:  {len(orphan_labels)}")
    if corrupt_count > 0:
        print(f"  ⚠️  Corrupt images: {corrupt_count} (sampled 10)")
    
    print(f"\n  Class distribution:")
    for cls_id in sorted(CLASS_NAMES.keys()):
        count = class_counts.get(cls_id, 0)
        bar = '█' * min(50, count // max(1, total_annotations // 50 + 1))
        pct = (count / total_annotations * 100) if total_annotations > 0 else 0
        print(f"    {cls_id}: {CLASS_NAMES[cls_id]:15s} │ {count:7d} ({pct:5.1f}%) {bar}")
    
    return {
        'images': len(images),
        'labels': len(labels),
        'annotations': total_annotations,
        'background': empty_labels,
        'class_counts': dict(class_counts),
        'missing_labels': len(missing_labels),
        'orphan_labels': len(orphan_labels),
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    
    # Check data.yaml
    yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
    if os.path.exists(yaml_path):
        print(f"\n✅ data.yaml found")
        with open(yaml_path) as f:
            print(f.read())
    else:
        print(f"\n❌ data.yaml NOT found at {yaml_path}")
    
    # Verify each split
    total_images = 0
    total_annotations = 0
    total_class_counts = defaultdict(int)
    all_ok = True
    
    for split in ['train', 'val', 'test']:
        result = verify_split(split)
        if result:
            total_images += result['images']
            total_annotations += result['annotations']
            for cls_id, count in result['class_counts'].items():
                total_class_counts[cls_id] += count
            if result['missing_labels'] > 0 or result['orphan_labels'] > 0:
                all_ok = False
        else:
            all_ok = False
    
    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"  Total images:      {total_images}")
    print(f"  Total annotations: {total_annotations}")
    print(f"\n  Overall class distribution:")
    for cls_id in sorted(CLASS_NAMES.keys()):
        count = total_class_counts.get(cls_id, 0)
        pct = (count / total_annotations * 100) if total_annotations > 0 else 0
        status = "✅" if count > 0 else "⚠️  (no samples)"
        print(f"    {cls_id}: {CLASS_NAMES[cls_id]:15s} │ {count:7d} ({pct:5.1f}%) {status}")
    
    if all_ok:
        print(f"\n✅ Dataset looks good! Ready for training.")
    else:
        print(f"\n⚠️  Some issues found — review warnings above.")
    
    print("=" * 60)
