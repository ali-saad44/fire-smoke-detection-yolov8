# 🔥 Fire & Smoke Detection from Drone Video

AI-powered real-time fire and smoke detection system using **YOLOv8** trained on 22,000+ aerial images. Upload drone footage and get instant detection with bounding boxes and confidence scores.



## Overview

This system detects **fireballs** and **smoke plumes** from aerial drone footage using a fine-tuned YOLOv8s model. It includes a Flask-based web interface where users can upload videos or images and receive annotated results with detection statistics.

The detection pipeline:

```
Drone Video → YOLOv8 (per-frame detection) → Bounding Boxes + Confidence → Annotated Output
```

---

## Features

- **2-Class Detection** — Fireball and Smoke Plume detection from aerial perspectives
- **Video & Image Support** — Upload MP4, AVI, MOV, MKV, or JPG/PNG files
- **Real-Time Inference** — ~10ms per frame on GPU (~98 FPS)
- **Professional Web Interface** — Dark-themed, responsive design with drag-and-drop upload
- **Download Results** — Download annotated videos with detection overlays
- **Detection Statistics** — Frame count, detection rate, per-class breakdown

---

## Project Structure

```
projrct/
├── configs/                          # Model and pipeline configurations
│   ├── pipeline_config.yaml          # Full pipeline settings (tracker, GPS, logging)
│   └── train_config.yaml             # Training hyperparameters
│
├── data/                             # ⚠️ NOT included (see Data Preparation below)
│   ├── raw/                          # Raw downloaded datasets
│   │   ├── wa_yolo/                  # WA-YOLO dataset (images + labels)
│   │   ├── flame1/                   # FLAME 1 dataset (fire/no-fire images)
│   │   ├── dfire/                    # DFire dataset (images + YOLO labels)
│   │   └── hard_negatives/           # False alarm prevention images
│   │       ├── dust_clouds/
│   │       ├── sun_glare/
│   │       └── campfires/
│   └── yolo_dataset/                 # Merged & split training data
│       ├── train/images/ & labels/
│       ├── val/images/ & labels/
│       ├── test/images/ & labels/
│       └── data.yaml
│
├── models/
│   ├── yolov8_blast/
│   │   └── best.pt                   # ⚠️ Trained weights (download separately)
│   ├── exports/                      # ONNX exports
│   └── pretrained_weights/           # Base YOLOv8 weights
│
├── src/                              # Data preparation scripts
│   ├── convert_datasets.py           # Merge & split datasets into YOLO format
│   ├── auto_annotate_flame3.py       # Auto-annotate FLAME 1 classification images
│   ├── verify_dataset.py             # Verify dataset splits and class distribution
│   └── __init__.py
│
├── web/                              # Flask Web Application
│   ├── app.py                        # Main Flask server + YOLOv8 inference
│   ├── static/
│   │   ├── style.css                 # Premium dark-theme UI
│   │   ├── app.js                    # Frontend logic (upload, results display)
│   │   └── results/                  # Processed output files (auto-generated)
│   ├── templates/
│   │   └── index.html                # Main web page
│   └── uploads/                      # Temporary upload directory
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Datasets

Two public datasets were merged to create the training data:

| # | Dataset | Source | Content | Size |
|---|---------|--------|---------|------|
| 1 | **FLAME 3** | [IEEE Dataport](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones) | Aerial fire images (classification → auto-annotated to YOLO) | ~2 GB |
| 2 | **DFire** | [GitHub](https://github.com/gaiasd/DFireDataset) | Fire & smoke images with YOLO annotations | ~3 GB |

Additionally, **~9,900 hard negative images** (dust clouds, sun glare, campfires, clouds) were added as background images to reduce false alarms.

### Final Dataset Statistics

| Split | Images | Labels |
|-------|--------|--------|
| Train | 15,947 | 15,947 |
| Val | 3,417 | 3,417 |
| Test | 3,418 | 3,418 |
| **Total** | **22,782** | **22,782** |

### Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `fireball` | Fireballs, explosions, intense flames |
| 1 | `smoke_plume` | Rising smoke columns, haze, smoke plumes |

> **Note:** The original plan included 4 classes (fireball, flash, smoke_plume, debris_cloud), but `flash` and `debris_cloud` were removed due to insufficient training data. Labels were remapped to 2 classes for better performance.

---

## Data Preparation

Since the dataset is too large for GitHub, you need to prepare it locally:

### Step 1: Create the data directory

```bash
# Create required folders
mkdir -p data/raw/wa_yolo data/raw/flame1 data/raw/dfire
mkdir -p data/raw/hard_negatives/dust_clouds
mkdir -p data/raw/hard_negatives/sun_glare
mkdir -p data/raw/hard_negatives/campfires
mkdir -p data/yolo_dataset/train/images data/yolo_dataset/train/labels
mkdir -p data/yolo_dataset/val/images data/yolo_dataset/val/labels
mkdir -p data/yolo_dataset/test/images data/yolo_dataset/test/labels
```

### Step 2: Download datasets

1. **WA-YOLO** — Download from [https://osf.io/H34Q6](https://osf.io/H34Q6) → Extract to `data/raw/wa_yolo/`
2. **FLAME 1** — Download from [IEEE Dataport](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones) (free account required) → Extract to `data/raw/flame1/`
3. **DFire** — Clone or download from [GitHub](https://github.com/gaiasd/DFireDataset) → Extract to `data/raw/dfire/`

### Step 3: Merge and split

```bash
# Auto-annotate FLAME 1 images
python src/auto_annotate_flame3.py

# Merge all datasets and create train/val/test splits
python src/convert_datasets.py

# Verify the final dataset
python src/verify_dataset.py
```

### Step 4: Create data.yaml

Create `data/yolo_dataset/data.yaml`:

```yaml
path: ./data/yolo_dataset
train: train/images
val: val/images
test: test/images

nc: 2
names:
  0: fireball
  1: smoke_plume
```

---

## Model Training (Kaggle)

Training was performed on **Kaggle** using a **Tesla T4 GPU** with the following two-phase strategy:

### Phase A — Frozen Backbone (50 epochs)
- Freeze first 10 layers (backbone)
- Learning rate: `0.001`
- Optimizer: AdamW
- Purpose: Train the detection head while preserving pretrained features

### Phase B — Full Fine-Tuning (67 epochs, early stopped)
- Unfreeze all layers
- Learning rate: `0.0003` (lower to prevent catastrophic forgetting)
- Patience: 20 epochs (early stopping triggered at epoch 67)
- Purpose: Fine-tune the entire model for blast-specific features

### Kaggle Setup Notes

1. Upload the merged `yolo_dataset/` folder as a Kaggle Dataset
2. Use a **GPU T4 x2** accelerator
3. Apply the PyTorch compatibility patch for `torch.load`:
   ```python
   import torch
   _original = torch.load
   def _patched(*args, **kwargs):
       kwargs['weights_only'] = False
       return _original(*args, **kwargs)
   torch.load = _patched
   ```
4. Install latest ultralytics: `!pip install ultralytics --upgrade -q`
5. Fix `data.yaml` paths to point to Kaggle working directory (`/kaggle/working/yolo_dataset`)
6. Total training time: **~8.5 hours** (Phase A: 2.9h + Phase B: 5.7h)

---

## Training Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| **mAP@50** | 0.784 |
| **mAP@50:95** | 0.469 |
| **Precision** | 0.797 |
| **Recall** | 0.716 |

### Per-Class Results

| Class | Precision | Recall | mAP@50 |
|-------|-----------|--------|--------|
| Fireball | 0.827 | 0.779 | 0.833 |
| Smoke Plume | 0.767 | 0.653 | 0.735 |




### Setup

```bash

# Install dependencies
pip install -r requirements.txt
```

### Model Weights

Download the trained `best.pt` weights and place them at:

```
models/yolov8_blast/best.pt
```

> The weights file is ~22.5 MB. It is excluded from this repository via `.gitignore` due to size.

---


### Optional: Install FFmpeg

For in-browser video playback, install [FFmpeg](https://ffmpeg.org/download.html) and add it to your system PATH. Without FFmpeg, you can still download and play processed videos with VLC.

---




