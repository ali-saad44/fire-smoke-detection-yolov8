"""
Blast Detection Web Demo — Flask Application
Upload video or images and detect fireball/smoke using YOLOv8.
"""
import os
import sys
import uuid
import time
import json
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'blast-detection-2026'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload

ALLOWED_IMAGE_EXT = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
ALLOWED_VIDEO_EXT = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'yolov8_blast', 'best.pt')
CONFIDENCE_THRESHOLD = 0.45

# Class colors (BGR for OpenCV)
CLASS_COLORS = {
    0: (0, 69, 255),    # fireball — orange-red
    1: (180, 180, 180), # smoke_plume — gray
}
CLASS_NAMES = {0: 'Smoke', 1: 'Fire'}

# ============================================================
# LOAD MODEL
# ============================================================
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# ============================================================
# HELPERS
# ============================================================
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXT


def is_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXT


def draw_detections(frame, results):
    """Draw bounding boxes and labels on frame."""
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS.get(cls_id, (0, 255, 0))
            label = CLASS_NAMES.get(cls_id, f'class_{cls_id}')
            label_text = f"{label} {conf:.0%}"

            # Draw filled rectangle behind text
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Draw label background
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            detections.append({
                'class': label,
                'confidence': round(conf, 3),
                'bbox': [x1, y1, x2, y2]
            })

    return frame, detections


def process_image(filepath, result_id):
    """Process a single image and return results."""
    frame = cv2.imread(filepath)
    if frame is None:
        return None, []

    results = model.predict(frame, imgsz=640, conf=CONFIDENCE_THRESHOLD, verbose=False)
    annotated, detections = draw_detections(frame.copy(), results)

    # Save result
    result_filename = f"{result_id}.jpg"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    cv2.imwrite(result_path, annotated)

    return result_filename, detections


def process_video(filepath, result_id):
    """Process video frame by frame and return annotated video + stats."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return None, {}

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video
    result_filename = f"{result_id}.mp4"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0
    class_stats = {name: 0 for name in CLASS_NAMES.values()}
    all_detections = []
    frames_with_detection = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model.predict(frame, imgsz=640, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated, detections = draw_detections(frame.copy(), results)

        # Add frame counter overlay
        cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if detections:
            frames_with_detection += 1

        for d in detections:
            total_detections += 1
            class_stats[d['class']] = class_stats.get(d['class'], 0) + 1

        out.write(annotated)

    cap.release()
    out.release()

    # Convert mp4v to browser-compatible h264
    h264_filename = f"{result_id}_h264.mp4"
    h264_path = os.path.join(app.config['RESULTS_FOLDER'], h264_filename)

    # Try ffmpeg conversion for browser compatibility
    ffmpeg_cmd = f'ffmpeg -i "{result_path}" -vcodec libx264 -acodec aac -y "{h264_path}" -loglevel quiet'
    ret_code = os.system(ffmpeg_cmd)

    if ret_code == 0 and os.path.exists(h264_path):
        os.remove(result_path)
        result_filename = h264_filename
    # If ffmpeg fails, keep the mp4v version

    stats = {
        'total_frames': frame_count,
        'frames_with_detection': frames_with_detection,
        'total_detections': total_detections,
        'class_stats': class_stats,
        'fps': fps,
        'resolution': f"{width}x{height}",
        'detection_rate': round(frames_with_detection / max(frame_count, 1) * 100, 1)
    }

    return result_filename, stats


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Use: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Save uploaded file
    result_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{result_id}_{filename}")
    file.save(filepath)

    start_time = time.time()

    try:
        if is_image(filename):
            result_filename, detections = process_image(filepath, result_id)
            processing_time = round(time.time() - start_time, 2)

            if result_filename is None:
                return jsonify({'error': 'Failed to process image'}), 500

            return jsonify({
                'type': 'image',
                'result_url': url_for('static', filename=f'results/{result_filename}'),
                'detections': detections,
                'total_detections': len(detections),
                'processing_time': processing_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        elif is_video(filename):
            result_filename, stats = process_video(filepath, result_id)
            processing_time = round(time.time() - start_time, 2)

            if result_filename is None:
                return jsonify({'error': 'Failed to process video'}), 500

            return jsonify({
                'type': 'video',
                'result_url': url_for('static', filename=f'results/{result_filename}'),
                'stats': stats,
                'processing_time': processing_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/model-info')
def model_info():
    """Return model metadata."""
    return jsonify({
        'model_name': 'YOLOv8s — Blast Detection',
        'classes': list(CLASS_NAMES.values()),
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'input_size': 640,
        'metrics': {
            'mAP@50': 0.784,
            'precision': 0.797,
            'recall': 0.716,
            'fireball_mAP': 0.833,
            'smoke_mAP': 0.735
        }
    })


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("BLAST DETECTION WEB DEMO")
    print("=" * 50)
    print(f"Model: {MODEL_PATH}")
    print(f"Open: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
