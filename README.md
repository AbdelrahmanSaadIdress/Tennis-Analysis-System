# 🎾 Tennis Analysis System

**A computer vision system that analyzes tennis match footage — detecting players, tracking the ball, extracting court keypoints, and overlaying real-time performance statistics.**

</div>

---

## 📽️ Output Demo

> **Watch the system in action:**
https://github.com/user-attachments/assets/c9a440d8-7f21-472a-b8af-3278fd85a764
> *The output video shows players tracked with bounding boxes, the ball traced across frames, a live mini-court overlay, and an analysis panel displaying speed metrics — all rendered in real time.*

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Pipeline](#-system-pipeline)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Models](#-models)
- [Results & Metrics](#-results--metrics)

---

## 🧠 Overview

The **Tennis Analysis System** is an end-to-end computer vision pipeline designed to extract meaningful performance insights from tennis match videos. By combining state-of-the-art object detection, deep learning-based keypoint extraction, and statistical analysis, the system produces an annotated output video that empowers players and coaches with data-driven feedback.

The system measures:
- 🏃 **Player movement speed** across frames
- 🎾 **Ball shot speed** at the moment of impact
- 🔢 **Total number of shots** per player
- 📍 **Player positions** mapped on a mini-court overlay

---

## 🔄 System Pipeline

The following describes the complete processing pipeline — from raw input video to the final annotated output:

```
Input Video
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1 — Player Detection (YOLOv8)                    │
│  Detect all persons in each frame using a pretrained    │
│  YOLOv8 model. Filter detections to retain only the     │
│  two court players based on position and confidence.    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2 — Tennis Ball Detection (Fine-tuned YOLO)      │
│  A custom YOLO model fine-tuned on tennis footage       │
│  detects the ball across frames. Handles motion blur,   │
│  occlusion, and small object size challenges.           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3 — Ball Tracking & Interpolation                │
│  Track the ball trajectory across frames. Apply         │
│  interpolation to fill gaps caused by missed            │
│  detections (e.g., due to speed or occlusion).          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4 — Court Keypoint Extraction (ResNet50 CNN)     │
│  A custom CNN (based on ResNet50) is fine-tuned to      │
│  detect 14 court keypoints (lines, net, baselines).     │
│  These keypoints anchor all spatial transformations.    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 5 — Perspective Transformation & Mapping         │
│  Using the detected court keypoints, a homography       │
│  matrix maps real-world coordinates onto the            │
│  bird's-eye-view mini-court representation.             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 6 — Speed & Shot Analysis (Pandas)               │
│  Calculate player movement speed (m/s) and ball shot    │
│  speed using frame-to-frame displacement + FPS.         │
│  Detect shot moments using trajectory inflection.       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 7 — Visualization & Annotation (OpenCV)          │
│  Overlay bounding boxes, ball trail, mini-court,        │
│  and statistics panel onto each frame. Encode and       │
│  export the final annotated video.                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
                   Output Video
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎯 **Player Detection** | YOLOv8-powered detection with court-based filtering to isolate the two active players |
| 🟡 **Ball Tracking** | Fine-tuned YOLO model with trajectory interpolation for smooth, continuous tracking |
| 🗺️ **Court Keypoint Detection** | ResNet50 CNN trained on a custom dataset to precisely localize 14 court landmarks |
| 📐 **Mini-Court Overlay** | Real-time bird's-eye-view court projection showing accurate player & ball positions |
| ⚡ **Speed Analysis** | Frame-by-frame computation of player and ball speed with rolling averages |
| 🏸 **Shot Detection** | Automatic identification of shot events using ball trajectory inflection analysis |
| 📊 **Stats Dashboard** | On-screen HUD displaying player speed, ball speed, and shot count per player |

---

## 🏗️ Architecture

```
Tennis-Analysis-System/
│
├── Input Processing
│   └── VideoCapture via OpenCV — frame extraction at native FPS
│
├── Detection Models
│   ├── YOLOv8 (Ultralytics)  ──► Player bounding boxes
│   └── Fine-tuned YOLO       ──► Tennis ball localization
│
├── Deep Learning — Court Understanding
│   └── ResNet50 (PyTorch CNN) ──► 14 court keypoints regression
│
├── Geometry & Spatial Mapping
│   └── Homography Matrix     ──► Real-court ↔ Mini-court projection
│
├── Statistical Engine
│   └── Pandas DataFrames     ──► Speed, distance, shot count calculations
│
└── Output Rendering
    └── OpenCV VideoWriter    ──► Annotated video export
```

---

## 🛠️ Technologies Used

| Technology | Role |
|---|---|
| **YOLOv8** (Ultralytics) | Real-time player detection in each video frame |
| **Fine-tuned YOLO** | Custom-trained ball detector optimized for tennis footage |
| **PyTorch** | Deep learning framework for training and inference |
| **ResNet50 (CNN)** | Backbone for court keypoint regression — fine-tuned on custom annotated dataset |
| **OpenCV** | Frame reading, image annotation, video writing |
| **Pandas** | Structured data storage and computation for speed/shot statistics |
| **NumPy** | Numerical operations for homography and coordinate transforms |

---

## 📁 Project Structure

```
Tennis-Analysis-System/
│
├── input_videos/               # Raw tennis match videos
├── output_videos/              # Annotated output videos
├── models/                     # Trained model weights
│   ├── yolov8x.pt              # Pretrained YOLOv8 player detector
│   ├── ball_detector.pt        # Fine-tuned YOLO ball detection model
│   └── keypoints_model.pth     # Custom ResNet50 court keypoint model
│
├── trackers/
│   ├── player_tracker.py       # Player detection and frame tracking
│   └── ball_tracker.py         # Ball detection, tracking & interpolation
│
├── court_line_detector/
│   └── court_line_detector.py  # CNN-based keypoint extraction
│
├── mini_court/
│   └── mini_court.py           # Mini-court projection and rendering
│
├── utils/
│   ├── video_utils.py          # Frame reading/writing utilities
│   ├── player_stats_drawer.py  # Stats HUD overlay
│   ├── speed_distance_utils.py # Speed and distance calculations
│   └── bbox_utils.py           # Bounding box helpers
│
├── training/                   # Model training notebooks
│   ├── tennis_ball_detector_training.ipynb
│   └── tennis_court_keypoints_training.ipynb
│
├── constants.py                # Court dimensions and config constants
├── main.py                     # Entry point — runs the full pipeline
└── requirements.txt
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for real-time inference)
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/AbdelrahmanSaadIdress/Tennis-Analysis-System.git
cd Tennis-Analysis-System

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download pre-trained model weights and place them in /models
#    - ball_detector.pt  →  fine-tuned YOLO ball detection model
#    - keypoints_model.pth  →  ResNet50 court keypoint model
```

---

## 🚀 Usage

```bash
# Place your input video in the input_videos/ folder, then run:
python main.py
```

The pipeline will automatically:
1. Load the input video from `input_videos/`
2. Run all detection and analysis stages
3. Export the annotated video to `output_videos/`

> **Note:** The first run may take longer as models are loaded into memory. GPU inference is strongly recommended for smooth performance.

---

## 🤖 Models

### 1. Player Detector — YOLOv8
- Pretrained YOLOv8x from Ultralytics on COCO dataset
- Detects all persons in frame, then filters to the two court players using proximity-to-court logic

### 2. Ball Detector — Fine-tuned YOLO
- YOLO model fine-tuned on a custom tennis ball dataset
- Handles challenging scenarios: motion blur, small object size, high-speed trajectories
- Trajectory interpolation fills frames with missed detections

### 3. Court Keypoint Model — ResNet50 CNN (PyTorch)
- ResNet50 backbone with the final fully-connected layer replaced for keypoint regression
- Fine-tuned on a labeled dataset of tennis court images
- Outputs 14 (x, y) coordinate pairs representing court line intersections
- These keypoints drive the perspective transformation used for the mini-court

---

## 📈 Results & Metrics

The system produces the following real-time overlays on the output video:

| Metric | Description |
|---|---|
| 🏃 Player Speed | Current movement speed in km/h, updated per frame |
| ⚡ Avg Player Speed | Rolling average speed over the rally |
| 🎾 Ball Shot Speed | Speed of the ball at the moment of each shot (km/h) |
| 📊 Avg Ball Speed | Average ball speed across detected shots |
| 📍 Mini-Court | Live positional map of both players and ball |

---
