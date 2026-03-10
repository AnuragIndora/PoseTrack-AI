# PoseTrack AI

**PoseTrack AI** is a real-time human pose detection and posture correction system built using Python, OpenCV, MediaPipe, and NumPy.  
It tracks body landmarks, counts squats or hand raises, detects bad posture, and provides visual feedback to help users improve fitness and alignment.

---

## Project Overview
This project demonstrates practical applications of **Computer Vision** and **AI/ML** using Python.  

It includes two parts:  
- **Section 2:** Practice coding questions covering OpenCV webcam usage, face detection, and MediaPipe pose landmark detection.  
- **Section 5:** Full assignment implementation – a real-time posture correction system with squat/hand-raise counting and feedback.

---

## Technologies Used
- **Python 3.10+**  
- **OpenCV** – Real-time video capture and visualization  
- **MediaPipe** – Pose detection and body landmark extraction  
- **NumPy** – Angle calculation and mathematical computations  

---

## Section 2 – Practice / Coding Questions
This folder contains Python scripts for practice questions:  

1. `webcam_opencv.py` – Open webcam using OpenCV.  
2. `face_detection_opencv.py` – Detect faces using OpenCV.  
3. `pose_landmarks_mediapipe.py` – Detect body landmarks using MediaPipe.  
4. `squat_counter_example.py` – Example logic to count squats.  
5. `posture_correction_example.py` – Sample code to analyze posture and show feedback.

> These scripts are smaller, modular examples to understand OpenCV and MediaPipe functionalities.

---

## Section 5 – Assignment Task
This section contains the **full real-time posture detection system**, combining pose detection, squat/hand-raise counting, and feedback visualization.  

- File: `real_time_posture_correction.py`  
- Features:
  - Detect human body landmarks using MediaPipe Pose Landmarker.  
  - Track squats or hand raises and count repetitions.  
  - Detect and display correct/incorrect posture in real-time.  
  - Draw landmarks and skeleton connections on video feed.

---

## Setup & Usage

1. **Clone the repository:**
```bash
git clone https://github.com/AnuragIndora/PoseTrack-AI.git
cd PoseTrack-AI
````

2. **Install dependencies:**

```bash
pip install opencv-python mediapipe numpy
```

3. **Run Section 2 scripts (practice questions):**

```bash
python webcam_opencv.py
python face_detection_opencv.py
python pose_landmarks_mediapipe.py
```

4. **Run Section 5 assignment (full project):**

```bash
python real_time_posture_correction.py
```

> Make sure your webcam is connected and accessible.

---

## Features

* Real-time pose detection and visualization
* Squat and hand-raise counting
* Posture analysis with real-time feedback
* Skeleton and landmark display
* Modular scripts for practice and full assignment

---

## Applications

* Home fitness coaching and exercise monitoring
* Yoga posture and physiotherapy monitoring
* Sports training and movement analysis


This project is for **educational and internship submission purposes**.
Feel free to use and modify it for learning or personal projects.

```

