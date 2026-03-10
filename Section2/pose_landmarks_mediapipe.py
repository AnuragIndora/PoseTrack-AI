import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

# Path to your downloaded model
model_path = "../pose_landmarker_full.task"

# Connections between landmarks (like in old mp_pose.POSE_CONNECTIONS)
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11),
    (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (15, 17), (16, 18),
    (17, 19), (18, 20), (19, 21), (20, 22),
    (11, 23), (12, 24), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29),
    (28, 30), (29, 31), (30, 32)
]

# Initialize Pose Landmarker
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb_frame))

    # Get timestamp in ms
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    # Pose detection
    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.pose_landmarks:
        for pose in result.pose_landmarks:
            # Draw landmarks
            landmark_points = []
            for lm in pose:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                landmark_points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw skeleton lines
            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                    cv2.line(frame, landmark_points[start_idx], landmark_points[end_idx], (0, 0, 255), 2)

    cv2.imshow("Pose Detection with Skeleton", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()