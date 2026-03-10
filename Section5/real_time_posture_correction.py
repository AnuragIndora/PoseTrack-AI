import cv2
import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks.python import vision
import numpy as np

# Mediapipe Tasks
BaseOptions = tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

# Load your model
model_path = "../pose_landmarker_full.task"  # Update path if needed

# Skeleton connections for drawing
POSE_CONNECTIONS = [
    (11,13),(13,15),
    (12,14),(14,16),
    (11,12),
    (11,23),(12,24),
    (23,24),
    (23,25),(25,27),
    (24,26),(26,28)
]

# Function to calculate angle between 3 points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# PoseLandmarker options
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = PoseLandmarker.create_from_options(options)

# Video capture
cap = cv2.VideoCapture(0)

rep_counter = 0
stage = "UP"
frame_id = 0

posture = "Correct Posture"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    h, w, _ = frame.shape

    # Convert to RGB for Mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Detect pose
    results = detector.detect_for_video(mp_image, frame_id)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks[0]

        # Convert landmarks to pixel coordinates
        def get_point(id):
            return [landmarks[id].x * w, landmarks[id].y * h]

        shoulder = get_point(11)
        hip = get_point(23)
        knee = get_point(25)
        ankle = get_point(27)

        # Calculate angles
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)
        back_angle = calculate_angle(shoulder, hip, knee)

        # -----------------------------
        # Squat Rep Logic (Correct)
        # -----------------------------
        if knee_angle < 90:
            stage = "DOWN"

        if knee_angle > 160 and stage == "DOWN":
            stage = "UP"
            rep_counter += 1

        # -----------------------------
        # Posture Logic (Torso check)
        # -----------------------------
        if back_angle < 150:
            posture = "Bad Posture"
            color = (0,0,255)
        else:
            posture = "Good Posture"
            color = (0,255,0)

        # -----------------------------
        # Feedback
        # -----------------------------
        feedback = ""

        if knee_angle > 170:
            feedback = "Start Squat"

        if 90 < knee_angle < 140:
            feedback = "Go Lower"

        if knee_angle < 70:
            feedback = "Too Low!"

        if posture == "Bad Posture":
            feedback = "Keep Back Straight!"

        # -----------------------------
        # Draw landmarks
        # -----------------------------
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0,255,0), -1)

        # Draw skeleton
        for p1, p2 in POSE_CONNECTIONS:
            x1, y1 = int(landmarks[p1].x * w), int(landmarks[p1].y * h)
            x2, y2 = int(landmarks[p2].x * w), int(landmarks[p2].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255,0,0), 2)

        # Draw angles
        cv2.putText(frame, f"Knee: {int(knee_angle)}",
                    (int(knee[0]), int(knee[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.putText(frame, f"Hip: {int(hip_angle)}",
                    (int(hip[0]), int(hip[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # Posture display
        cv2.putText(frame, posture, (10,140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Feedback display
        if feedback:
            cv2.putText(frame, feedback, (10,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        

    # Draw counter UI
    cv2.rectangle(frame,(0,0),(250,120),(0,0,0),-1)
    cv2.putText(frame,f"REPS: {rep_counter}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,f"STAGE: {stage}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    cv2.imshow("AI Squat Trainer", frame)

    # Stop at 5 reps
    if rep_counter >= 5:
        cv2.putText(frame, "Goal Reached!", (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        cv2.imshow("AI Squat Trainer", frame)
        cv2.waitKey(2000)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()