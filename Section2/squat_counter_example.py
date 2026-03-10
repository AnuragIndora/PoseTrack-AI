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

model_path = "../pose_landmarker_full.task"

# Skeleton connections
POSE_CONNECTIONS = [
    (11,13),(13,15),
    (12,14),(14,16),
    (11,12),
    (11,23),(12,24),
    (23,24),
    (23,25),(25,27),
    (24,26),(26,28)
]

# Angle function
def calculate_angle(a,b,c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians * 180 / np.pi)

    if angle > 180:
        angle = 360-angle

    return angle


# Pose model options
options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
)

# Create detector
detector = PoseLandmarker.create_from_options(options)

# Webcam
cap = cv2.VideoCapture(0)

rep_counter = 0
stage = "UP"

frame_id = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    results = detector.detect_for_video(mp_image, frame_id)

    if results.pose_landmarks:

        landmarks = results.pose_landmarks[0]

        h,w,_ = frame.shape

        # Extract joints
        shoulder = [
            landmarks[11].x,
            landmarks[11].y
        ]

        hip = [
            landmarks[23].x,
            landmarks[23].y
        ]

        knee = [
            landmarks[25].x,
            landmarks[25].y
        ]

        ankle = [
            landmarks[27].x,
            landmarks[27].y
        ]

        # Angle calculation
        knee_angle = calculate_angle(hip,knee,ankle)
        hip_angle = calculate_angle(shoulder,hip,knee)

        # Squat logic
        if knee_angle > 160:
            stage = "UP"

        if knee_angle < 90 and stage == "UP":
            stage = "DOWN"
            rep_counter += 1

        # Draw joints
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(frame,(x,y),4,(0,255,0),-1)

        # Draw skeleton
        for connection in POSE_CONNECTIONS:

            p1 = connection[0]
            p2 = connection[1]

            x1 = int(landmarks[p1].x * w)
            y1 = int(landmarks[p1].y * h)

            x2 = int(landmarks[p2].x * w)
            y2 = int(landmarks[p2].y * h)

            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)

        # Draw angles
        kx = int(knee[0]*w)
        ky = int(knee[1]*h)

        cv2.putText(frame,
                    f"Knee:{int(knee_angle)}",
                    (kx,ky),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(0,255,255),2)

        hx = int(hip[0]*w)
        hy = int(hip[1]*h)

        cv2.putText(frame,
                    f"Hip:{int(hip_angle)}",
                    (hx,hy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(255,255,0),2)

    # UI
    cv2.rectangle(frame,(0,0),(250,120),(0,0,0),-1)

    cv2.putText(frame,
                f"REPS: {rep_counter}",
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(255,255,255),2)

    cv2.putText(frame,
                f"STAGE: {stage}",
                (10,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,(0,255,255),2)

    cv2.imshow("AI Squat Trainer",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()