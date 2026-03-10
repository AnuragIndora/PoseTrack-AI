import cv2
import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks.python import vision
import numpy as np

BaseOptions = tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

model_path = "../pose_landmarker_full.task"

def calculate_angle(a,b,c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - \
              np.arctan2(a[1]-b[1],a[0]-b[0])

    angle = np.abs(radians * 180 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

detector = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

frame_id = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect_for_video(mp_image, frame_id)

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        h,w,_ = frame.shape

        shoulder = [landmarks[11].x, landmarks[11].y]
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]

        back_angle = calculate_angle(shoulder,hip,knee)

        if back_angle < 150:
            posture = "Bad Posture"
            color = (0,0,255)
        else:
            posture = "Good Posture"
            color = (0,255,0)

        cv2.putText(frame,
                    posture,
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,color,2)

        for lm in landmarks:

            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(frame,(x,y),4,(0,255,0),-1)

    cv2.imshow("Real-Time Posture Correction",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()