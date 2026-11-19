"""Video -> MediaPipe Holistic keypoints. Shared by offline extraction and the live demo
so that training and inference use exactly the same representation.
Output: array (MAX_FRAMES, 75, 3) = 33 pose + 21 left hand + 21 right hand landmarks (x, y, z).
Landmarks that are not detected are all-zero."""
import numpy as np

MAX_FRAMES = 64
STEP = 2


def make_holistic():
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=0,
                                          refine_face_landmarks=False)


def _pts(lm, n):
    if lm is None:
        return np.zeros((n, 3), np.float32)
    return np.array([[p.x, p.y, p.z] for p in lm.landmark], np.float32)


def video_to_keypoints(path, holistic, max_frames=MAX_FRAMES, step=STEP):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            r = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frames.append(np.concatenate([_pts(r.pose_landmarks, 33),
                                          _pts(r.left_hand_landmarks, 21),
                                          _pts(r.right_hand_landmarks, 21)]))
        i += 1
    cap.release()
    if not frames:
        return None
    arr = np.stack(frames)
    idx = np.linspace(0, len(arr) - 1, max_frames).astype(int)
    return arr[idx]
