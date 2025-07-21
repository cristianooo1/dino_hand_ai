import cv2
import numpy as np
import time
import pathlib
import math
from collections import deque

import mediapipe as mp  # type: ignore
from mediapipe.tasks.python import vision  # type: ignore
from mediapipe.tasks.python import BaseOptions  # type: ignore

# Configuration constants
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
TOUCH_THRESHOLD = 0.3
FAR_THRESHOLD = 0.5
NOISE_THRESHOLD = 0.05
STABILITY_FRAMES = 5
HISTORY_LENGTH = 10

# Drawing settings
MARGIN = 10
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
COLOR_TOUCH = (0, 0, 255)
COLOR_FAR = (255, 0, 255)
COLOR_NEUTRAL = (255, 255, 0)
COLOR_HAND = (88, 205, 54)

class GestureDetector:
    def __init__(self):
        self.history = deque(maxlen=HISTORY_LENGTH)
        self.current = "neutral"
        self.count = 0
        self.last_stable = 0.0

    def detect(self, distance: float):
        self.history.append(distance)
        avg = sum(self.history) / len(self.history)
        changed = abs(avg - self.last_stable) > NOISE_THRESHOLD

        if avg <= TOUCH_THRESHOLD:
            gesture = "touch"
        elif avg >= FAR_THRESHOLD:
            gesture = "far"
        else:
            gesture = "neutral"

        if gesture == self.current:
            self.count += 1
        else:
            self.current = gesture
            self.count = 1

        stable = self.count >= STABILITY_FRAMES
        if stable and changed:
            self.last_stable = avg
        return gesture, stable, changed, avg


def calculate_distance(world_landmarks):
    thumb = world_landmarks[THUMB_TIP]
    index = world_landmarks[INDEX_FINGER_TIP]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
    center = coords.mean(axis=0)

    rel_thumb = np.array([thumb.x, thumb.y, thumb.z]) - center
    rel_index = np.array([index.x, index.y, index.z]) - center
    dist = np.linalg.norm(rel_thumb - rel_index)

    wrist = world_landmarks[0]
    middle = world_landmarks[12]
    span = math.dist((wrist.x, wrist.y, wrist.z), (middle.x, middle.y, middle.z))
    norm = dist / span if span > 0 else 0
    return norm


def draw_hand(frame, result, detector: GestureDetector):
    h, w, _ = frame.shape
    for idx, (lm, wlm, handed) in enumerate(zip(
        result.hand_landmarks,
        result.hand_world_landmarks,
        result.handedness
    )):
        norm = calculate_distance(wlm)
        gesture, stable, changed, avg = detector.detect(norm)

        # Coordinates
        tx, ty = int(lm[THUMB_TIP].x * w), int(lm[THUMB_TIP].y * h)
        ix, iy = int(lm[INDEX_FINGER_TIP].x * w), int(lm[INDEX_FINGER_TIP].y * h)

        # Draw circles
        cv2.circle(frame, (tx, ty), 6, COLOR_NEUTRAL, -1)
        cv2.circle(frame, (ix, iy), 6, COLOR_NEUTRAL, -1)
        cv2.putText(frame, handed[0].category_name, (tx, ty - 20), FONT, FONT_SCALE, COLOR_HAND, FONT_THICKNESS)

        # Line color
        color = COLOR_TOUCH if gesture == "touch" else COLOR_FAR if gesture == "far" else COLOR_NEUTRAL
        thickness = 4 if stable else 2
        cv2.line(frame, (tx, ty), (ix, iy), color, thickness)

        # Labels
        status = "STABLE" if stable else "..."
        cv2.putText(frame, f"{gesture.upper()} {status}", (tx, ty - 40), FONT, FONT_SCALE, color, FONT_THICKNESS)
        cv2.putText(frame, f"{avg:.3f}", (tx, ty - 60), FONT, FONT_SCALE, color, FONT_THICKNESS)

        # Execute callback
        if stable and changed:
            if gesture == "touch":
                print("🤏 Touch action")
            elif gesture == "far":
                print("✋ Far action")
            else:
                print("👌 Neutral action")

    return frame


def main():
    model = pathlib.Path(__file__).with_name("hand_landmarker.task")
    
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        result_callback=lambda res, img, ts: setattr(main, 'result', res)
    )

    detector = GestureDetector()
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    start = time.monotonic()
    main.result = None
    print(f"MODELLLLLLLLLLLLLLLL: {model}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = int((time.monotonic() - start) * 1000)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mp_img, ts)

            if main.result:
                frame = draw_hand(frame, main.result, detector)

            cv2.imshow("Hand Gesture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
