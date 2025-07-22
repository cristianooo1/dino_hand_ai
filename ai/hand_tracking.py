import cv2
import numpy as np
import time
import pathlib
import pyautogui

import mediapipe as mp  # type: ignore
from mediapipe.tasks.python import vision  # type: ignore
from mediapipe.tasks.python import BaseOptions  # type: ignore

# def draw_frame_sections(frame):
#     h, w, _ = frame.shape
#     section_height = h // 3

#     # Draw horizontal lines to show sections
#     cv2.line(frame, (0, section_height), (w, section_height), (255, 255, 255), 2)
#     cv2.line(frame, (0, section_height * 2), (w, section_height * 2), (255, 255, 255), 2)
    
#     # Add section labels
#     cv2.putText(frame, "TOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     cv2.putText(frame, "MIDDLE", (10, section_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     cv2.putText(frame, "BOTTOM", (10, section_height * 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
#     return frame

def draw_frame_sections(frame):
    h, w, _ = frame.shape
    section_width = w // 2

    # Draw horizontal lines to show sections
    cv2.line(frame, (section_width-30, 0), (section_width-30, h), (255, 255, 255), 2)
    cv2.line(frame, (section_width+30, 0), (section_width +30, h), (255, 255, 255), 2)
    
    # Add section labels
    cv2.putText(frame, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "CENTER", (section_width -60, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "RIGHT", (section_width + 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


def get_finger_section(x_coord, frame_width):
    """Determine which section the finger tip is in"""
    section_width = frame_width // 2
    
    if x_coord < (section_width-30):
        return "LEFT"
    elif x_coord < (section_width +30):
        return "CENTER"
    else:
        return "RIGHT"

def draw_hand_landmarks(frame, result):
    h, w, _ = frame.shape
    INDEX_FINGER_TIP = 8
    
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for i, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Highlight index finger tip
                if i == INDEX_FINGER_TIP:
                    cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Red circle for index finger tip
                    
                    # Determine and display section
                    section = get_finger_section(x, w)
                    cv2.putText(frame, f"Index: {section}", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    move_mouse(x, y)
                    
                    

    return frame

def move_mouse(xc, yc):
    # pyautogui.moveTo(xc, yc, duration=1)
    print(pyautogui.size())

def main():
    model = pathlib.Path(__file__).with_name("hand_landmarker.task")
    
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        result_callback=lambda res, img, ts: setattr(main, 'result', res)
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    start = time.monotonic()
    main.result = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = int((time.monotonic() - start) * 1000)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mp_img, ts)

            frame = draw_frame_sections(frame)

            if main.result:
                frame = draw_hand_landmarks(frame, main.result)

            cv2.imshow("Hand Gesture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()