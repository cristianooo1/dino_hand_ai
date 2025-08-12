import cv2
import numpy as np
import time
import pathlib
import pyautogui
from types import SimpleNamespace

import mediapipe as mp 
from mediapipe.tasks.python import vision 
from mediapipe.tasks.python import BaseOptions  

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

def draw_ui_sections(frame, tip, left_active, right_active):
    """
    Draw two rectangular "buttons" (left and right). If `tip` is inside a rect,
    that rect becomes green while the tip remains inside. Otherwise red.
    Returns (frame, left_active, right_active).
    """
    h, w, _ = frame.shape

    # sizing/placement (adjust to taste)
    margin_x = 100
    box_w = int(w * 0.25)      # width of each big square
    box_h = int(h * 0.5)       # height
    top_y = (h - box_h) // 2

    # left rectangle coords
    lx1, ly1 = margin_x, top_y
    lx2, ly2 = margin_x + box_w, top_y + box_h

    # right rectangle coords
    rx1, ry1 = w - margin_x - box_w, top_y
    rx2, ry2 = w - margin_x, top_y + box_h

    # check if tip is inside
    if tip is not None:
        tx, ty = tip
        left_hit = (lx1 <= tx <= lx2) and (ly1 <= ty <= ly2)
        right_hit = (rx1 <= tx <= rx2) and (ry1 <= ty <= ry2)
    else:
        left_hit = False
        right_hit = False

    # update active states (True while inside, False when outside)
    left_active = left_hit
    right_active = right_hit

    # colors BGR
    green = (0, 255, 0)
    red = (0, 0, 255)
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Outer filled rect (red or green)
    outer_left_color = green if left_active else red
    outer_right_color = green if right_active else red
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), outer_left_color, thickness=-1)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), outer_right_color, thickness=-1)

    # inner white rectangle (margin inside the colored border) - emulate the image style
    inner_margin = 20
    cv2.rectangle(frame, (lx1 + inner_margin, ly1 + inner_margin),
                         (lx2 - inner_margin, ly2 - inner_margin), white, thickness=-1)
    cv2.rectangle(frame, (rx1 + inner_margin, ry1 + inner_margin),
                         (rx2 - inner_margin, ry2 - inner_margin), white, thickness=-1)

    # black border for clarity
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), black, thickness=3)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), black, thickness=3)

    # Labels
    cv2.putText(frame, "POS1", (lx1 + box_w//6, ly1 + box_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, "POS2", (rx1 + box_w//6, ry1 + box_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return frame, left_active, right_active


# def get_finger_section(x_coord, frame_width):
#     """Determine which section the finger tip is in"""
#     section_width = frame_width // 2
    
#     if x_coord < (section_width-30):
#         return "LEFT"
#     elif x_coord < (section_width +30):
#         return "CENTER"
#     else:
#         return "RIGHT"

def get_index_tip_coords(result, frame_shape):
    """Return (x,y) pixel coords of index fingertip or None."""
    if not result or not result.hand_landmarks:
        return None
    h, w = frame_shape[:2]
    # use first detected hand only
    hand_landmarks = result.hand_landmarks[0]
    INDEX_FINGER_TIP = 8
    try:
        lm = hand_landmarks[INDEX_FINGER_TIP]
        x = int(lm.x * w)
        y = int(lm.y * h)
        return (x, y)
    except Exception:
        return None

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
                    cv2.putText(frame, f"Index", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame


def define_model():
    model = pathlib.Path(__file__).with_name("hand_landmarker.task")

    storage = SimpleNamespace(result = None)

    def result_callback(res, img, ts):
        storage.result = res
    
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        result_callback=result_callback
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    landmarker._result_storage = storage
    landmarker.get_result = lambda: landmarker._result_storage.result

    return landmarker

