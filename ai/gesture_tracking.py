# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run gesture recognition with USB webcam under WSL2."""

import argparse
import sys
import time

import pathlib
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Globals for FPS calculation
COUNTER, FPS = 0, 0
START_TIME = time.monotonic()


def run(model: str,
        num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float,
        min_tracking_confidence: float,
        camera_id: int,
        width: int,
        height: int) -> None:
    """Continuously run inference on frames captured from the USB webcam."""
    # Open webcam via V4L2 backend (necessary under WSL2)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        sys.exit(f'ERROR: Cannot open camera with id={camera_id} via V4L2.')

    # Visualization params
    row_size = 50  # pixels for FPS text row
    left_margin = 24
    text_color = (0, 0, 0)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    label_text_color = (255, 255, 255)
    label_font_size = 1
    label_thickness = 2

    recognition_frame = None
    recognition_results = []

    def save_result(result: vision.GestureRecognizerResult,
                    unused_output_image: mp.Image,
                    timestamp_ms: int):
        """Callback for live-stream results from MediaPipe."""
        global COUNTER, FPS, START_TIME
        # Update FPS every `fps_avg_frame_count` frames
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.monotonic() - START_TIME)
            START_TIME = time.monotonic()

        recognition_results.append(result)
        COUNTER += 1

    # Initialize MediaPipe gesture recognizer
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=save_result
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # Main loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Check USB/IP setup.')

        # Mirror image for natural interaction
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Asynchronous recognition
        recognizer.recognize_async(mp_image, time.monotonic_ns() // 1_000_000)

        # Overlay FPS
        fps_text = f'FPS = {FPS:.1f}'
        cv2.putText(frame, fps_text, (left_margin, row_size),
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        # If we have a result, draw it
        if recognition_results:
            result = recognition_results.pop(0)
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                # Compute bounding box
                xs = [lm.x for lm in hand_landmarks]
                ys = [lm.y for lm in hand_landmarks]
                x_min_px = int(min(xs) * frame.shape[1])
                y_min_px = int(min(ys) * frame.shape[0])
                y_max_px = int(max(ys) * frame.shape[0])

                # Gesture label
                if result.gestures:
                    gesture = result.gestures[idx][0]
                    text = f'{gesture.category_name} ({gesture.score:.2f})'

                    (tw, th), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_DUPLEX,
                        label_font_size, label_thickness)
                    ty = y_min_px - 10
                    if ty < th:
                        ty = y_max_px + th + 10

                    cv2.putText(frame, text, (x_min_px, ty),
                                cv2.FONT_HERSHEY_DUPLEX,
                                label_font_size, label_text_color,
                                label_thickness, cv2.LINE_AA)

                # Draw landmarks
                proto = landmark_pb2.NormalizedLandmarkList()
                proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in hand_landmarks
                ])
                mp_drawing.draw_landmarks(
                    frame,
                    proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            recognition_frame = frame

        if recognition_frame is not None:
            cv2.imshow('gesture_recognition', recognition_frame)

        # Exit on ESC
        if cv2.waitKey(1) == 27:
            break

    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path to the .task model.',
        default= pathlib.Path(__file__).with_name("gesture_recognizer.task"))
    parser.add_argument(
        '--numHands',
        type=int,
        help='Max number of hands to detect.',
        default=1)
    parser.add_argument(
        '--minHandDetectionConfidence',
        type=float,
        help='Min confidence for hand detection.',
        default=0.5)
    parser.add_argument(
        '--minHandPresenceConfidence',
        type=float,
        help='Min confidence for hand presence.',
        default=0.5)
    parser.add_argument(
        '--minTrackingConfidence',
        type=float,
        help='Min confidence for hand tracking.',
        default=0.5)
    parser.add_argument(
        '--cameraId',
        type=int,
        help='VideoCapture camera ID (usually 0).',
        default=0)
    parser.add_argument(
        '--frameWidth',
        type=int,
        help='Width of captured frames.',
        default=640)
    parser.add_argument(
        '--frameHeight',
        type=int,
        help='Height of captured frames.',
        default=480)

    args = parser.parse_args()

    run(
        args.model,
        args.numHands,
        args.minHandDetectionConfidence,
        args.minHandPresenceConfidence,
        args.minTrackingConfidence,
        args.cameraId,
        args.frameWidth,
        args.frameHeight
    )


if __name__ == '__main__':
    main()
