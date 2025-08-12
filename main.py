import threading
import pathlib
import time
import cv2
import pyautogui

from ai.hand_tracking import *

from server.dino_server import HTTPServer
from playwright.sync_api import sync_playwright


def main():

    GAME_DIR    = pathlib.Path(__file__).parent / "server" / "t-rex-runner"
    PORT        = 8000
    server      = HTTPServer(port =PORT, game_dir=GAME_DIR)

    server_thread = threading.Thread(target=server.start_http_server, daemon=True)
    server_thread.start()
    time.sleep(1)  

    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False, args=[
        "--window-size=1280,720",
        "--window-position=0,0"])
    context = browser.new_context(viewport={"width": 1280, "height": 720})  
    page = context.new_page()
    page.goto(f"http://localhost:{server.port}/server/t-rex-runner")
    print("🦖 Dino loaded in offline mode. Press 'q' (then Enter) to quit.")

    landmarker = define_model()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)  # Or cv2.WINDOW_AUTOSIZE
    cv2.moveWindow("Live Feed", 1920, 0)  # Position (X=200, Y=150) on screen


    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    start = time.monotonic()
    main.result = None

    left_active = False
    prev_left = False
    right_active = False
    prev_right = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = int((time.monotonic() - start) * 1000)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mp_img, ts)

            tip = get_index_tip_coords(landmarker.get_result(), frame.shape) 

            frame, left_active, right_active = draw_ui_sections(frame, tip, left_active, right_active)

            # if left_active:
            #     # finger is currently inside left box
            #     print("LEFT box is active (finger inside)")

            # if right_active:
            #     # finger is currently inside right box
            #     print("RIGHT box is active (finger inside)")

            # edge-triggered checks (enter / exit events)
            if left_active and not prev_left:
                print(">>> finger ENTERED LEFT box")
                pyautogui.press("space")

            if not left_active and prev_left:
                print(">>> finger LEFT LEFT box")
                # do one-time exit action here

            if right_active and not prev_right:
                print(">>> finger ENTERED RIGHT box")
                pyautogui.press("down")
                

            if not right_active and prev_right:
                print(">>> finger LEFT RIGHT box")

            # update previous state for the next frame
            prev_left = left_active
            prev_right = right_active

            if landmarker.get_result():
                frame = draw_hand_landmarks(frame, landmarker.get_result())

            cv2.imshow("Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        browser.close()
        playwright.stop()

if __name__ == "__main__":
    main()