from server.dino_server import HTTPServer
from playwright.sync_api import sync_playwright
import time
import threading
import pathlib

GAME_DIR = pathlib.Path(__file__).parent / "t-rex-runner"

PORT     = 8000
server = HTTPServer(port =PORT, game_dir=GAME_DIR)

def main():
    # 1) Start HTTP server in background
    server_thread = threading.Thread(target=server.start_http_server, daemon=True)
    server_thread.start()
    time.sleep(1)  # give it a moment to bind

    # 2) Launch Playwright and open Dino
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()  
        page = context.new_page()
        page.goto(f"http://localhost:{server.port}/server/t-rex-runner")
        print("🦖 Dino loaded in offline mode. Press 'q' (then Enter) to quit.")

        # 3) Wait for user signal to quit
        while True:
            if input().strip().lower() == "q":
                break

        # 4) Clean up
        browser.close()
        print("🦖 Browser closed. Exiting…")

if __name__ == "__main__":
    main()
