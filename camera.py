"""
camera.py
Threaded webcam capture.

Why threaded: a plain `cv2.VideoCapture.read()` call blocks the calling
thread until the next frame is ready. If your main loop does
read() -> mediapipe inference -> filter -> render, all in series, the
capture wait and the inference cost stack on top of each other and you
get low FPS (this is the "12 FPS" stage described in the reference post).

Running capture on its own thread and always keeping only the newest
frame decouples "waiting for the camera" from "processing a frame", so
the main loop only ever processes the most recent frame available.
"""

import threading
import time
import cv2


class ThreadedCamera:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source {src}")

        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread = None

        # Prime the buffer with one synchronous read so a frame is
        # available immediately after construction.
        ok, frame = self.cap.read()
        if ok:
            self._frame = frame

    def start(self):
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        """Returns (ok, latest_frame). Non-blocking."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()
