"""
hand_tracking.py
Thin wrapper around MediaPipe's Hand Landmarker.

Recent mediapipe releases (0.10.3x+ / 1.0.x) removed the old
`mp.solutions.hands` API entirely -> `AttributeError: module
'mediapipe' has no attribute 'solutions'`. This is not something in
this project's control; Google dropped it. This file now uses
mediapipe's current Tasks API (`mediapipe.tasks.python.vision`)
instead, which is what's actually shipped in the mediapipe version
you have installed.

Everything downstream (gestures.py, smoothing.py, filters.py,
main.py, test_hand.py) is unaffected — HandTracker still exposes the
same process()/close() interface and the same
{"label": ..., "points": [...]} shape per hand.

Needs a model file, hand_landmarker.task, in this same folder. It's
downloaded automatically the first time you run the app (needs
internet once). If auto-download ever fails (e.g. a locked-down
network), download it yourself and place it next to this file:

    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Landmark indices we actually use elsewhere (of MediaPipe's 21 per hand)
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_TIP = 20

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_model():
    """Download the .task model file next to this script if it's not
    already there (or looks like a truncated/failed download)."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    print(f"[hand_tracking] Downloading hand landmark model to {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[hand_tracking] Model downloaded.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download {MODEL_URL} ({e}). Download it manually and "
            f"save it as: {MODEL_PATH}"
        ) from e


class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.6, tracking_confidence=0.5):
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_idx = 0  # VIDEO mode needs monotonically increasing timestamps

    def process(self, frame_bgr):
        """
        Returns a list of hands, each:
            {
                "label": "Left" | "Right",   # MediaPipe's handedness label
                "points": [(x, y), ...]      # 21 pixel-space (x, y) tuples
            }
        Note: MediaPipe's left/right labeling assumes a mirrored
        (selfie-style) input. We deliberately do NOT flip the frame
        (matches the reference video), so treat "Left"/"Right" here as
        just a consistent per-hand ID rather than anatomical truth.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_idx += 1
        result = self._landmarker.detect_for_video(mp_image, self._frame_idx)

        hands_out = []
        if result.hand_landmarks and result.handedness:
            for lm_set, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name  # "Left" | "Right"
                points = [(int(lm.x * w), int(lm.y * h)) for lm in lm_set]
                hands_out.append({"label": label, "points": points})
        return hands_out

    def close(self):
        self._landmarker.close()
