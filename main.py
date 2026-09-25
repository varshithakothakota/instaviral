"""
main.py
Oriel - hand-controlled visual filter window.

Controls:
  q / ESC  - quit
  Two hands (index+thumb tips forming a quad) -> filter appears inside the quad
  One hand                                     -> filter masked to that hand's shape
  Hands together like prayer                   -> filter fills the whole frame
  Pinch (thumb+index touching) in two-hand mode -> cycle to the next filter
"""

import time
import cv2

from camera import ThreadedCamera
from hand_tracking import HandTracker
from smoothing import LandmarkSmoother
from gestures import (
    GestureState,
    build_quad,
    one_hand_hull,
    STATE_IDLE,
    STATE_ONE_HAND,
    STATE_TWO_HAND_QUAD,
    STATE_FULLSCREEN,
)
from filters import (
    FILTERS,
    apply_filter_in_quad,
    apply_filter_in_hull,
    apply_filter_fullscreen,
)


def main():
    cam = ThreadedCamera(src=0, width=1280, height=720).start()
    tracker = HandTracker(max_hands=2)
    smoother = LandmarkSmoother(alpha=0.4)
    gesture_state = GestureState()

    filter_idx = 0
    prev_time = time.time()
    fps = 0.0

    window_name = "Oriel"
    cv2.namedWindow(window_name)

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue

            hands = tracker.process(frame)
            hands = smoother.smooth(hands)
            state, cycle_filter = gesture_state.update(hands)

            if cycle_filter:
                filter_idx = (filter_idx + 1) % len(FILTERS)

            active_filter = FILTERS[filter_idx]
            output = frame

            # Guard against the confirmed `state` (which is debounced
            # over a few frames, see GestureState) briefly disagreeing
            # with the actual hand count this exact frame — e.g. state
            # is still TWO_HAND_QUAD for a couple more frames after a
            # hand drops out. Without this check hands[0]/build_quad
            # would index past the end of `hands` and crash.
            if state == STATE_TWO_HAND_QUAD and len(hands) == 2:
                quad = build_quad(hands)
                output = apply_filter_in_quad(frame, quad, active_filter)
            elif state == STATE_ONE_HAND and len(hands) >= 1:
                hull = one_hand_hull(hands[0])
                output = apply_filter_in_hull(frame, hull, active_filter)
            elif state == STATE_FULLSCREEN and len(hands) >= 1:
                output = apply_filter_fullscreen(frame, active_filter)
            # STATE_IDLE, or a state/hand-count mismatch this frame -> raw frame

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.putText(
                output, f"{state} | filter {filter_idx + 1}/{len(FILTERS)} | {fps:4.1f} FPS",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            cv2.imshow(window_name, output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cam.stop()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
