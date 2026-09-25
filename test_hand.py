"""
test_hand.py
Debug view: draws landmarks and prints the pinch_ratio /
palm-distance numbers used by gestures.py, so you can calibrate
PINCH_RATIO_THRESHOLD and PRAYER_DIST_RATIO_THRESHOLD for your own
camera and lighting instead of trusting the defaults blindly.
"""

import cv2
from camera import ThreadedCamera
from hand_tracking import HandTracker
from gestures import pinch_ratio, is_prayer_pose


def main():
    cam = ThreadedCamera(src=0).start()
    tracker = HandTracker(max_hands=2)
    print("Press q to quit.")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue

            hands = tracker.process(frame)
            for hand in hands:
                for (x, y) in hand["points"]:
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                ratio = pinch_ratio(hand)
                x0, y0 = hand["points"][0]
                cv2.putText(frame, f"{hand['label']} pinch={ratio:.2f}",
                            (x0, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 0), 1)

            prayer = is_prayer_pose(hands)
            cv2.putText(frame, f"prayer={prayer}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("test_hand - landmark debug", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
