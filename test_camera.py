"""
test_camera.py
Sanity check: does the threaded camera open and stream frames?
Run this FIRST when setting up on a new machine, before touching
hand tracking or filters.
"""

import cv2
from camera import ThreadedCamera


def main():
    cam = ThreadedCamera(src=0).start()
    print("Press q to quit.")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue
            cv2.imshow("test_camera - raw feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
