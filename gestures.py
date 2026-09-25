"""
gestures.py
Turns raw hand landmarks into the app's states:

  0 hands  -> IDLE          (no effect)
  1 hand   -> ONE_HAND       (effect masked to that hand's convex hull)
  2 hands  -> TWO_HAND_QUAD  (effect inside quad: L.index, R.index, R.thumb, L.thumb)
  prayer   -> FULLSCREEN     (both hands together, vertical) -> effect fills the frame
  pinch    -> triggers a debounced "cycle to next filter" event

All thresholds below are TUNABLE DEFAULTS, not measured from any
source video — recalibrate against your own camera/lighting by
printing `pinch_ratio` and `palm_distance_ratio` and watching the
values as you form each gesture.
"""

import time
import numpy as np
from hand_tracking import WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP

PINCH_RATIO_THRESHOLD = 0.35       # pinch_dist / hand_scale below this = "pinched"
PRAYER_DIST_RATIO_THRESHOLD = 0.6  # palm-to-palm dist / hand_scale below this = "together"
PINCH_DEBOUNCE_SECONDS = 0.6

STATE_IDLE = "IDLE"
STATE_ONE_HAND = "ONE_HAND"
STATE_TWO_HAND_QUAD = "TWO_HAND_QUAD"
STATE_FULLSCREEN = "FULLSCREEN"


STATE_CONFIRM_FRAMES = 3  # raw state must repeat this many frames running before the app switches to it


def _dist(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def hand_scale(hand):
    """Rough reference size of a hand in pixels, used to normalize
    other distances so thresholds stay stable as the hand moves nearer/
    farther from the camera."""
    return _dist(hand["points"][WRIST], hand["points"][MIDDLE_MCP]) + 1e-6


def pinch_ratio(hand):
    """thumb-tip to index-tip distance, normalized by hand_scale.
    Small value = pinched."""
    d = _dist(hand["points"][THUMB_TIP], hand["points"][INDEX_TIP])
    return d / hand_scale(hand)


def is_pinching(hand):
    return pinch_ratio(hand) < PINCH_RATIO_THRESHOLD


def palm_center(hand):
    pts = np.array(hand["points"])
    return tuple(pts.mean(axis=0))


def is_prayer_pose(hands):
    """Both hands present, palms close together."""
    if len(hands) != 2:
        return False
    c0, c1 = palm_center(hands[0]), palm_center(hands[1])
    scale = (hand_scale(hands[0]) + hand_scale(hands[1])) / 2.0
    return _dist(c0, c1) / scale < PRAYER_DIST_RATIO_THRESHOLD


def build_quad(hands):
    """
    Build the ordered 4-point quad from two hands' thumb/index tips.
    Order returned: [top_left, top_right, bottom_right, bottom_left]
    matching cv2.getPerspectiveTransform's expected dst ordering.

    We use handedness label to decide which hand is "left"/"right" in
    the frame rather than sorting by raw x, so the quad doesn't flip
    inside-out if hands cross.
    """
    by_label = {h["label"]: h for h in hands}
    if "Left" not in by_label or "Right" not in by_label:
        # Two hands with the same label reading (tracking noise) -
        # fall back to sorting by x position.
        h_sorted = sorted(hands, key=lambda h: h["points"][WRIST][0])
        left, right = h_sorted[0], h_sorted[1]
    else:
        left, right = by_label["Left"], by_label["Right"]

    top_left = left["points"][INDEX_TIP]
    bottom_left = left["points"][THUMB_TIP]
    top_right = right["points"][INDEX_TIP]
    bottom_right = right["points"][THUMB_TIP]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def one_hand_hull(hand):
    """Convex hull of all 21 landmarks -> polygon for masking."""
    pts = np.array(hand["points"], dtype=np.int32)
    return __import__("cv2").convexHull(pts)


class GestureState:
    """Owns the debounce timer for pinch-triggered filter cycling and
    exposes the current high-level app state each frame.

    Also debounces the state itself: a single noisy frame (MediaPipe
    briefly drops a hand, or the prayer-distance check flickers True/
    False right at the threshold) used to switch the rendered state
    immediately, which is a second, separate cause of the visual
    "shaking" beyond raw landmark jitter. Now a candidate state has to
    repeat for STATE_CONFIRM_FRAMES in a row before it's adopted.
    """

    def __init__(self):
        self._last_pinch_time = 0.0
        self._confirmed_state = STATE_IDLE
        self._candidate_state = STATE_IDLE
        self._candidate_count = 0

    def _raw_state(self, hands):
        if len(hands) == 0:
            return STATE_IDLE
        elif is_prayer_pose(hands):
            return STATE_FULLSCREEN
        elif len(hands) == 1:
            return STATE_ONE_HAND
        else:
            return STATE_TWO_HAND_QUAD

    def update(self, hands):
        now = time.time()
        cycle_filter = False

        raw_state = self._raw_state(hands)

        if raw_state == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = raw_state
            self._candidate_count = 1

        if self._candidate_count >= STATE_CONFIRM_FRAMES:
            self._confirmed_state = raw_state

        state = self._confirmed_state

        # Pinch cycling only makes sense in the two-hand quad state, and
        # only when we actually have 2 hands *this frame* (the confirmed
        # state can lag raw_state by up to STATE_CONFIRM_FRAMES frames).
        if state == STATE_TWO_HAND_QUAD and len(hands) == 2:
            any_pinching = any(is_pinching(h) for h in hands)
            if any_pinching and (now - self._last_pinch_time) > PINCH_DEBOUNCE_SECONDS:
                cycle_filter = True
                self._last_pinch_time = now

        return state, cycle_filter
