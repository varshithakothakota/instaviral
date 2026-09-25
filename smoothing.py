"""
smoothing.py
Exponential-moving-average smoothing of hand landmarks across frames.

Raw MediaPipe landmarks jitter a few pixels frame-to-frame even when a
hand is perfectly still. Since every hull/quad/pinch/prayer calculation
is redone from scratch each frame directly on those raw points, that
jitter shows up directly as visible shaking in the composited output.
This smooths each landmark point over time before anything else in the
pipeline touches it.

Usage (in main.py):

    smoother = LandmarkSmoother(alpha=0.4)
    ...
    hands = tracker.process(frame)
    hands = smoother.smooth(hands)
    state, cycle_filter = gesture_state.update(hands)
"""

import numpy as np


class LandmarkSmoother:
    def __init__(self, alpha=0.4):
        """
        alpha: weight given to the new (raw) frame each update, in
        (0, 1]. Lower = smoother but more lag; higher = snappier but
        jitterier. 0.35-0.5 is a reasonable starting range — retune by
        eye against your own camera.
        """
        self.alpha = alpha
        self._prev = {}  # hand label ("Left"/"Right") -> Nx2 float32 array

    def smooth(self, hands):
        smoothed = []
        seen_labels = set()

        for hand in hands:
            label = hand.get("label", "Unknown")
            seen_labels.add(label)

            pts = np.array(hand["points"], dtype=np.float32)

            prev = self._prev.get(label)
            if prev is not None and prev.shape == pts.shape:
                pts = self.alpha * pts + (1.0 - self.alpha) * prev

            self._prev[label] = pts

            new_hand = dict(hand)
            new_hand["points"] = [(float(x), float(y)) for x, y in pts]
            smoothed.append(new_hand)

        # Drop smoothing state for hands no longer in frame, so a hand
        # that leaves and re-enters doesn't snap-blend with stale data.
        for label in list(self._prev.keys()):
            if label not in seen_labels:
                del self._prev[label]

        return smoothed
