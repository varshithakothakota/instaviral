"""
filters.py
The four visual styles cycled through via pinch, plus the geometry
helpers that let any filter be rendered inside a warped quad
(two-hand mode) or a hand-shaped mask (one-hand mode).

Pipeline for two-hand quad mode (matches the tilted-grid evidence
seen in the reference video, where the filter's cell grid rotated
WITH the hand quad rather than staying screen-aligned):

    source frame --warp(quad -> rect)--> canonical rect
                                              |
                                        apply filter()
                                              |
    dest frame  <--warp(rect -> quad)-- filtered rect  (masked composite)

CANONICAL_SIZE is a tunable default (arbitrary — not measured from
the source).
"""

import cv2
import numpy as np

CANONICAL_SIZE = (400, 300)  # (width, height)


# ---------------------------------------------------------------------
# Filters. Each takes a BGR uint8 image and returns a BGR uint8 image
# of the same size. All parameters here (cell size, levels, etc.) are
# guesses tuned to visually resemble the reference video, not measured
# from it.
# ---------------------------------------------------------------------

def cartoon_ghibli(bgr, num_down=2, num_bilateral=6):
    """
    Colourful painterly/cartoon look (replaces the old red duotone).

    Uses the classic "downsample -> repeated small bilateral filters ->
    upsample" cartoonify pipeline rather than cv2.stylization(), which
    benchmarked at ~0.4 FPS on a 1280x720 frame (unusably slow — this
    alone was a big part of the "shaking"). This pipeline benchmarks at
    ~20+ FPS at full frame and 100+ FPS on the smaller crops used by
    quad/hand-hull mode.
    """
    color = bgr
    for _ in range(num_down):
        color = cv2.pyrDown(color)
    for _ in range(num_bilateral):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=9, sigmaSpace=7)
    for _ in range(num_down):
        color = cv2.pyrUp(color)
    color = cv2.resize(color, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    # push saturation for a more vivid, painted-anime feel
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # crisp dark outlines on top, like ink linework
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=9, C=4,
    )
    edges_3c = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges_3c)


def mosaic_gray(bgr, cell=12):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    out = np.zeros_like(gray)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            block = gray[y:y + cell, x:x + cell]
            if block.size == 0:
                continue
            brightness = block.mean()
            # quantize to a few levels to get the blocky poster look
            level = int(brightness // 64) * 64
            out[y:y + cell, x:x + cell] = level
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def neon_posterize(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_HSV)


def bw_posterize(bgr, levels=4):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    step = 255 // (levels - 1)
    quantized = (gray // step) * step
    return cv2.cvtColor(quantized.astype(np.uint8), cv2.COLOR_GRAY2BGR)


FILTERS = [cartoon_ghibli, mosaic_gray, neon_posterize, bw_posterize]


# ---------------------------------------------------------------------
# Geometry: applying a filter inside a quad or a hull mask
# ---------------------------------------------------------------------

def _feathered_composite(base, overlay, mask_u8, feather=12):
    """
    Blend `overlay` over `base` using mask_u8 (0/255), with the mask
    edge Gaussian-blurred first. A hard binary mask (the old
    np.where(...)) hard-cuts every pixel right at the hull/quad
    boundary, so a 1px landmark jitter frame-to-frame makes that whole
    boundary ring flicker on/off — this is a big part of what reads as
    "shaking". Feathering the edge turns that into a soft, stable
    gradient instead.
    """
    mask_blur = cv2.GaussianBlur(mask_u8, (0, 0), sigmaX=feather)
    alpha = (mask_blur.astype(np.float32) / 255.0)[:, :, None]
    blended = overlay.astype(np.float32) * alpha + base.astype(np.float32) * (1 - alpha)
    return blended.astype(np.uint8)


def apply_filter_in_quad(frame, quad_pts, filter_fn):
    """
    quad_pts: float32 array of 4 points [TL, TR, BR, BL] in the source
    frame's pixel coordinates.
    Returns `frame` with the filtered content composited inside the quad.
    """
    w, h = CANONICAL_SIZE
    dst_rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # source quad -> canonical rectangle
    m_fwd = cv2.getPerspectiveTransform(quad_pts, dst_rect)
    canonical = cv2.warpPerspective(frame, m_fwd, (w, h))

    filtered_canonical = filter_fn(canonical)

    # canonical rectangle -> back into the quad's position in the frame
    m_inv = cv2.getPerspectiveTransform(dst_rect, quad_pts)
    warped_back = cv2.warpPerspective(
        filtered_canonical, m_inv, (frame.shape[1], frame.shape[0])
    )

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad_pts.astype(np.int32), 255)

    return _feathered_composite(frame, warped_back, mask)


def apply_filter_in_hull(frame, hull_pts, filter_fn):
    """
    hull_pts: Nx1x2 int32 array from cv2.convexHull (one hand's outline).
    Filters the whole frame, then only keeps that filtered result
    inside the hull.
    """
    filtered_full = filter_fn(frame)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull_pts, 255)

    return _feathered_composite(frame, filtered_full, mask)


def apply_filter_fullscreen(frame, filter_fn):
    return filter_fn(frame)
