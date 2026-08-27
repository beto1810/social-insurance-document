import numpy as np
import cv2
from PIL import Image



def order_points(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 điểm theo thứ tự tl, tr, br, bl để warp đúng chiều."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def crop_quad(image: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Crop + warp 1 vùng tứ giác (poly 4 điểm) thành ảnh chữ nhật thẳng."""
    rect = order_points(poly.astype("float32"))
    (tl, tr, br, bl) = rect

    w_top = np.linalg.norm(tr - tl)
    w_bottom = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)

    max_w = max(int(w_top), int(w_bottom), 1)
    max_h = max(int(h_left), int(h_right), 1)

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_w, max_h))
    return warped

import cv2
import numpy as np


def suppress_red_stamp(
    image_bgr: np.ndarray,
) -> np.ndarray:
    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV,
    )

    h, s, v = cv2.split(hsv)

    # Red hue
    red_hue = (
        (h <= 12)
        | (h >= 168)
    )

    # Stamp normally has higher color saturation
    colorful = s > 45

    # Avoid treating very dark pixels as stamp,
    # because black text should be preserved.
    not_too_dark = v > 80

    red_mask = (
        red_hue
        & colorful
        & not_too_dark
    )

    result = image_bgr.copy()

    result[red_mask] = (
        255,
        255,
        255,
    )

    return result