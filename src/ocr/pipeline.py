from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from src.ocr.exporter import build_labelme_json
from src.ocr.orientation import DocumentOrientationCorrector
from src.ocr.preprocessing import suppress_red_stamp
from src.ocr.recognizer import recognize_texts


ORIENTATION_CONF_THRESHOLD = 0.8


def _load_oriented_image(
    image_path: Path,
    orient_corrector: DocumentOrientationCorrector,
    enable_orientation: bool,
) -> tuple[Image.Image, int, float]:
    """Load one RGB image and apply trusted document rotation."""
    if not enable_orientation:
        return Image.open(image_path).convert("RGB"), 0, 0.0

    corrected_image, angle, confidence = orient_corrector.correct(image_path)
    if angle == 0 or confidence < ORIENTATION_CONF_THRESHOLD:
        corrected_image.close()
        return Image.open(image_path).convert("RGB"), 0, confidence

    print(
        f"Orientation corrected: {angle}° "
        f"(confidence={confidence:.2f})"
    )
    return corrected_image, angle, confidence


def _filter_polygons(
    polygons,
    scores,
    score_thresh: float,
) -> list:
    """Keep detection polygons whose confidence meets the threshold."""
    return [
        polygon
        for polygon, score in zip(polygons, scores)
        if score >= score_thresh
    ]


def _print_timings(
    timings: dict[str, float],
    detected_count: int,
    filtered_count: int,
) -> None:
    """Print the stable OCR profiling summary used by the CLI."""
    print("\n" + "=" * 60)
    print("OCR PERFORMANCE")
    print("=" * 60)
    for label, key in (
        ("Orientation", "orientation"),
        ("Load image", "load_image"),
        ("Stamp suppression", "stamp_suppression"),
        ("Detection", "detection"),
        ("Filter", "filter"),
        ("Recognition", "recognition"),
        ("Build JSON", "build_json"),
    ):
        print(f"{label:<18}: {timings[key]:.3f}s")
    print("-" * 60)
    print(f"{'TOTAL':<18}: {timings['total']:.3f}s")
    print(f"Detected regions  : {detected_count}")
    print(f"Filtered regions  : {filtered_count}")
    print("=" * 60)


def process_image(
    img_path: str | Path,
    detector,
    recognizer,
    orient_corrector,
    score_thresh: float = 0.3,
    recognition_batch_size: int = 32,
    enable_orientation: bool = True,
) -> dict:

    total_start = time.perf_counter()

    timings = {}

    img_path = Path(img_path)
    corrected_image = None

    try:
        # ====================================================
        # 0. Orientation
        # ====================================================

        t = time.perf_counter()

        corrected_image, angle, confidence = _load_oriented_image(
            image_path=img_path,
            orient_corrector=orient_corrector,
            enable_orientation=enable_orientation,
        )

        timings["orientation"] = (
            time.perf_counter() - t
        )

        # ====================================================
        # 1. Load image
        # ====================================================

        t = time.perf_counter()

        image_bgr = cv2.cvtColor(
            np.ascontiguousarray(corrected_image),
            cv2.COLOR_RGB2BGR,
        )

        timings["load_image"] = (
            time.perf_counter() - t
        )

        # ====================================================
        # 2. Stamp suppression
        # ====================================================

        t = time.perf_counter()

        detection_bgr = suppress_red_stamp(
            image_bgr
        )

        timings["stamp_suppression"] = (
            time.perf_counter() - t
        )

        # ====================================================
        # 3. Text detection
        # ====================================================

        t = time.perf_counter()

        det_result = detector.predict(
            detection_bgr
        )[0]

        dt_polys = det_result["dt_polys"]

        dt_scores = det_result.get(
            "dt_scores",
            [1.0] * len(dt_polys),
        )

        timings["detection"] = (
            time.perf_counter() - t
        )

        # ====================================================
        # 4. Filter
        # ====================================================

        t = time.perf_counter()

        filtered_polys = _filter_polygons(
            polygons=dt_polys,
            scores=dt_scores,
            score_thresh=score_thresh,
        )

        timings["filter"] = (
            time.perf_counter() - t
        )

        # ====================================================
        # 5. VietOCR
        # ====================================================

        t = time.perf_counter()

        texts, text_confidences = recognize_texts(
            image_bgr,
            filtered_polys,
            recognizer,
            batch_size=recognition_batch_size,
        )

        timings["recognition"] = (
            time.perf_counter() - t
        )

        # ====================================================
        # 6. Build JSON
        # ====================================================

        t = time.perf_counter()

        height, width = image_bgr.shape[:2]

        data = build_labelme_json(
            img_path,
            filtered_polys,
            texts,
            confidences=text_confidences,
            image_size=(width, height),
        )

        data["imageWidth"] = width
        data["imageHeight"] = height

        data["orientation"] = {
            "angle": angle,
            "confidence": confidence,
        }

        timings["build_json"] = (
            time.perf_counter() - t
        )

        # ====================================================
        # TOTAL
        # ====================================================

        timings["total"] = (
            time.perf_counter()
            - total_start
        )

        # ====================================================
        # PRINT PROFILING
        # ====================================================

        _print_timings(
            timings=timings,
            detected_count=len(dt_polys),
            filtered_count=len(filtered_polys),
        )

        return {
            "document": data,
            "image_bgr": image_bgr,
            "timings": timings,
        }

    finally:
        if corrected_image is not None:
            corrected_image.close()