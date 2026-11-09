# tests/test_recognizer_batch_size.py

import json
import math
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# Adjust these imports to your project
from src.ocr.detector import load_detector
from src.ocr.recognizer import load_recognizer
from src.ocr.preprocessing import crop_quad


IMAGE_PATH = "/Users/dat/Documents/AI-document-understanding/data/document-kie/image/1_PDFsam_Om dau 2_page_001.png"

BATCH_SIZES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
]

# Number of repeated benchmark runs per batch size.
#
# First run is discarded because it can contain:
# - MPS warm-up
# - memory allocation
# - kernel compilation
# - internal caching
REPEATS = 5

# Set True if you want detailed logs for every internal batch.
VERBOSE_BATCH = True

# Randomize benchmark order to reduce bias from:
# - model warming
# - MPS caching
# - thermal effects
RANDOMIZE_ORDER = True

RANDOM_SEED = 42


# ============================================================
# DETECTOR
# ============================================================

def extract_polygons_from_detector_result(result):
    """
    Convert Paddle/PaddleX detector output into a list of polygons.

    Expected output:

    [
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        ...
    ]

    Adjust this function if your detector returns a slightly
    different structure.
    """

    polygons = []

    for item in result:

        # --------------------------------------------------------
        # Case 1:
        # PaddleX result object with `.json`
        # --------------------------------------------------------

        if hasattr(item, "json"):

            data = item.json

            if callable(data):
                data = data()

            if isinstance(data, str):
                data = json.loads(data)

            if isinstance(data, dict):

                res = data.get(
                    "res",
                    data,
                )

                found = res.get(
                    "dt_polys",
                    res.get(
                        "polys",
                        [],
                    ),
                )

                if found is not None:
                    polygons.extend(found)

                continue

        # --------------------------------------------------------
        # Case 2:
        # result itself is a dictionary
        # --------------------------------------------------------

        if isinstance(item, dict):

            res = item.get(
                "res",
                item,
            )

            found = res.get(
                "dt_polys",
                res.get(
                    "polys",
                    [],
                ),
            )

            if found is not None:
                polygons.extend(found)

    return polygons


def detect_text_regions(
    image_bgr: np.ndarray,
    detector,
):
    """
    Run text detection once on the full document image.
    """

    print()
    print("=" * 90)
    print("DETECTION")
    print("=" * 90)

    start = time.perf_counter()

    result = detector.predict(
        image_bgr
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    polygons = (
        extract_polygons_from_detector_result(
            result
        )
    )

    print(
        f"Detected regions : {len(polygons)}"
    )

    print(
        f"Detection time   : {elapsed:.3f}s"
    )

    return polygons, elapsed


# ============================================================
# CROPPING
# ============================================================

def prepare_crops(
    image_bgr: np.ndarray,
    polygons,
):
    """
    Crop all detected text regions once.

    These exact same crops are reused for every batch-size test.
    """

    print()
    print("=" * 90)
    print("PREPARE CROPS")
    print("=" * 90)

    crops = []

    skipped = 0

    start_time = time.perf_counter()

    for polygon in polygons:

        polygon = np.asarray(
            polygon,
            dtype=np.float32,
        )

        crop = crop_quad(
            image_bgr,
            polygon,
        )

        if (
            crop is None
            or crop.size == 0
            or crop.shape[0] < 2
            or crop.shape[1] < 2
        ):
            skipped += 1
            continue

        crop_rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB,
        )

        crop_pil = Image.fromarray(
            crop_rgb
        )

        crops.append(
            crop_pil
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Detected polygons : {len(polygons)}"
    )

    print(
        f"Valid crops       : {len(crops)}"
    )

    print(
        f"Skipped crops     : {skipped}"
    )

    print(
        f"Cropping time     : {elapsed:.3f}s"
    )

    return crops


# ============================================================
# SINGLE RECOGNITION RUN
# ============================================================

def run_recognition_once(
    crops,
    recognizer,
    batch_size: int,
    verbose: bool = False,
):
    """
    Run recognition across all crops for one configured
    batch size.

    Returns:
        predictions
        elapsed_time
        actual_batch_sizes
    """

    predictions = []

    actual_batch_sizes = []

    num_calls = math.ceil(
        len(crops) / batch_size
    )

    total_start = time.perf_counter()

    for batch_index, start in enumerate(
        range(
            0,
            len(crops),
            batch_size,
        ),
        start=1,
    ):

        batch = crops[
            start:start + batch_size
        ]

        actual_batch_sizes.append(
            len(batch)
        )

        batch_start = time.perf_counter()

        # ========================================================
        # THIS IS THE IMPORTANT PART
        #
        # One predict_batch call receives the entire batch.
        # ========================================================

        batch_predictions = (
            recognizer.predict_batch(
                batch
            )
        )

        batch_elapsed = (
            time.perf_counter()
            - batch_start
        )

        predictions.extend(
            batch_predictions
        )

        if verbose:

            per_image = (
                batch_elapsed / len(batch)
            )

            print(
                f"      "
                f"Batch "
                f"{batch_index:02d}/"
                f"{num_calls:02d}"
                f" | actual={len(batch):3d}"
                f" | time={batch_elapsed:7.3f}s"
                f" | per_image={per_image:7.4f}s"
            )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    return (
        predictions,
        total_elapsed,
        actual_batch_sizes,
    )


# ============================================================
# BENCHMARK ONE BATCH SIZE
# ============================================================

def benchmark_batch_size(
    crops,
    recognizer,
    batch_size: int,
    repeats: int = 5,
):
    """
    Benchmark one configured batch size multiple times.

    First run is discarded from the median.
    """

    print()
    print("-" * 90)

    print(
        f"Configured batch size: {batch_size}"
    )

    expected_calls = math.ceil(
        len(crops) / batch_size
    )

    print(
        f"Total crops           : {len(crops)}"
    )

    print(
        f"Recognizer calls      : {expected_calls}"
    )

    times = []

    all_predictions = None

    actual_batch_sizes = None

    for repeat_index in range(
        repeats
    ):

        print(
            f"\n    Run "
            f"{repeat_index + 1}/{repeats}"
        )

        (
            predictions,
            elapsed,
            current_batch_sizes,
        ) = run_recognition_once(
            crops=crops,
            recognizer=recognizer,
            batch_size=batch_size,
            verbose=VERBOSE_BATCH,
        )

        times.append(
            elapsed
        )

        all_predictions = (
            predictions
        )

        actual_batch_sizes = (
            current_batch_sizes
        )

        print(
            f"    Total recognition time: "
            f"{elapsed:.3f}s"
        )

    # ------------------------------------------------------------
    # Discard first timing
    # ------------------------------------------------------------

    if len(times) > 1:
        stable_times = times[1:]
    else:
        stable_times = times

    median_time = statistics.median(
        stable_times
    )

    mean_time = statistics.mean(
        stable_times
    )

    min_time = min(
        stable_times
    )

    max_time = max(
        stable_times
    )

    ms_per_crop = (
        median_time
        / len(crops)
        * 1000
    )

    crops_per_second = (
        len(crops)
        / median_time
    )

    print()
    print(
        f"    Actual batches     : "
        f"{actual_batch_sizes}"
    )

    print(
        f"    All runs           : "
        f"{[round(x, 3) for x in times]}"
    )

    print(
        f"    Stable runs        : "
        f"{[round(x, 3) for x in stable_times]}"
    )

    print(
        f"    Median             : "
        f"{median_time:.3f}s"
    )

    return {
        "batch_size": batch_size,
        "num_calls": expected_calls,
        "actual_batch_sizes": actual_batch_sizes,
        "all_times": times,
        "stable_times": stable_times,
        "median_time": median_time,
        "mean_time": mean_time,
        "min_time": min_time,
        "max_time": max_time,
        "ms_per_crop": ms_per_crop,
        "crops_per_second": crops_per_second,
        "predictions": all_predictions,
    }


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_summary(
    results,
):
    print()
    print()
    print("=" * 110)
    print("FINAL SUMMARY - SORTED BY MEDIAN RECOGNITION TIME")
    print("=" * 110)

    print(
        f"{'Batch':>8}"
        f"{'Calls':>8}"
        f"{'Actual batches':>22}"
        f"{'Median':>12}"
        f"{'Mean':>12}"
        f"{'Min':>12}"
        f"{'Max':>12}"
        f"{'ms/crop':>12}"
        f"{'crop/s':>10}"
    )

    print("-" * 110)

    sorted_results = sorted(
        results,
        key=lambda x: x["median_time"],
    )

    for result in sorted_results:

        actual = ",".join(
            str(x)
            for x
            in result["actual_batch_sizes"]
        )

        print(
            f"{result['batch_size']:>8}"
            f"{result['num_calls']:>8}"
            f"{actual:>22}"
            f"{result['median_time']:>11.3f}s"
            f"{result['mean_time']:>11.3f}s"
            f"{result['min_time']:>11.3f}s"
            f"{result['max_time']:>11.3f}s"
            f"{result['ms_per_crop']:>12.2f}"
            f"{result['crops_per_second']:>10.2f}"
        )

    print("=" * 110)

    fastest = sorted_results[0]

    print()
    print(
        f"FASTEST CONFIGURED BATCH SIZE : "
        f"{fastest['batch_size']}"
    )

    print(
        f"ACTUAL BATCHES                : "
        f"{fastest['actual_batch_sizes']}"
    )

    print(
        f"MEDIAN RECOGNITION TIME       : "
        f"{fastest['median_time']:.3f}s"
    )

    print(
        f"MEDIAN TIME PER CROP          : "
        f"{fastest['ms_per_crop']:.2f} ms"
    )

    print(
        f"THROUGHPUT                    : "
        f"{fastest['crops_per_second']:.2f} crops/sec"
    )

    print("=" * 110)

    return fastest


# ============================================================
# CHECK PREDICTION CONSISTENCY
# ============================================================

def compare_predictions(
    results,
):
    """
    Optional sanity check.

    Different batch sizes should normally return the same text.
    """

    print()
    print("=" * 90)
    print("PREDICTION CONSISTENCY CHECK")
    print("=" * 90)

    if not results:
        return

    reference = results[0]

    reference_predictions = (
        reference["predictions"]
    )

    reference_batch = (
        reference["batch_size"]
    )

    for result in results[1:]:

        predictions = (
            result["predictions"]
        )

        different = []

        for i, (
            ref_pred,
            current_pred,
        ) in enumerate(
            zip(
                reference_predictions,
                predictions,
            )
        ):

            if ref_pred != current_pred:

                different.append(
                    (
                        i,
                        ref_pred,
                        current_pred,
                    )
                )

        print(
            f"Batch "
            f"{reference_batch}"
            f" vs "
            f"{result['batch_size']}: "
            f"{len(different)} differences"
        )

        if different:

            for (
                index,
                ref_text,
                current_text,
            ) in different[:5]:

                print(
                    f"    Crop {index:02d}"
                )

                print(
                    f"        "
                    f"batch {reference_batch}: "
                    f"{ref_text}"
                )

                print(
                    f"        "
                    f"batch {result['batch_size']}: "
                    f"{current_text}"
                )


# ============================================================
# PRINT SAMPLE OCR RESULTS
# ============================================================

def print_sample_predictions(
    fastest,
    limit=15,
):
    print()
    print("=" * 90)
    print("SAMPLE OCR PREDICTIONS")
    print("=" * 90)

    predictions = fastest[
        "predictions"
    ]

    for index, prediction in enumerate(
        predictions[:limit],
        start=1,
    ):

        print(
            f"{index:02d}. {prediction}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print("VIETOCR BATCH SIZE BENCHMARK")
    print("=" * 90)

    print(
        f"Image        : {IMAGE_PATH}"
    )

    print(
        f"Repeats      : {REPEATS}"
    )

    print(
        f"Batch sizes  : {BATCH_SIZES}"
    )

    # ========================================================
    # 1. LOAD IMAGE
    # ========================================================

    image_path = Path(
        IMAGE_PATH
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image does not exist: "
            f"{image_path}"
        )

    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:

        raise RuntimeError(
            f"Could not read image: "
            f"{image_path}"
        )

    height, width = (
        image_bgr.shape[:2]
    )

    print(
        f"Image size   : "
        f"{width} x {height}"
    )

    # ========================================================
    # 2. LOAD MODELS ONCE
    # ========================================================

    print()
    print("=" * 90)
    print("LOAD MODELS")
    print("=" * 90)

    start = time.perf_counter()

    detector = load_detector()

    detector_load_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Detector loaded in "
        f"{detector_load_time:.3f}s"
    )

    start = time.perf_counter()

    recognizer = (
        load_recognizer()
    )

    recognizer_load_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Recognizer loaded in "
        f"{recognizer_load_time:.3f}s"
    )

    # ========================================================
    # 3. DETECTION - RUN ONLY ONCE
    # ========================================================

    polygons, detection_time = (
        detect_text_regions(
            image_bgr=image_bgr,
            detector=detector,
        )
    )

    if not polygons:

        raise RuntimeError(
            "Detector returned no text regions."
        )

    # ========================================================
    # 4. CROP - RUN ONLY ONCE
    # ========================================================

    crops = prepare_crops(
        image_bgr=image_bgr,
        polygons=polygons,
    )

    if not crops:

        raise RuntimeError(
            "No valid OCR crops were created."
        )

    # ========================================================
    # 5. WARM UP RECOGNIZER
    # ========================================================

    print()
    print("=" * 90)
    print("RECOGNIZER WARM-UP")
    print("=" * 90)

    warmup_size = min(
        4,
        len(crops),
    )

    warmup_start = (
        time.perf_counter()
    )

    _ = recognizer.predict_batch(
        crops[:warmup_size]
    )

    warmup_elapsed = (
        time.perf_counter()
        - warmup_start
    )

    print(
        f"Warm-up images : "
        f"{warmup_size}"
    )

    print(
        f"Warm-up time   : "
        f"{warmup_elapsed:.3f}s"
    )

    # ========================================================
    # 6. PREPARE BATCH TEST ORDER
    # ========================================================

    batch_sizes = (
        BATCH_SIZES.copy()
    )

    # No need to test a configured size larger
    # than the crop count multiple times.
    #
    # Example:
    #
    # 41 crops
    #
    # batch 64  -> actual [41]
    # batch 128 -> actual [41]
    #
    # They are effectively the same input batch.
    #
    # We keep them here intentionally so you can
    # observe whether predict_batch behaves differently.
    # ========================================================

    if RANDOMIZE_ORDER:

        random.seed(
            RANDOM_SEED
        )

        random.shuffle(
            batch_sizes
        )

    print()
    print(
        "Benchmark order:"
    )

    print(
        batch_sizes
    )

    # ========================================================
    # 7. BENCHMARK
    # ========================================================

    print()
    print("=" * 90)
    print("RECOGNITION BENCHMARK")
    print("=" * 90)

    results = []

    for batch_size in batch_sizes:

        result = (
            benchmark_batch_size(
                crops=crops,
                recognizer=recognizer,
                batch_size=batch_size,
                repeats=REPEATS,
            )
        )

        results.append(
            result
        )

    # ========================================================
    # 8. SUMMARY
    # ========================================================

    fastest = print_summary(
        results
    )

    # ========================================================
    # 9. VERIFY OCR OUTPUT
    # ========================================================

    compare_predictions(
        results
    )

    # ========================================================
    # 10. SHOW SAMPLE TEXT
    # ========================================================

    print_sample_predictions(
        fastest
    )

    # ========================================================
    # 11. PIPELINE SUMMARY
    # ========================================================

    print()
    print("=" * 90)
    print("DOCUMENT SUMMARY")
    print("=" * 90)

    print(
        f"Image size           : "
        f"{width} x {height}"
    )

    print(
        f"Detected regions     : "
        f"{len(polygons)}"
    )

    print(
        f"Valid OCR crops      : "
        f"{len(crops)}"
    )

    print(
        f"Detection time       : "
        f"{detection_time:.3f}s"
    )

    print(
        f"Best recognition     : "
        f"{fastest['median_time']:.3f}s"
    )

    print(
        f"Best configured batch: "
        f"{fastest['batch_size']}"
    )

    print(
        f"Best actual batches  : "
        f"{fastest['actual_batch_sizes']}"
    )

    print(
        f"Detection + OCR      : "
        f"{detection_time + fastest['median_time']:.3f}s"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()