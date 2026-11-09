import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from huggingface_hub import hf_hub_download
from src.device import get_device
from src.ocr.preprocessing import crop_quad



HF_REPO_ID = "datnguyen1810/social-insurance-document-models"

HF_VIETOCR_FILE = (
    "ocr/vietocr_social_insurance_seq2seq/vgg_seq2seq_social_insurance.pth"
)

def get_vietocr_weights() -> Path:
    return Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_VIETOCR_FILE,
        )
    )

def build_recognizer(
    device: str | None = None,
    weights_path: str | Path | None = None,
) -> Predictor:

    if device is None:
        device = get_device()

    if weights_path is None:
        weights_path = get_vietocr_weights()

    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned VietOCR weights not found: {weights_path}"
        )

    total_start = time.perf_counter()

    print("[Recognizer] Loading config...",flush=True)

    t = time.perf_counter()

    config = Cfg.load_config_from_name("vgg_seq2seq")

    print(
        f"[Recognizer] Config loaded in "
        f"{time.perf_counter() - t:.2f}s",
        flush=True,
    )

    config["weights"] = str(weights_path)
    config["device"] = device

    # Complete fine-tuned checkpoint,
    # so no pretrained CNN download is needed.
    config["cnn"]["pretrained"] = False

    # Faster inference.
    config["predictor"] = config.get("predictor", {})
    config["predictor"]["beamsearch"] = False

    print(
        f"[Recognizer] Weights: {weights_path}",
        flush=True,
    )

    print(
        f"[Recognizer] Device: {device}",
        flush=True,
    )

    print(
        "[Recognizer] Creating VietOCR Predictor...",
        flush=True,
    )

    t = time.perf_counter()

    predictor = Predictor(config)

    print(
        f"[Recognizer] Predictor initialized in "
        f"{time.perf_counter() - t:.2f}s",
        flush=True,
    )

    print(
        f"[Recognizer] Total recognizer load time: "
        f"{time.perf_counter() - total_start:.2f}s",
        flush=True,
    )

    return predictor


def recognize_texts(
    image_bgr: np.ndarray,
    polygons,
    recognizer: Predictor,
    batch_size: int = 32,
    verbose: bool = False,
) -> tuple[list[str], list[float | None]]:

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1"
        )

    # ========================================================
    # 1. Crop all detected regions
    # ========================================================

    crops: list[Image.Image | None] = []

    for poly in polygons:

        crop = crop_quad(
            image_bgr,
            np.asarray(poly),
        )

        if (
            crop.size == 0
            or crop.shape[0] < 2
            or crop.shape[1] < 2
        ):
            crops.append(None)
            continue

        crop_rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB,
        )

        crops.append(
            Image.fromarray(crop_rgb)
        )

    # ========================================================
    # 2. Get valid crops
    # ========================================================

    valid_items = [
        (index, crop)
        for index, crop in enumerate(crops)
        if crop is not None
    ]

    # ========================================================
    # 4. Batch prediction helper
    # ========================================================

    predict_batch = getattr(
        recognizer,
        "predict_batch",
        None,
    )
    has_native_batch = callable(predict_batch)

    if verbose:
        print(
            "[OCR] Native predict_batch:",
            has_native_batch,
        )

    def to_confidence(probability) -> float | None:
        if probability is None:
            return None

        values = np.asarray(probability, dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            return None

        return float(np.mean(values))

    def predict_items(
        items: list[
            tuple[int, Image.Image]
        ],
    ) -> dict[int, tuple[str, float | None]]:

        predictions: dict[int, tuple[str, float | None]] = {}

        if not items:
            return predictions

        for start in range(
            0,
            len(items),
            batch_size,
        ):

            batch_items = items[
                start:start + batch_size
            ]

            batch_indices = [
                index
                for index, _ in batch_items
            ]

            batch_images = [
                image
                for _, image in batch_items
            ]

            if has_native_batch:
                batch_predictions, batch_probabilities = predict_batch(
                    batch_images,
                    return_prob=True,
                )

            else:
                batch_results = [
                    recognizer.predict(image, return_prob=True)
                    for image in batch_images
                ]
                batch_predictions = [result[0] for result in batch_results]
                batch_probabilities = [result[1] for result in batch_results]

            for index, text, probability in zip(
                batch_indices,
                batch_predictions,
                batch_probabilities,
            ):
                predictions[index] = (text, to_confidence(probability))

        return predictions

    # ========================================================
    # 5. Main OCR pass
    # ========================================================

    predicted = predict_items(
        valid_items
    )

    texts = [predicted.get(index, ("", None))[0] for index in range(len(crops))]
    confidences = [
        predicted.get(index, ("", None))[1]
        for index in range(len(crops))
    ]

    return texts, confidences