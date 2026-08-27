from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
)
from huggingface_hub import snapshot_download


from src.device import get_device

from src.kie.postprocessing import merge_related_entities


HF_REPO_ID = "datnguyen1810/social-insurance-document-models"
HF_KIE_FOLDER = "kie/layoutxlm"

DEVICE = get_device()
MAX_LENGTH = 512




def get_kie_model_path() -> Path:
    repo_path = snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=[
            f"{HF_KIE_FOLDER}/*",
        ],
    )

    return Path(repo_path) / HF_KIE_FOLDER

def _empty_prediction() -> dict:
    """Return the stable empty KIE response for documents without text."""
    return {
        "entities": [],
        "structured": {},
    }


def _move_inputs_to_device(
    encoding: dict,
    device: str,
) -> dict:
    """Move tensor inputs while preserving processor metadata."""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in encoding.items()
    }


# ============================================================
# Bounding boxes
# ============================================================


def polygon_to_bbox(
    points: list[list[float]],
) -> list[int]:
    """
    Convert polygon points:

        [
            [x1, y1],
            [x2, y2],
            ...
        ]

    into:

        [x0, y0, x1, y1]
    """

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    return [
        int(min(xs)),
        int(min(ys)),
        int(max(xs)),
        int(max(ys)),
    ]


def normalize_bbox(
    bbox: list[int],
    width: int,
    height: int,
) -> list[int]:
    """
    LayoutLM/LayoutXLM expects bounding boxes
    normalized to the range 0 -> 1000.
    """

    x0, y0, x1, y1 = bbox

    normalized = [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]

    return [
        max(
            0,
            min(1000, value),
        )
        for value in normalized
    ]


# ============================================================
# Reading order
# ============================================================


def sort_shapes_reading_order(
    shapes: list[dict],
) -> list[dict]:
    """
    Sort OCR regions:

        top -> bottom
        left -> right

    Your LabelMe JSON currently stores many regions
    in reverse/spatial order, so this step is important.
    """

    items = []

    for shape in shapes:
        text = shape.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        points = shape.get("points")

        if not points:
            continue

        bbox = polygon_to_bbox(points)

        x0, y0, x1, y1 = bbox

        items.append(
            {
                "shape": shape,
                "x": x0,
                "y": y0,
                "height": y1 - y0,
            }
        )

    items.sort(
        key=lambda item: (
            item["y"],
            item["x"],
        )
    )

    return [
        item["shape"]
        for item in items
    ]


# ============================================================
# Split OCR regions into words
# ============================================================


def split_shape_into_words(
    text: str,
    bbox: list[int],
) -> tuple[
    list[str],
    list[list[int]],
]:
    """
    Split one OCR region:

        "Họ và tên: NGUYỄN THANH TÂM"

    into:

        Họ
        và
        tên:
        NGUYỄN
        THANH
        TÂM

    Approximate individual word boxes based on
    character length.
    """

    words = text.split()

    if not words:
        return [], []

    x0, y0, x1, y1 = bbox

    width = max(
        x1 - x0,
        1,
    )

    # Include spaces in approximate length
    total_length = sum(
        len(word)
        for word in words
    )

    total_length += max(
        0,
        len(words) - 1,
    )

    current_x = x0

    word_boxes = []

    for index, word in enumerate(words):

        ratio = (
            len(word)
            / max(total_length, 1)
        )

        word_width = max(
            1,
            int(width * ratio),
        )

        word_x0 = current_x

        if index == len(words) - 1:
            word_x1 = x1
        else:
            word_x1 = min(
                x1,
                word_x0 + word_width,
            )

        word_boxes.append(
            [
                word_x0,
                y0,
                word_x1,
                y1,
            ]
        )

        # approximate whitespace
        space_width = max(
            1,
            int(
                width
                / max(total_length, 1)
            ),
        )

        current_x = (
            word_x1
            + space_width
        )

        current_x = min(
            current_x,
            x1,
        )

    return (
        words,
        word_boxes,
    )


def extract_words_and_boxes(
    document: dict,
) -> tuple[
    list[str],
    list[list[int]],
    list[int],
]:
    shapes = sort_shapes_reading_order(
        document["shapes"]
    )

    words: list[str] = []
    boxes: list[list[int]] = []
    region_indices: list[int] = []

    for region_index, shape in enumerate(shapes):

        text = shape.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        bbox = polygon_to_bbox(
            shape["points"]
        )

        (
            shape_words,
            shape_boxes,
        ) = split_shape_into_words(
            text,
            bbox,
        )

        words.extend(shape_words)
        boxes.extend(shape_boxes)

        # Every word remembers which OCR region it came from
        region_indices.extend(
            [region_index] * len(shape_words)
        )

    return (
        words,
        boxes,
        region_indices,
    )


# ============================================================
# Label helpers
# ============================================================


def clean_label(
    label: str,
) -> str:
    if label.startswith("B-"):
        return label[2:]

    if label.startswith("I-"):
        return label[2:]

    return label


# ============================================================
# Token -> word predictions
# ============================================================


def convert_predictions_to_words(
    words: list[str],
    word_ids: list[int | None],
    prediction_ids: torch.Tensor,
    confidence_values: torch.Tensor,
    id2label: dict,
) -> tuple[
    list[str],
    list[float],
]:
    """
    LayoutLM tokenizer can split words into subwords.

    Example:

        NGUYỄN
          ↓
        ▁NGUY
        ỄN

    We only use the first subtoken prediction for
    each original OCR word.
    """

    word_predictions: dict[int, str] = {}
    word_confidences: dict[int, float] = {}

    for token_index, word_id in enumerate(
        word_ids
    ):
        if word_id is None:
            continue

        if word_id in word_predictions:
            continue

        pred_id = int(
            prediction_ids[
                token_index
            ].item()
        )

        label = id2label.get(
            pred_id,
            str(pred_id),
        )

        confidence = float(
            confidence_values[
                token_index
            ].item()
        )

        word_predictions[word_id] = label
        word_confidences[word_id] = (
            confidence
        )

    predicted_labels = []
    confidences = []

    for word_index in range(
        len(words)
    ):
        predicted_labels.append(
            word_predictions.get(
                word_index,
                "O",
            )
        )

        confidences.append(
            word_confidences.get(
                word_index,
                0.0,
            )
        )

    return (
        predicted_labels,
        confidences,
    )


# ============================================================
# Debugging
# ============================================================


def print_reading_order(
    words: list[str],
    boxes: list[list[int]],
    limit: int = 120,
) -> None:
    print()
    print("=" * 120)
    print("READING ORDER")
    print("=" * 120)

    for index, (
        word,
        bbox,
    ) in enumerate(
        zip(
            words[:limit],
            boxes[:limit],
        )
    ):
        print(
            f"{index:>4} | "
            f"{word:<35} | "
            f"{bbox}"
        )


def print_all_entities(
    words: list[str],
    labels: list[str],
    confidences: list[float],
) -> None:
    print()
    print("=" * 120)
    print("NON-O PREDICTIONS")
    print("=" * 120)

    found = False

    for index, (
        word,
        label,
        confidence,
    ) in enumerate(
        zip(
            words,
            labels,
            confidences,
        )
    ):
        if label == "O":
            continue

        found = True

        print(
            f"{index:>4} | "
            f"{word:<35} | "
            f"{label:<30} | "
            f"{confidence:.4f}"
        )

    if not found:
        print(
            "No entity labels predicted."
        )


def print_name_context(
    words: list[str],
    boxes: list[list[int]],
    labels: list[str],
    confidences: list[float],
    context_size: int = 12,
) -> None:
    """
    Show surrounding words for predictions of:

        PATIENT_NAME
        PARENT_NAME
    """

    interesting_indices = []

    for index, label in enumerate(
        labels
    ):
        entity_label = clean_label(
            label
        )

        if entity_label in {
            "PATIENT_NAME",
            "PARENT_NAME",
        }:
            interesting_indices.append(
                index
            )

    print()
    print("=" * 120)
    print(
        "CONTEXT AROUND "
        "PATIENT_NAME / PARENT_NAME"
    )
    print("=" * 120)

    if not interesting_indices:
        print(
            "No PATIENT_NAME or "
            "PARENT_NAME predicted."
        )
        return

    # Build ranges around interesting words
    ranges = []

    for index in interesting_indices:

        start = max(
            0,
            index - context_size,
        )

        end = min(
            len(words),
            index + context_size + 1,
        )

        if not ranges:
            ranges.append(
                [start, end]
            )
            continue

        previous = ranges[-1]

        if start <= previous[1]:
            previous[1] = max(
                previous[1],
                end,
            )
        else:
            ranges.append(
                [start, end]
            )

    for context_index, (
        start,
        end,
    ) in enumerate(
        ranges,
        start=1,
    ):
        print()
        print(
            f"Context #{context_index}: "
            f"{start} -> {end - 1}"
        )
        print("-" * 120)

        for index in range(
            start,
            end,
        ):
            label = labels[index]

            entity_label = clean_label(
                label
            )

            marker = ""

            if entity_label == "PATIENT_NAME":
                marker = (
                    " <<< PATIENT_NAME"
                )

            elif entity_label == "PARENT_NAME":
                marker = (
                    " <<< PARENT_NAME"
                )

            print(
                f"{index:>4} | "
                f"{words[index]:<32} | "
                f"{label:<27} | "
                f"{confidences[index]:.4f} | "
                f"{boxes[index]}"
                f"{marker}"
            )


# ============================================================
# BIO -> entities
# ============================================================


def build_entities(
    words: list[str],
    predicted_labels: list[str],
    confidences: list[float],
    word_region_indices: list[int],
) -> list[dict]:

    entities: list[dict] = []

    current_label = None
    current_words = []
    current_confidences = []
    current_region_indices = []

    def flush() -> None:
        nonlocal current_label
        nonlocal current_words
        nonlocal current_confidences
        nonlocal current_region_indices

        if (
            current_label
            and current_words
        ):
            avg_confidence = (
                sum(current_confidences)
                / len(current_confidences)
            )

            entities.append(
                {
                    "label": current_label,
                    "text": " ".join(
                        current_words
                    ),
                    "confidence": round(
                        avg_confidence,
                        4,
                    ),

                    # Important for postprocessing
                    "region_indexes": sorted(
                        set(current_region_indices)
                    ),
                }
            )

        current_label = None
        current_words = []
        current_confidences = []
        current_region_indices = []

    for (
        word,
        bio_label,
        confidence,
        region_index,
    ) in zip(
        words,
        predicted_labels,
        confidences,
        word_region_indices,
    ):

        if bio_label == "O":
            flush()
            continue

        if bio_label.startswith("B-"):

            flush()

            current_label = bio_label[2:]

            current_words = [
                word
            ]

            current_confidences = [
                confidence
            ]

            current_region_indices = [
                region_index
            ]

            continue

        if bio_label.startswith("I-"):

            entity_label = bio_label[2:]

            if current_label == entity_label:

                current_words.append(
                    word
                )

                current_confidences.append(
                    confidence
                )

                current_region_indices.append(
                    region_index
                )

            else:
                flush()

                current_label = entity_label

                current_words = [
                    word
                ]

                current_confidences = [
                    confidence
                ]

                current_region_indices = [
                    region_index
                ]

            continue

        flush()

    flush()

    return entities


def build_structured(
    entities: list[dict],
) -> dict[str, list[str]]:
    structured: dict[
        str,
        list[str],
    ] = {}

    for entity in entities:

        label = entity["label"]
        text = entity["text"]

        structured.setdefault(
            label,
            [],
        ).append(text)

    return structured


# ============================================================
# Reusable LayoutXLM inference pipeline
# ============================================================


class LayoutXLMInference:
    """
    Reusable Stage 3 KIE inference.

    Model is loaded ONCE when this class is created.

    Can be used by:
        - src/kie/main.py
        - src/pipeline.py
        - FastAPI later
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = DEVICE,
        max_length: int = MAX_LENGTH,
        debug: bool = False,
    ) -> None:

        if model_path is None:
            model_path = get_kie_model_path()

        self.model_path = Path(model_path)
        self.device = device
        self.max_length = max_length
        self.debug = debug

        print()
        print("=" * 120)
        print("Loading LayoutXLM KIE")
        print("=" * 120)

        print(f"Model : {self.model_path}")
        print(f"Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            apply_ocr=False,
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

        print(
            "Loaded model class:",
            type(self.model).__name__,
        )

        print(
            "Model type:",
            self.model.config.model_type,
        )

    # ============================================================
    # Main prediction
    # ============================================================

    def predict(
        self,
        document: dict,
        image: Image.Image,
    ) -> dict:
        """
        Run LayoutXLM KIE.

        Parameters
        ----------
        document:
            OCR result in your current LabelMe format:

            {
                "imageWidth": ...,
                "imageHeight": ...,
                "shapes": [
                    {
                        "text": "...",
                        "points": [...]
                    }
                ]
            }

        image:
            PIL RGB image corresponding to the OCR document.

        Returns
        -------
        {
            "entities": [...],
            "structured": {...}
        }
        """

        # --------------------------------------------------------
        # Image size
        # --------------------------------------------------------

        image_width = document[
            "imageWidth"
        ]

        image_height = document[
            "imageHeight"
        ]

        # --------------------------------------------------------
        # OCR regions -> words
        #
        # IMPORTANT:
        # extract_words_and_boxes()
        # already includes:
        #
        #   sort_shapes_reading_order()
        #       ↓
        #   split_shape_into_words()
        #
        # Keep this exactly consistent with training/inference.
        # --------------------------------------------------------

        (
            words,
            raw_boxes,
            word_region_indices,
        ) = extract_words_and_boxes(
            document
        )

        if not words:
            return _empty_prediction()

        # --------------------------------------------------------
        # Normalize bounding boxes
        # --------------------------------------------------------

        boxes = [
            normalize_bbox(
                bbox,
                image_width,
                image_height,
            )
            for bbox in raw_boxes
        ]

        if self.debug:

            print()
            print(
                f"Words: {len(words)}"
            )

            print_reading_order(
                words,
                boxes,
                limit=120,
            )

        # --------------------------------------------------------
        # LayoutXLM processor
        # --------------------------------------------------------

        encoding = self.processor(
            images=image,
            text=words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # --------------------------------------------------------
        # Get word IDs BEFORE moving tensors
        # --------------------------------------------------------

        word_ids = encoding.word_ids(
            batch_index=0
        )

        # --------------------------------------------------------
        # Token count / truncation check
        # --------------------------------------------------------

        attention_mask = (
            encoding[
                "attention_mask"
            ][0]
        )

        encoded_tokens = int(
            attention_mask
            .sum()
            .item()
        )

        if self.debug:

            print(
                f"Encoded tokens: "
                f"{encoded_tokens}"
            )

        if (
            encoded_tokens
            >= self.max_length
        ):

            print(
                "[WARNING] LayoutXLM sequence "
                f"reached max_length="
                f"{self.max_length}."
            )

            print(
                "[WARNING] Some OCR context "
                "may have been truncated."
            )

        # --------------------------------------------------------
        # Move tensors to device
        # --------------------------------------------------------

        model_inputs = _move_inputs_to_device(
            encoding,
            self.device,
        )

        # --------------------------------------------------------
        # Inference
        # --------------------------------------------------------

        with torch.inference_mode():

            outputs = self.model(
                **model_inputs
            )

        logits = outputs.logits

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
        )

        prediction_ids = (
            probabilities.argmax(
                dim=-1
            )[0]
        )

        confidence_values = (
            probabilities.max(
                dim=-1
            ).values[0]
        )

        # --------------------------------------------------------
        # Subtokens -> OCR words
        # --------------------------------------------------------

        (
            predicted_labels,
            confidences,
        ) = convert_predictions_to_words(
            words=words,
            word_ids=word_ids,
            prediction_ids=prediction_ids,
            confidence_values=confidence_values,
            id2label=self.id2label,
        )

        # --------------------------------------------------------
        # Debug
        # --------------------------------------------------------

        if self.debug:

            print_all_entities(
                words=words,
                labels=predicted_labels,
                confidences=confidences,
            )

            print_name_context(
                words=words,
                boxes=boxes,
                labels=predicted_labels,
                confidences=confidences,
                context_size=12,
            )

        # --------------------------------------------------------
        # --------------------------------------------------------
        # BIO -> entities
        # --------------------------------------------------------

        entities = build_entities(
            words=words,
            predicted_labels=predicted_labels,
            confidences=confidences,
            word_region_indices=word_region_indices,
        )

        # --------------------------------------------------------
        # KIE postprocessing
        # --------------------------------------------------------

        sorted_regions = sort_shapes_reading_order(
            document["shapes"]
        )

        entities = merge_related_entities(
            entities=entities,
            ocr_regions=sorted_regions,
        )

        # --------------------------------------------------------
        # Structured result
        # --------------------------------------------------------

        structured = build_structured(
            entities
        )

        # --------------------------------------------------------
        # Result
        # --------------------------------------------------------

        result = {
            "entities": entities,
            "structured": structured,
        }

        return result


# ============================================================
# Convenience function
#
# Optional.
# Useful for quick calls, but DON'T use this inside the main
# product pipeline because it loads LayoutXLM every call.
# ============================================================


def run_layoutxlm(
    document: dict,
    image: Image.Image,
    model_path: str | Path | None = None,
    device: str = DEVICE,
    debug: bool = False,
) -> dict:

    inference = LayoutXLMInference(
        model_path=model_path,
        device=device,
        debug=debug,
    )

    return inference.predict(
        document=document,
        image=image,
    )