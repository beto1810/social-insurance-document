from paddleocr import TextDetection


def build_detector(
    device: str | None = None,
    score_thresh: float = 0.5,
) -> TextDetection:

    # PaddleOCR does not use PyTorch "mps"
    # in this TextDetection API.
    if device == "mps":
        device = "cpu"

    detector_config = {
        "model_name": "PP-OCRv5_mobile_det",
        "box_thresh": score_thresh,
    }

    if device is not None:
        detector_config["device"] = device

    return TextDetection(
        **detector_config,
    )