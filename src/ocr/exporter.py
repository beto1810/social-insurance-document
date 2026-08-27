# exporter.py

from pathlib import Path
from PIL import Image

LABEL_DEFAULT = "text"
LABELME_VERSION = "0.4.36"


def build_labelme_json(
    img_path: Path,
    dt_polys,
    texts,
    confidences=None,
    image_size: tuple[int, int] | None = None,
) -> dict:
    if image_size is None:
        with Image.open(img_path) as im:
            w, h = im.size
    else:
        w, h = image_size

    if confidences is None:
        confidences = [None] * len(texts)

    shapes = []
    for poly, text, confidence in zip(dt_polys, texts, confidences):
        points = [
            [float(x), float(y)]
            for x, y in poly
        ]

        shapes.append(
            {
                "label": LABEL_DEFAULT,
                "text": text,
                "confidence": confidence,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
        )

    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }