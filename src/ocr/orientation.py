from pathlib import Path

from paddleocr import DocImgOrientationClassification
from PIL import Image, ImageOps


class DocumentOrientationCorrector:
    def __init__(self):
        self.model = DocImgOrientationClassification(
            model_name="PP-LCNet_x1_0_doc_ori"
        )

    def correct(self, image_path: str | Path):
        image_path = str(image_path)

        # Load image and normalize EXIF rotation first
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)

        # Predict visual document orientation
        results = list(self.model.predict(image_path))

        if not results:
            raise RuntimeError(
                f"Orientation model returned no result for {image_path}"
            )

        result = results[0]

        # PaddleOCR output typically contains:
        # class_ids, scores, label_names
        res = result.json["res"]

        label_name = res["label_names"][0]

        # Depending on PaddleOCR version this may look like:
        # "0", "90", "180", "270"
        # or possibly "90_degree"
        angle = int(
            str(label_name)
            .replace("_degree", "")
            .replace("°", "")
        )

        confidence = float(res["scores"][0])

        if angle != 0:
            image = image.rotate(
                angle,
                expand=True,
            )

        return image, angle, confidence

if __name__ == "__main__":
    image_path = (
        "data/document-kie/image/"
        "162_PDFsam_Om dau 2_page_001.png"
    )

    corrector = DocumentOrientationCorrector()

    corrected_image, angle, confidence = corrector.correct(
        image_path
    )

    print(f"Detected rotation: {angle}°")
    print(f"Confidence: {confidence:.4f}")

    corrected_image.save(
        "data/document-kie/image/corrected_test.png"
    )