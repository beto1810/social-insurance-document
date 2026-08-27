from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import pymupdf
from PIL import Image

from src.device import get_device as auto_get_device

# ============================================================
# Stage 2 - OCR
# ============================================================

from src.ocr.detector import build_detector
from src.ocr.orientation import DocumentOrientationCorrector
from src.ocr.pipeline import process_image
from src.ocr.recognizer import build_recognizer

# ============================================================
# Stage 3 - KIE
# ============================================================

from src.kie.infer_layoutxlm import LayoutXLMInference
from src.kie.postprocessing import build_validation_notifications


# ============================================================
# Configuration
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}


# ============================================================
# Helper functions
# ============================================================


def build_structured_with_confidence(
    entities: list[dict],
) -> dict[str, list[dict]]:
    """
    Group extracted values without dropping confidence scores.
    """

    structured: dict[str, list[dict]] = {}

    for entity in entities:

        label = entity.get("label")

        if not label:
            continue

        structured.setdefault(
            label,
            [],
        ).append(
            {
                "text": entity.get(
                    "text",
                    "",
                ),
                "confidence": entity.get(
                    "confidence"
                ),
            }
        )

    return structured


def _to_pil_image(
    image_bgr,
) -> Image.Image:
    """
    Convert OpenCV BGR image to PIL RGB image.
    """

    return Image.fromarray(
        cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )
    )


def _build_pipeline_result(
    image_path: Path,
    document: dict,
    kie_result: dict,
    timings: dict,
    source_document: str | None = None,
    page_number: int | None = None,
) -> dict:
    """
    Combine OCR, KIE, confidence, validation,
    page information, and timing data.
    """

    entities = kie_result.get(
        "entities",
        [],
    )

    result = {
        "image": str(image_path),
        "orientation": document.get(
            "orientation"
        ),
        "imageWidth": document[
            "imageWidth"
        ],
        "imageHeight": document[
            "imageHeight"
        ],
        "ocr_regions": document[
            "shapes"
        ],
        "entities": entities,
        "structured": kie_result.get(
            "structured",
            {},
        ),
        "structured_with_confidence": (
            build_structured_with_confidence(
                entities
            )
        ),
        "notifications": (
            build_validation_notifications(
                entities
            )
        ),
        "timings": timings,
    }

    if source_document is not None:
        result["source_document"] = (
            source_document
        )

    if page_number is not None:
        result["page_number"] = (
            page_number
        )

    return result


# ============================================================
# Device
# ============================================================


def get_device() -> str:
    """Return the best available PyTorch device in priority order: cuda -> mps -> cpu."""
    return auto_get_device()


# ============================================================
# PDF utilities
# ============================================================


def get_pdf_page_count(
    pdf_path: str | Path,
) -> int:
    """
    Return number of pages in a PDF.
    """

    pdf_path = Path(
        pdf_path
    )

    with pymupdf.open(
        str(pdf_path)
    ) as pdf:
        return len(pdf)


def render_pdf_pages(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
) -> list[Path]:
    """
    Render all PDF pages into PNG images.

    Args:
        pdf_path:
            Input PDF.

        output_dir:
            Temporary/output folder
            for page images.

        dpi:
            Rendering resolution.

            Recommended:
                150 = faster
                200 = good default
                250 = higher quality
                300 = highest quality / slower

    Returns:
        List of generated PNG paths.
    """

    pdf_path = Path(
        pdf_path
    )

    output_dir = Path(
        output_dir
    )

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    page_paths: list[Path] = []

    zoom = dpi / 72.0

    matrix = pymupdf.Matrix(
        zoom,
        zoom,
    )

    pdf = pymupdf.open(
        str(pdf_path)
    )

    try:

        total_pages = len(pdf)

        print()
        print("=" * 80)
        print("PDF RENDERING")
        print("=" * 80)

        print(
            f"PDF               : "
            f"{pdf_path.name}"
        )

        print(
            f"Pages             : "
            f"{total_pages}"
        )

        print(
            f"DPI               : "
            f"{dpi}"
        )

        for page_index in range(
            total_pages
        ):

            page = pdf[
                page_index
            ]

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            page_path = (
                output_dir
                / (
                    f"{pdf_path.stem}"
                    f"_page_"
                    f"{page_index + 1:03d}"
                    f".png"
                )
            )

            pixmap.save(
                str(page_path)
            )

            page_paths.append(
                page_path
            )

            print(
                f"[PDF] Rendered "
                f"page "
                f"{page_index + 1}/"
                f"{total_pages}: "
                f"{pixmap.width}x"
                f"{pixmap.height}"
            )

    finally:

        pdf.close()

    return page_paths


# ============================================================
# Document Pipeline
# ============================================================


class DocumentPipeline:
    """
    Full document understanding pipeline.

    Supported input:
        - PDF
        - JPG
        - JPEG
        - PNG
        - BMP
        - TIFF
        - WEBP

    Architecture:

        PDF / Image
            ↓
        Input preprocessing

        PDF:
            PDF
             ↓
            Render pages
             ↓
            Image

        Image:
            Use directly

             ↓
        Stage 2 OCR
             ↓
        LabelMe-like OCR document
             ↓
        Stage 3 LayoutXLM KIE
             ↓
        Postprocessing
             ↓
        Structured entities


    IMPORTANT:

    All models are loaded ONCE when
    DocumentPipeline is created.

    Repeated calls to:

        process()
        process_document()
        process_many()

    reuse the same models.
    """

    def __init__(
        self,
        kie_model_path: str | Path | None = None,
        device: str | None = None,
        score_thresh: float = 0.1,
        kie_debug: bool = False,
        recognition_batch_size: int = 32,
        enable_orientation: bool = True,
        pdf_dpi: int = 200,
    ) -> None:

        # ====================================================
        # Configuration
        # ====================================================

        if device is None:
            device = get_device()

        self.device = device

        self.score_thresh = (
            score_thresh
        )

        self.recognition_batch_size = (
            recognition_batch_size
        )

        self.enable_orientation = (
            enable_orientation
        )

        self.pdf_dpi = (
            pdf_dpi
        )

        # ====================================================
        # Pipeline startup
        # ====================================================

        pipeline_load_start = (
            time.perf_counter()
        )

        print()
        print("=" * 80)
        print("Loading Document Pipeline")
        print("=" * 80)

        print(
            f"Device            : "
            f"{self.device}"
        )

        print(
            f"PDF DPI           : "
            f"{self.pdf_dpi}"
        )

        # ====================================================
        # Stage 2 - Detector
        # ====================================================

        print()
        print(
            "[Stage 2] "
            "Loading detector..."
        )

        t = time.perf_counter()

        self.detector = (
            build_detector(
                device=self.device,
                score_thresh=self.score_thresh,
            )
        )

        detector_load_time = (
            time.perf_counter()
            - t
        )

        print(
            f"[Stage 2] "
            f"Detector loaded in "
            f"{detector_load_time:.3f}s"
        )

        # ====================================================
        # Stage 2 - VietOCR
        # ====================================================

        print()
        print(
            "[Stage 2] "
            "Loading recognizer..."
        )

        t = time.perf_counter()

        self.recognizer = (
            build_recognizer(
                self.device
            )
        )

        recognizer_load_time = (
            time.perf_counter()
            - t
        )

        print(
            f"[Stage 2] "
            f"Recognizer loaded in "
            f"{recognizer_load_time:.3f}s"
        )

        # ====================================================
        # Stage 2 - Orientation
        # ====================================================

        print()
        print(
            "[Stage 2] "
            "Loading orientation model..."
        )

        t = time.perf_counter()

        self.orientation_corrector = (
            DocumentOrientationCorrector()
        )

        orientation_load_time = (
            time.perf_counter()
            - t
        )

        print(
            f"[Stage 2] "
            f"Orientation model loaded in "
            f"{orientation_load_time:.3f}s"
        )

        # ====================================================
        # Stage 3 - LayoutXLM
        # ====================================================

        print()
        print(
            "[Stage 3] "
            "Loading LayoutXLM..."
        )

        t = time.perf_counter()

        self.kie = (
            LayoutXLMInference(
                model_path=kie_model_path,
                device=self.device,
                debug=kie_debug,
            )
        )

        kie_load_time = (
            time.perf_counter()
            - t
        )

        print(
            f"[Stage 3] "
            f"LayoutXLM loaded in "
            f"{kie_load_time:.3f}s"
        )

        # ====================================================
        # Startup summary
        # ====================================================

        pipeline_load_time = (
            time.perf_counter()
            - pipeline_load_start
        )

        print()
        print("=" * 60)
        print(
            "PIPELINE STARTUP PERFORMANCE"
        )
        print("=" * 60)

        print(
            f"Detector          : "
            f"{detector_load_time:.3f}s"
        )

        print(
            f"Recognizer        : "
            f"{recognizer_load_time:.3f}s"
        )

        print(
            f"Orientation model : "
            f"{orientation_load_time:.3f}s"
        )

        print(
            f"LayoutXLM         : "
            f"{kie_load_time:.3f}s"
        )

        print("-" * 60)

        print(
            f"TOTAL STARTUP     : "
            f"{pipeline_load_time:.3f}s"
        )

        print("=" * 60)

        print()
        print(
            "[INFO] Pipeline ready."
        )

    # ========================================================
    # Process ONE image
    # ========================================================

    def process(
        self,
        image_path: str | Path,
        source_document: str | None = None,
        page_number: int | None = None,
    ) -> dict:
        """
        Process exactly ONE image.

        This remains the core OCR + KIE method.

        PDF pages are rendered as images and
        passed through this function.
        """

        total_start = (
            time.perf_counter()
        )

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

        if (
            image_path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            raise ValueError(
                "process() only accepts "
                f"images, got: "
                f"{image_path.suffix}"
            )

        print()
        print("=" * 80)

        if page_number is not None:

            print(
                f"Processing page "
                f"{page_number}: "
                f"{image_path.name}"
            )

        else:

            print(
                f"Processing: "
                f"{image_path.name}"
            )

        print("=" * 80)

        # ====================================================
        # Stage 2 - OCR
        # ====================================================

        print()
        print(
            "STAGE 2 - OCR"
        )

        print("-" * 80)

        stage2_start = (
            time.perf_counter()
        )

        ocr_result = process_image(
            img_path=image_path,
            detector=self.detector,
            recognizer=self.recognizer,
            orient_corrector=(
                self.orientation_corrector
            ),
            score_thresh=(
                self.score_thresh
            ),
            recognition_batch_size=(
                self.recognition_batch_size
            ),
            enable_orientation=(
                self.enable_orientation
            ),
        )

        stage2_time = (
            time.perf_counter()
            - stage2_start
        )

        document = (
            ocr_result[
                "document"
            ]
        )

        image_bgr = (
            ocr_result[
                "image_bgr"
            ]
        )

        stage2_details = (
            ocr_result.get(
                "timings",
                {},
            )
        )

        print(
            f"[OCR] Regions: "
            f"{len(document['shapes'])}"
        )

        # ====================================================
        # OCR debug output
        # ====================================================

        for index, shape in enumerate(
            document["shapes"]
        ):

            print(
                f"  [{index:03d}] "
                f"{shape.get('text', '')}"
            )

        # ====================================================
        # OpenCV BGR -> PIL RGB
        # ====================================================

        conversion_start = (
            time.perf_counter()
        )

        image_pil = (
            _to_pil_image(
                image_bgr
            )
        )

        conversion_time = (
            time.perf_counter()
            - conversion_start
        )

        # ====================================================
        # Stage 3 - KIE
        # ====================================================

        print()
        print(
            "STAGE 3 - KIE"
        )

        print("-" * 80)

        stage3_start = (
            time.perf_counter()
        )

        # IMPORTANT:
        # predict only once.
        kie_result = (
            self.kie.predict(
                document=document,
                image=image_pil,
            )
        )

        stage3_time = (
            time.perf_counter()
            - stage3_start
        )

        print(
            f"[PERF] Stage 3 KIE: "
            f"{stage3_time:.3f}s"
        )

        # ====================================================
        # Total time
        # ====================================================

        total_time = (
            time.perf_counter()
            - total_start
        )

        # ====================================================
        # Performance summary
        # ====================================================

        print()
        print("=" * 60)

        print(
            "FULL PIPELINE PERFORMANCE"
        )

        print("=" * 60)

        print(
            f"Stage 2 OCR       : "
            f"{stage2_time:.3f}s"
        )

        if stage2_details:

            if (
                "orientation"
                in stage2_details
            ):

                print(
                    f"  Orientation     : "
                    f"{stage2_details['orientation']:.3f}s"
                )

            if (
                "load_image"
                in stage2_details
            ):

                print(
                    f"  Load image      : "
                    f"{stage2_details['load_image']:.3f}s"
                )

            if (
                "stamp_suppression"
                in stage2_details
            ):

                print(
                    f"  Stamp suppress  : "
                    f"{stage2_details['stamp_suppression']:.3f}s"
                )

            if (
                "detection"
                in stage2_details
            ):

                print(
                    f"  Detection       : "
                    f"{stage2_details['detection']:.3f}s"
                )

            if (
                "recognition"
                in stage2_details
            ):

                print(
                    f"  Recognition     : "
                    f"{stage2_details['recognition']:.3f}s"
                )

            if (
                "build_json"
                in stage2_details
            ):

                print(
                    f"  Build JSON      : "
                    f"{stage2_details['build_json']:.3f}s"
                )

        print(
            f"BGR -> PIL        : "
            f"{conversion_time:.3f}s"
        )

        print(
            f"Stage 3 KIE       : "
            f"{stage3_time:.3f}s"
        )

        print("-" * 60)

        print(
            f"TOTAL             : "
            f"{total_time:.3f}s"
        )

        print("=" * 60)

        # ====================================================
        # Final result
        # ====================================================

        return _build_pipeline_result(
            image_path=image_path,
            document=document,
            kie_result=kie_result,
            source_document=source_document,
            page_number=page_number,
            timings={
                "stage2_ocr": (
                    stage2_time
                ),
                "stage2_details": (
                    stage2_details
                ),
                "image_conversion": (
                    conversion_time
                ),
                "stage3_kie": (
                    stage3_time
                ),
                "total": (
                    total_time
                ),
            },
        )

    # ========================================================
    # Process ONE document
    # ========================================================

    def process_document(
        self,
        file_path: str | Path,
        pdf_dpi: int | None = None,
    ) -> dict:
        """
        Process either an image or a PDF.

        Image:
            processed directly.

        PDF:
            pages are rendered into temporary PNG files,
            then each page runs through the SAME OCR/KIE
            pipeline.
        """

        file_path = Path(
            file_path
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Document not found: "
                f"{file_path}"
            )

        extension = (
            file_path.suffix.lower()
        )

        if (
            extension
            not in SUPPORTED_EXTENSIONS
        ):

            raise ValueError(
                f"Unsupported document "
                f"type: {extension}"
            )

        if pdf_dpi is None:

            pdf_dpi = (
                self.pdf_dpi
            )

        document_start = (
            time.perf_counter()
        )

        print()
        print("#" * 80)
        print(
            f"DOCUMENT: "
            f"{file_path.name}"
        )
        print("#" * 80)

        # ====================================================
        # IMAGE
        # ====================================================

        if (
            extension
            in IMAGE_EXTENSIONS
        ):

            result = self.process(
                image_path=file_path,
                source_document=(
                    str(file_path)
                ),
                page_number=1,
            )

            total_time = (
                time.perf_counter()
                - document_start
            )

            return {
                "success": True,
                "document": (
                    str(file_path)
                ),
                "document_name": (
                    file_path.name
                ),
                "document_type": (
                    "image"
                ),
                "total_pages": 1,
                "successful_pages": 1,
                "failed_pages": 0,
                "pages": [
                    {
                        "page_number": 1,
                        "success": True,
                        "result": result,
                    }
                ],
                "timings": {
                    "pdf_rendering": 0.0,
                    "processing": (
                        result[
                            "timings"
                        ][
                            "total"
                        ]
                    ),
                    "total": total_time,
                },
            }

        # ====================================================
        # PDF
        # ====================================================

        print()
        print("=" * 80)
        print("PDF DOCUMENT")
        print("=" * 80)

        print(
            f"File              : "
            f"{file_path}"
        )

        print(
            f"Render DPI        : "
            f"{pdf_dpi}"
        )

        pages: list[dict] = []

        successful_pages = 0
        failed_pages = 0

        rendering_time = 0.0
        page_processing_time = 0.0

        # Temporary directory automatically
        # disappears after processing.
        with tempfile.TemporaryDirectory(
            prefix="document_pipeline_"
        ) as temp_dir:

            # ================================================
            # Render PDF
            # ================================================

            render_start = (
                time.perf_counter()
            )

            page_paths = (
                render_pdf_pages(
                    pdf_path=file_path,
                    output_dir=temp_dir,
                    dpi=pdf_dpi,
                )
            )

            rendering_time = (
                time.perf_counter()
                - render_start
            )

            total_pages = len(
                page_paths
            )

            print()
            print(
                f"[PDF] Rendering "
                f"finished in "
                f"{rendering_time:.3f}s"
            )

            print(
                f"[PDF] Processing "
                f"{total_pages} pages..."
            )

            # ================================================
            # Process PDF pages
            # ================================================

            for (
                page_number,
                page_path,
            ) in enumerate(
                page_paths,
                start=1,
            ):

                print()
                print("#" * 80)

                print(
                    f"PDF PAGE "
                    f"{page_number}/"
                    f"{total_pages}"
                )

                print("#" * 80)

                page_start = (
                    time.perf_counter()
                )

                try:

                    result = (
                        self.process(
                            image_path=(
                                page_path
                            ),
                            source_document=(
                                str(
                                    file_path
                                )
                            ),
                            page_number=(
                                page_number
                            ),
                        )
                    )

                    page_time = (
                        time.perf_counter()
                        - page_start
                    )

                    page_processing_time += (
                        page_time
                    )

                    pages.append(
                        {
                            "page_number": (
                                page_number
                            ),
                            "success": True,
                            "duration": (
                                page_time
                            ),
                            "result": (
                                result
                            ),
                        }
                    )

                    successful_pages += 1

                except Exception as exc:

                    page_time = (
                        time.perf_counter()
                        - page_start
                    )

                    page_processing_time += (
                        page_time
                    )

                    print(
                        f"[FAILED] "
                        f"PDF page "
                        f"{page_number}: "
                        f"{exc}"
                    )

                    pages.append(
                        {
                            "page_number": (
                                page_number
                            ),
                            "success": False,
                            "duration": (
                                page_time
                            ),
                            "error": (
                                str(exc)
                            ),
                        }
                    )

                    failed_pages += 1

        # ====================================================
        # PDF summary
        # ====================================================

        total_document_time = (
            time.perf_counter()
            - document_start
        )

        print()
        print("=" * 60)
        print(
            "PDF DOCUMENT PERFORMANCE"
        )
        print("=" * 60)

        print(
            f"Pages             : "
            f"{len(pages)}"
        )

        print(
            f"Success           : "
            f"{successful_pages}"
        )

        print(
            f"Failed            : "
            f"{failed_pages}"
        )

        print(
            f"PDF rendering     : "
            f"{rendering_time:.3f}s"
        )

        print(
            f"Page processing   : "
            f"{page_processing_time:.3f}s"
        )

        print("-" * 60)

        print(
            f"TOTAL DOCUMENT    : "
            f"{total_document_time:.3f}s"
        )

        if successful_pages > 0:

            print(
                f"Average/page      : "
                f"{page_processing_time / successful_pages:.3f}s"
            )

        print("=" * 60)

        return {
            "success": (
                failed_pages == 0
            ),
            "document": (
                str(file_path)
            ),
            "document_name": (
                file_path.name
            ),
            "document_type": (
                "pdf"
            ),
            "total_pages": (
                len(pages)
            ),
            "successful_pages": (
                successful_pages
            ),
            "failed_pages": (
                failed_pages
            ),
            "pages": (
                pages
            ),
            "timings": {
                "pdf_rendering": (
                    rendering_time
                ),
                "page_processing": (
                    page_processing_time
                ),
                "total": (
                    total_document_time
                ),
            },
        }

    # ========================================================
    # Process MANY documents
    # ========================================================

    def process_many(
        self,
        file_paths: list[str | Path],
        pdf_dpi: int | None = None,
    ) -> list[dict]:
        """
        Process multiple PDFs/images sequentially.

        IMPORTANT:
        The same detector, recognizer,
        orientation model and LayoutXLM model
        are reused across all documents.
        """

        batch_start = (
            time.perf_counter()
        )

        results = []

        success_count = 0
        failed_count = 0

        total_documents = len(
            file_paths
        )

        if total_documents == 0:
            return []

        print()
        print("=" * 80)

        print(
            f"BATCH PROCESSING: "
            f"{total_documents} documents"
        )

        print("=" * 80)

        for (
            index,
            file_path,
        ) in enumerate(
            file_paths,
            start=1,
        ):

            print()
            print(
                f"[DOCUMENT "
                f"{index}/"
                f"{total_documents}] "
                f"{file_path}"
            )

            try:

                document_start = (
                    time.perf_counter()
                )

                result = (
                    self.process_document(
                        file_path=file_path,
                        pdf_dpi=pdf_dpi,
                    )
                )

                document_time = (
                    time.perf_counter()
                    - document_start
                )

                results.append(
                    {
                        "success": (
                            result.get(
                                "success",
                                False,
                            )
                        ),
                        "document": (
                            str(file_path)
                        ),
                        "duration": (
                            document_time
                        ),
                        "result": (
                            result
                        ),
                    }
                )

                if result.get(
                    "success",
                    False,
                ):

                    success_count += 1

                else:

                    failed_count += 1

            except Exception as exc:

                print(
                    f"[FAILED] "
                    f"{file_path}: "
                    f"{exc}"
                )

                results.append(
                    {
                        "success": False,
                        "document": (
                            str(file_path)
                        ),
                        "error": (
                            str(exc)
                        ),
                    }
                )

                failed_count += 1

        # ====================================================
        # Batch performance
        # ====================================================

        batch_time = (
            time.perf_counter()
            - batch_start
        )

        average_time = (
            batch_time
            / total_documents
        )

        print()
        print("=" * 60)

        print(
            "BATCH PERFORMANCE"
        )

        print("=" * 60)

        print(
            f"Documents         : "
            f"{total_documents}"
        )

        print(
            f"Success           : "
            f"{success_count}"
        )

        print(
            f"Failed            : "
            f"{failed_count}"
        )

        print(
            f"Total             : "
            f"{batch_time:.3f}s"
        )

        print(
            f"Average           : "
            f"{average_time:.3f}s/document"
        )

        print("=" * 60)

        return results


# ============================================================
# CLI helper
# ============================================================


def print_document_result(
    result: dict,
) -> None:
    """
    Pretty-print document results.
    """

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(
        f"Document: "
        f"{result.get('document')}"
    )

    print(
        f"Type: "
        f"{result.get('document_type')}"
    )

    print(
        f"Pages: "
        f"{result.get('total_pages')}"
    )

    # ========================================================
    # Print every page
    # ========================================================

    for page in result.get(
        "pages",
        [],
    ):

        page_number = (
            page.get(
                "page_number"
            )
        )

        print()
        print("=" * 80)

        print(
            f"PAGE "
            f"{page_number}"
        )

        print("=" * 80)

        if not page.get(
            "success",
            False,
        ):

            print(
                f"FAILED: "
                f"{page.get('error')}"
            )

            continue

        page_result = (
            page[
                "result"
            ]
        )

        # ====================================================
        # Entities
        # ====================================================

        print()
        print(
            "FINAL ENTITIES"
        )
        print("-" * 80)

        print(
            json.dumps(
                page_result.get(
                    "entities",
                    [],
                ),
                ensure_ascii=False,
                indent=2,
            )
        )

        # ====================================================
        # Structured
        # ====================================================

        print()
        print(
            "STRUCTURED RESULT "
            "WITH CONFIDENCE"
        )
        print("-" * 80)

        print(
            json.dumps(
                page_result.get(
                    "structured_with_confidence",
                    {},
                ),
                ensure_ascii=False,
                indent=2,
            )
        )

        # ====================================================
        # Notifications
        # ====================================================

        notifications = (
            page_result.get(
                "notifications",
                [],
            )
        )

        if notifications:

            print()
            print(
                "REVIEW NOTIFICATIONS"
            )

            print("-" * 80)

            print(
                json.dumps(
                    notifications,
                    ensure_ascii=False,
                    indent=2,
                )
            )


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Document Understanding Pipeline: "
            "PDF/Image -> OCR -> "
            "LayoutXLM KIE"
        )
    )

    # ========================================================
    # Input
    # ========================================================

    parser.add_argument(
        "--input",
        "--image",
        dest="input",
        required=True,
        nargs="+",
        help=(
            "One or more PDF/image "
            "documents to process"
        ),
    )

    # ========================================================
    # KIE model
    # ========================================================

    parser.add_argument(
        "--kie-model",
        default=None,
        help=(
            "Optional local LayoutXLM model path. "
            "Default: download from Hugging Face"
        ),
    )

    # ========================================================
    # Device
    # ========================================================

    parser.add_argument(
        "--device",
        default=None,
        choices=[
            "cpu",
            "cuda",
            "mps",
        ],
        help=(
            "Inference device. "
            "Default: auto "
            "(cuda -> mps -> cpu)"
        ),
    )

    # ========================================================
    # Detection threshold
    # ========================================================

    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.1,
        help=(
            "Stage 2 text detection "
            "confidence threshold"
        ),
    )

    # ========================================================
    # Recognition batch size
    # ========================================================

    parser.add_argument(
        "--recognition-batch-size",
        type=int,
        default=32,
        help=(
            "VietOCR recognition "
            "batch size"
        ),
    )

    # ========================================================
    # PDF rendering
    # ========================================================

    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=200,
        help=(
            "PDF page rendering DPI. "
            "Default: 200"
        ),
    )

    # ========================================================
    # Orientation
    # ========================================================

    parser.add_argument(
        "--disable-orientation",
        action="store_true",
        help=(
            "Disable document "
            "orientation correction"
        ),
    )

    # ========================================================
    # Output
    # ========================================================

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON path"
        ),
    )

    # ========================================================
    # KIE debug
    # ========================================================

    parser.add_argument(
        "--kie-debug",
        action="store_true",
        help=(
            "Print LayoutXLM reading "
            "order and token predictions"
        ),
    )

    args = (
        parser.parse_args()
    )

    # ========================================================
    # Validate input
    # ========================================================

    for input_file in args.input:

        path = Path(
            input_file
        )

        if not path.exists():

            parser.error(
                f"Input not found: "
                f"{path}"
            )

        if (
            path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):

            parser.error(
                f"Unsupported input "
                f"type: "
                f"{path.suffix}"
            )

    # ========================================================
    # Create pipeline ONCE
    # ========================================================

    pipeline = (
        DocumentPipeline(
            kie_model_path=(
                args.kie_model
            ),
            device=(
                args.device
            ),
            score_thresh=(
                args.score_thresh
            ),
            kie_debug=(
                args.kie_debug
            ),
            recognition_batch_size=(
                args.recognition_batch_size
            ),
            enable_orientation=(
                not args.disable_orientation
            ),
            pdf_dpi=(
                args.pdf_dpi
            ),
        )
    )

    # ========================================================
    # ONE document
    # ========================================================

    if len(
        args.input
    ) == 1:

        result = (
            pipeline.process_document(
                file_path=(
                    args.input[0]
                ),
                pdf_dpi=(
                    args.pdf_dpi
                ),
            )
        )

        print_document_result(
            result
        )

    # ========================================================
    # MULTIPLE documents
    # ========================================================

    else:

        result = (
            pipeline.process_many(
                file_paths=(
                    args.input
                ),
                pdf_dpi=(
                    args.pdf_dpi
                ),
            )
        )

        print()
        print("=" * 80)

        print(
            "BATCH RESULTS"
        )

        print("=" * 80)

        for item in result:

            print()
            print("-" * 80)

            print(
                f"Document: "
                f"{item['document']}"
            )

            if item.get(
                "success",
                False,
            ):

                document_result = (
                    item[
                        "result"
                    ]
                )

                print(
                    f"Type: "
                    f"{document_result.get('document_type')}"
                )

                print(
                    f"Pages: "
                    f"{document_result.get('total_pages')}"
                )

                print(
                    f"Duration: "
                    f"{item['duration']:.3f}s"
                )

                for page in (
                    document_result.get(
                        "pages",
                        [],
                    )
                ):

                    page_number = (
                        page.get(
                            "page_number"
                        )
                    )

                    print()
                    print(
                        f"Page "
                        f"{page_number}:"
                    )

                    if not page.get(
                        "success",
                        False,
                    ):

                        print(
                            f"FAILED: "
                            f"{page.get('error')}"
                        )

                        continue

                    structured = (
                        page[
                            "result"
                        ].get(
                            "structured_with_confidence",
                            {},
                        )
                    )

                    print(
                        json.dumps(
                            structured,
                            ensure_ascii=False,
                            indent=2,
                        )
                    )

                    notifications = (
                        page[
                            "result"
                        ].get(
                            "notifications",
                            [],
                        )
                    )

                    if notifications:

                        print()
                        print(
                            "REVIEW NOTIFICATIONS"
                        )

                        print(
                            json.dumps(
                                notifications,
                                ensure_ascii=False,
                                indent=2,
                            )
                        )

            else:

                print(
                    f"FAILED: "
                    f"{item.get('error')}"
                )

    # ========================================================
    # Optional save
    # ========================================================

    if args.output:

        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print(
            f"Saved result: "
            f"{output_path}"
        )


# ============================================================
# Entry point
# ============================================================


if __name__ == "__main__":
    main()