# app/utils.py

import io
import json
import tempfile
from pathlib import Path

import pymupdf
import pandas as pd


ALL_LABELS = [
    "TEMPLATE_TYPE",
    "HOSPITAL_NAME",
    "PATIENT_NAME",
    "DATE_OF_BIRTH",
    "GENDER",
    "SOCIAL_INSURANCE_NUMBER",
    "HEALTH_INSURANCE_CARD",
    "MEDICAL_CERTIFICATE_NUMBER",
    "CITIZEN_ID",
    "DIAGNOSIS",
    "LEAVE_DURATION",
    "LEAVE_START_DATE",
    "LEAVE_END_DATE",
    "PARENT_NAME",
]

OPTIONAL_LABELS = {"PARENT_NAME"}

REQUIRED_LABELS = [
    label
    for label in ALL_LABELS
    if label not in OPTIONAL_LABELS
]

TEMPLATE_TYPES = [
    "",
    "GIẤY CHỨNG NHẬN NGHỈ VIỆC HƯỞNG BHXH",
    "GIẤY RA VIỆN",
    "GIẤY CHỨNG NHẬN",
    "OTHER",
]


# ============================================================
# PDF
# ============================================================

def pdf_to_images(file, dpi=200):
    doc = pymupdf.open(
        stream=file.getvalue(),
        filetype="pdf",
    )

    matrix = pymupdf.Matrix(
        dpi / 72,
        dpi / 72,
    )

    stem = Path(file.name).stem

    pages = [
        {
            "page_number": i + 1,
            "filename": f"{stem}_page_{i + 1:03d}.png",
            "image_bytes": doc[i]
            .get_pixmap(
                matrix=matrix,
                alpha=False,
            )
            .tobytes("png"),
        }
        for i in range(len(doc))
    ]

    doc.close()

    return pages


# ============================================================
# TEMP FILE
# ============================================================

def save_temp_image(
    image_bytes,
    suffix=".png",
):
    file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    file.write(image_bytes)
    file.close()

    return Path(file.name)


# ============================================================
# KIE ENTITIES
# ============================================================

def get_entities(result):
    if not isinstance(result, dict):
        return []

    if isinstance(
        result.get("entities"),
        list,
    ):
        return result["entities"]

    for key in (
        "kie",
        "kie_result",
    ):
        value = result.get(key)

        if (
            isinstance(value, dict)
            and isinstance(
                value.get("entities"),
                list,
            )
        ):
            return value["entities"]

    return []


def get_entity_map(result):
    return {
        entity["label"]: entity
        for entity in get_entities(result)
        if entity.get("label")
    }


def get_detected_template(result):
    return (
        get_entity_map(result)
        .get(
            "TEMPLATE_TYPE",
            {},
        )
        .get(
            "text",
            "",
        )
        .strip()
    )


# ============================================================
# REQUIRED FIELDS
# ============================================================

def get_missing_required(result):
    detected = {
        entity.get("label")
        for entity in get_entities(result)
        if str(
            entity.get(
                "text",
                "",
            )
        ).strip()
    }

    return [
        label
        for label in REQUIRED_LABELS
        if label not in detected
    ]


def validate_review(df):
    return [
        row["Field"]
        for _, row in df.iterrows()
        if row["Required"]
        and not str(
            row["Value"]
            if pd.notna(row["Value"])
            else ""
        ).strip()
    ]


# ============================================================
# QUALITY
# ============================================================

def get_quality(result):
    if not isinstance(result, dict):
        return "good", ""

    quality = result.get(
        "quality",
        {},
    )

    if not isinstance(
        quality,
        dict,
    ):
        return "good", ""

    return (
        str(
            quality.get(
                "status",
                "good",
            )
        ).lower(),
        str(
            quality.get(
                "notification",
                "",
            )
        ),
    )


# ============================================================
# PROCESSING ITEMS
# ============================================================

def build_processing_items(files):
    items = []

    for index, file in enumerate(files):
        extension = Path(
            file.name
        ).suffix.lower()

        if extension == ".pdf":
            pages = pdf_to_images(file)

            for page in pages:
                page_number = page[
                    "page_number"
                ]

                items.append(
                    {
                        "document_id":
                            f"{index}_"
                            f"{Path(file.name).stem}_"
                            f"page_{page_number:03d}",

                        "source_filename":
                            file.name,

                        "filename":
                            page["filename"],

                        "page_number":
                            page_number,

                        "source_type":
                            "pdf",

                        "image_bytes":
                            page["image_bytes"],

                        "extension":
                            ".png",
                    }
                )

        else:
            items.append(
                {
                    "document_id":
                        f"{index}_"
                        f"{Path(file.name).stem}",

                    "source_filename":
                        file.name,

                    "filename":
                        file.name,

                    "page_number":
                        None,

                    "source_type":
                        "image",

                    "image_bytes":
                        file.getvalue(),

                    "extension":
                        extension or ".png",
                }
            )

    return items


# ============================================================
# DISPLAY NAME
# ============================================================

def display_name(item):
    if item.get(
        "source_type"
    ) == "pdf":
        return (
            f"{item['source_filename']} "
            f"- Page {item['page_number']}"
        )

    return item["source_filename"]


# ============================================================
# REVIEW DATAFRAME
# ============================================================

def build_entity_df(
    result,
    reviewed_values=None,
):
    reviewed_values = (
        reviewed_values
        or {}
    )

    raw = get_entity_map(
        result
    )

    return pd.DataFrame(
        [
            {
                "Field":
                    label,

                "Value":
                    reviewed_values.get(
                        label,
                        raw.get(
                            label,
                            {},
                        ).get(
                            "text",
                            "",
                        ),
                    ),

                "Confidence":
                    float(
                        raw.get(
                            label,
                            {},
                        ).get(
                            "confidence",
                            0,
                        )
                        or 0
                    ),

                "Detected":
                    label in raw,

                "Required":
                    label
                    not in OPTIONAL_LABELS,
            }
            for label in ALL_LABELS
        ]
    )


# ============================================================
# FINAL VALUE
# ============================================================

def get_final_value(
    item,
    label,
    reviewed_entities,
    final_templates,
):
    doc_id = item[
        "document_id"
    ]

    raw = get_entity_map(
        item["result"]
    )

    reviewed = reviewed_entities.get(
        doc_id,
        {},
    )

    if label == "TEMPLATE_TYPE":
        return final_templates.get(
            doc_id,
            raw.get(
                label,
                {},
            ).get(
                "text",
                "",
            ),
        )

    return reviewed.get(
        label,
        raw.get(
            label,
            {},
        ).get(
            "text",
            "",
        ),
    )


# ============================================================
# EXPORT DATAFRAME
# ============================================================

def build_export_df(
    results,
    reviewed_documents,
    reviewed_entities,
    final_templates,
    review_notes,
):
    rows = []

    for item in results:
        doc_id = item[
            "document_id"
        ]

        row = {
            "source_filename":
                item["source_filename"],

            "page_number":
                item.get(
                    "page_number",
                    "",
                ),

            "review_status":
                (
                    "Completed"
                    if doc_id
                    in reviewed_documents
                    else "Not Reviewed"
                ),

            "note":
                review_notes.get(
                    doc_id,
                    "",
                ),
        }

        row.update(
            {
                label:
                    get_final_value(
                        item,
                        label,
                        reviewed_entities,
                        final_templates,
                    )
                for label in ALL_LABELS
            }
        )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ============================================================
# EXCEL
# ============================================================

def create_excel(
    results,
    reviewed_documents,
    reviewed_entities,
    final_templates,
    review_notes,
):
    df = build_export_df(
        results,
        reviewed_documents,
        reviewed_entities,
        final_templates,
        review_notes,
    )

    buffer = io.BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:
        df.to_excel(
            writer,
            sheet_name="Extracted Data",
            index=False,
        )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# JSON
# ============================================================

def create_json(
    results,
    reviewed_documents,
    reviewed_entities,
    final_templates,
    review_notes,
):
    output = []

    for item in results:
        doc_id = item[
            "document_id"
        ]

        output.append(
            {
                "document_id":
                    doc_id,

                "source_filename":
                    item[
                        "source_filename"
                    ],

                "page_number":
                    item.get(
                        "page_number"
                    ),

                "review_status":
                    (
                        "Completed"
                        if doc_id
                        in reviewed_documents
                        else "Not Reviewed"
                    ),

                "final_template_type":
                    get_final_value(
                        item,
                        "TEMPLATE_TYPE",
                        reviewed_entities,
                        final_templates,
                    ),

                "note":
                    review_notes.get(
                        doc_id,
                        "",
                    ),

                "entities":
                    {
                        label:
                            get_final_value(
                                item,
                                label,
                                reviewed_entities,
                                final_templates,
                            )
                        for label
                        in ALL_LABELS
                    },

                "pipeline_result":
                    item[
                        "result"
                    ],
            }
        )

    return json.dumps(
        output,
        ensure_ascii=False,
        indent=2,
        default=str,
    )