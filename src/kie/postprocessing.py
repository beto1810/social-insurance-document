from __future__ import annotations

import re
import unicodedata
from datetime import date
from functools import lru_cache


# ============================================================
# Configuration
# ============================================================

NAME_LABELS = {"PATIENT_NAME", "PARENT_NAME"}

DATE_LABELS = {
    "DATE_OF_BIRTH",
    "EXAMINATION_DATE",
    "LEAVE_START_DATE",
    "LEAVE_END_DATE",
}

VIETNAMESE_SURNAMES = {
    "nguyen": "Nguyễn",
    "tran": "Trần",
    "le": "Lê",
    "pham": "Phạm",
    "hoang": "Hoàng",
    "huynh": "Huỳnh",
    "phan": "Phan",
    "vu": "Vũ",
    "vo": "Võ",
    "dang": "Đặng",
    "bui": "Bùi",
    "do": "Đỗ",
    "ho": "Hồ",
    "ngo": "Ngô",
    "duong": "Dương",
    "ly": "Lý",
    "dinh": "Đinh",
    "truong": "Trương",
    "mai": "Mai",
    "doan": "Đoàn",
    "lam": "Lâm",
    "trinh": "Trịnh",
    "dao": "Đào",
    "cao": "Cao",
    "chau": "Châu",
    "luu": "Lưu",
    "luong": "Lương",
    "thai": "Thái",
    "ta": "Tạ",
    "tong": "Tống",
    "ton": "Tôn",
    "quach": "Quách",
}

START_DATE_ANCHOR_SCORES = {
    "vao vien luc": 1.00,
    "ngay vao vien": 1.00,
    "vao vien ngay": 1.00,
    "nhap vien luc": 0.98,
    "nhap vien ngay": 0.98,
    "ngay nhap vien": 0.98,
    "dieu tri tu ngay": 0.95,
    "nghi tu ngay": 0.95,
    "nghi viec tu ngay": 0.98,
    "tu ngay": 0.78,
}

END_DATE_ANCHOR_SCORES = {
    "ra vien luc": 1.00,
    "ngay ra vien": 1.00,
    "ra vien ngay": 1.00,
    "xuat vien luc": 0.98,
    "xuat vien ngay": 0.98,
    "ngay xuat vien": 0.98,
    "dieu tri den ngay": 0.95,
    "nghi den ngay": 0.95,
    "nghi viec den ngay": 0.98,
    "den ngay": 0.78,
}

FIELD_LENGTH_RULES = {
    "CITIZEN_ID": ("digits", (9, 12)),
    "HEALTH_INSURANCE_CARD": ("alphanumeric", (10, 15)),
    "SOCIAL_INSURANCE_NUMBER": ("digits", (10,)),
}


# ============================================================
# General text helpers
# ============================================================

def _normalize_for_match(text: str) -> str:
    text = str(text or "").strip().upper()

    # Remove Vietnamese accents for easier keyword matching
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )

    return re.sub(r"\s+", " ", text)

def remove_vietnamese_accents(text: str) -> str:
    if not text:
        return ""
    text = text.replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


@lru_cache(maxsize=4096)
def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.lower().split())


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text.lower().strip())


@lru_cache(maxsize=4096)
def remove_all_spaces(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


@lru_cache(maxsize=4096)
def compact_code(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text or "").upper()


# ============================================================
# Final normalization
# ============================================================

def preserve_word_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original and original[0].isupper():
        return replacement
    return replacement


def normalize_vietnamese_surname(text: str) -> str:
    if not text:
        return text

    text = str(text).strip()
    match = re.match(r"^(\S+)(.*)$", text)
    if not match:
        return text

    first_word, remaining = match.groups()
    clean = re.sub(r"^[^\wÀ-ỹĐđ]+|[^\wÀ-ỹĐđ]+$", "", first_word)
    if not clean:
        return text

    surname = VIETNAMESE_SURNAMES.get(remove_vietnamese_accents(clean))
    if surname is None:
        return text

    normalized = preserve_word_case(clean, surname)
    return first_word.replace(clean, normalized, 1) + remaining


def normalize_template_type(text: str) -> str:
    if not text:
        return text

    match = re.match(r"^(\s*)(\S+)(.*)$", str(text))
    if not match:
        return text

    leading, first_word, rest = match.groups()
    if remove_vietnamese_accents(first_word) != "nghi":
        return text

    return leading + preserve_word_case(first_word, "Nghỉ") + rest


FINAL_DATE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?P<day>\d{1,2})\s*[/\-.\\|]\s*"
    r"(?P<month>\d{1,2})\s*[/\-.\\|]\s*"
    r"(?P<year>\d{4})(?!\d)"
)

VIETNAMESE_DATE_PATTERN = re.compile(
    r"(?:ng[aà]y\s*)?"
    r"(?P<day>\d{1,2})\s*"
    r"th[aá]ng\s*"
    r"(?P<month>\d{1,2})\s*"
    r"n[aă]m\s*"
    r"(?P<year>\d{4})",
    flags=re.IGNORECASE,
)


def normalize_date_value(text: str) -> str:
    if not text:
        return text

    raw = str(text).strip()
    match = FINAL_DATE_PATTERN.search(raw) or VIETNAMESE_DATE_PATTERN.search(raw)
    if not match:
        return raw

    try:
        parsed = date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except (ValueError, TypeError):
        return raw

    return f"{parsed.day:02d}/{parsed.month:02d}/{parsed.year:04d}"


def normalize_entity(entity: dict, debug: bool = False) -> dict:
    result = entity.copy()
    label = result.get("label")
    original = str(result.get("text", ""))

    if not original:
        return result

    normalized = original

    if label in NAME_LABELS:
        normalized = normalize_vietnamese_surname(original)
    elif label == "TEMPLATE_TYPE":
        normalized = normalize_template_type(original)
    elif label in DATE_LABELS:
        normalized = normalize_date_value(original)

    result["text"] = normalized

    if debug and normalized != original:
        print(f"[NORMALIZE] {label}: {original!r} -> {normalized!r}")

    return result


def normalize_entities(entities: list[dict], debug: bool = False) -> list[dict]:
    return [normalize_entity(entity, debug) for entity in entities]


# ============================================================
# Bounding box helpers
# ============================================================

def bbox_from_points(points: list[list[float]]) -> list[int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def prepare_regions(ocr_regions: list[dict]) -> list[dict]:
    regions = []

    for index, region in enumerate(ocr_regions):
        text = str(region.get("text", "")).strip()
        points = region.get("points")

        if not text or not points:
            continue

        regions.append(
            {
                "index": index,
                "text": text,
                "bbox": bbox_from_points(points),
            }
        )

    return regions


def vertical_overlap(box1: list[int], box2: list[int]) -> float:
    _, y1_min, _, y1_max = box1
    _, y2_min, _, y2_max = box2

    overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    height1 = max(1, y1_max - y1_min)
    height2 = max(1, y2_max - y2_min)

    return overlap / min(height1, height2)


def horizontal_distance(box1: list[int], box2: list[int]) -> int:
    x1_min, _, x1_max, _ = box1
    x2_min, _, x2_max, _ = box2

    if x2_min > x1_max:
        return x2_min - x1_max
    if x1_min > x2_max:
        return x1_min - x2_max
    return 0


def is_same_text_line(
    box1: list[int],
    box2: list[int],
    min_vertical_overlap: float = 0.35,
    max_center_distance: int = 20,
) -> bool:
    if vertical_overlap(box1, box2) >= min_vertical_overlap:
        return True

    _, y1_min, _, y1_max = box1
    _, y2_min, _, y2_max = box2

    height = min(y1_max - y1_min, y2_max - y2_min)
    center_distance = abs(
        (y1_min + y1_max) / 2 - (y2_min + y2_max) / 2
    )

    allowed = min(max_center_distance, max(2.0, height * 0.5))
    return center_distance <= allowed


def bbox_center_y(bbox: list[int]) -> float:
    return (bbox[1] + bbox[3]) / 2.0


def bbox_height(bbox: list[int]) -> float:
    return max(1.0, bbox[3] - bbox[1])


# ============================================================
# Insurance helpers
# ============================================================

def is_bhxh_anchor(text: str) -> bool:
    text = normalize_text(text)
    return "bhxh" in text and "bhyt" not in text and "ma the" not in text


def is_health_card_anchor(text: str) -> bool:
    text = normalize_text(text)
    return any(
        pattern in text
        for pattern in ("ma the", "the bhyt", "so the bhyt")
    )


def extract_anchor_value(text: str) -> str:
    match = re.search(
        r"(?:s[oố]\s*th[ẻe]\s*bhyt|th[ẻe]\s*bhyt\s*s[oố])"
        r"\s*[:\-]?\s*([A-Za-z]{1,4}\s*\d[\d\s]*)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return remove_all_spaces(match.group(1))

    if ":" not in text:
        return ""

    return text.split(":", 1)[1].strip()


def extract_bhxh_value(text: str) -> str:
    match = re.search(
        r"(?:m[aãà]\s*s[oố]\s*bhxh|bhxh)\s*[:\-]?\s*(\d{10})",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else ""


def classify_insurance_value(value: str) -> str:
    value = remove_all_spaces(value)

    if value.isdigit() and len(value) == 10 and value.startswith("7"):
        return "SOCIAL_INSURANCE_NUMBER"

    return "HEALTH_INSURANCE_CARD"


def looks_like_health_card_fragment(text: str) -> bool:
    raw = text.strip()
    if not raw:
        return False

    normalized = normalize_text(raw)

    forbidden = (
        "bhxh",
        "cccd",
        "cmnd",
        "ho chieu",
        "ngay sinh",
        "gioi tinh",
        "ho va ten",
        "ho ten",
        "ngay cap",
        "don vi",
        "chan doan",
    )

    if any(pattern in normalized for pattern in forbidden):
        return False

    if re.fullmatch(r"[A-Za-z]{1,4}", raw):
        return True

    if re.fullmatch(r"\d{1,15}", raw):
        return True

    compact = compact_code(raw)

    return (
        10 <= len(compact) <= 15
        and re.fullmatch(r"[A-Z0-9]+", compact) is not None
        and any(c.isdigit() for c in compact)
    )


def find_health_card_regions(
    anchor: dict,
    regions: list[dict],
    max_vertical_overlap_distance: int = 20,
    max_neighbor_gap: int = 220,
    min_vertical_overlap: float = 0.35,
) -> list[dict]:

    candidate_regions = []

    for region in regions:
        if region["index"] == anchor["index"]:
            continue

        if is_bhxh_anchor(region["text"]) or is_health_card_anchor(region["text"]):
            continue

        if not looks_like_health_card_fragment(region["text"]):
            continue

        if not is_same_text_line(
            anchor["bbox"],
            region["bbox"],
            min_vertical_overlap,
            max_vertical_overlap_distance,
        ):
            continue

        candidate_regions.append(region)

    row_regions = [anchor, *candidate_regions]
    row_regions.sort(key=lambda item: item["bbox"][0])

    anchor_position = next(
        (
            i
            for i, region in enumerate(row_regions)
            if region["index"] == anchor["index"]
        ),
        None,
    )

    if anchor_position is None:
        return [anchor]

    selected = [anchor]
    previous_box = anchor["bbox"]

    for region in reversed(row_regions[:anchor_position]):
        if horizontal_distance(previous_box, region["bbox"]) > max_neighbor_gap:
            break
        selected.append(region)
        previous_box = region["bbox"]

    previous_box = anchor["bbox"]

    for region in row_regions[anchor_position + 1 :]:
        if horizontal_distance(previous_box, region["bbox"]) > max_neighbor_gap:
            break
        selected.append(region)
        previous_box = region["bbox"]

    selected.sort(key=lambda item: item["bbox"][0])
    return selected


def build_health_card_value(anchor: dict, field_regions: list[dict]) -> str:
    values = []

    for region in sorted(field_regions, key=lambda item: item["bbox"][0]):
        text = region["text"].strip()

        if region["index"] == anchor["index"]:
            value = extract_anchor_value(text)
            if value:
                values.append(value)
            continue

        if is_bhxh_anchor(text):
            continue

        if looks_like_health_card_fragment(text):
            values.append(text)

    return remove_all_spaces("".join(values))

def clean_hospital_name_entities(entities: list[dict]) -> list[dict]:
    """
    Keep only the most likely actual hospital name.

    Example:
        BỆNH VIỆN ĐA KHOA VIỆT ĐỨC   -> keep
        Khoa Chấn thương chỉnh hình  -> remove
    """

    hospital_entities = [
        e for e in entities
        if e.get("label") == "HOSPITAL_NAME"
    ]

    if len(hospital_entities) <= 1:
        return entities

    hospital_keywords = (
        "BENH VIEN",
        "BV ",
        "BV.",
        "HOSPITAL",
        "TRUNG TAM Y TE",
        "TTYT",
        "PHONG KHAM",
    )

    department_keywords = (
        "KHOA ",
        "PHONG ",
        "BAN ",
        "DON VI ",
        "TRUNG TAM CHAN THUONG",
    )

    def hospital_score(entity: dict) -> float:
        text = _normalize_for_match(entity.get("text", ""))
        confidence = float(entity.get("confidence", 0.0))

        score = confidence

        # Strong signal: actual institution name
        if any(keyword in text for keyword in hospital_keywords):
            score += 2.0

        # Strong negative signal: department / unit
        if any(text.startswith(keyword) for keyword in department_keywords):
            score -= 2.0

        return score

    best_hospital = max(
        hospital_entities,
        key=hospital_score,
    )

    result = [
        e for e in entities
        if e.get("label") != "HOSPITAL_NAME"
    ]

    result.append(best_hospital)

    # Keep original entity order
    result.sort(
        key=lambda e: min(e.get("region_indexes", [999999]))
    )

    return result

def remove_consumed_entities(
    entities: list[dict],
    consumed_region_indexes: set[int],
) -> list[dict]:
    return [
        entity
        for entity in entities
        if not (
            set(entity.get("region_indexes", []))
            & consumed_region_indexes
        )
    ]


# ============================================================
# Leave date
# ============================================================

DATE_PATTERN = re.compile(
    r"(?:ng[aà]y\s*)?"
    r"(?P<day>\d{1,2})"
    r"\s*(?:/|-|th[aá]ng\s+)"
    r"(?P<month>\d{1,2})"
    r"\s*(?:/|-|n[aă]m\s+)"
    r"(?P<year>\d{4})",
    flags=re.IGNORECASE,
)


def extract_dates(text: str) -> list[dict]:
    results = []

    for match in DATE_PATTERN.finditer(text):
        try:
            parsed = date(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            )
        except ValueError:
            continue

        results.append(
            {
                "text": (
                    f"{parsed.day:02d}/"
                    f"{parsed.month:02d}/"
                    f"{parsed.year:04d}"
                ),
                "start": match.start(),
                "end": match.end(),
            }
        )

    return results


def get_template_type(entities: list[dict]) -> str:
    for entity in entities:
        if entity.get("label") == "TEMPLATE_TYPE":
            return normalize_text(entity.get("text", ""))
    return ""


def get_existing_entity(
    entities: list[dict],
    label: str,
) -> dict | None:

    candidates = [
        entity
        for entity in entities
        if entity.get("label") == label
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda entity: float(entity.get("confidence", 0.0)),
    )


def choose_date_after_anchor(
    text: str,
    normalized_text: str,
    normalized_anchor: str,
    dates: list[dict],
) -> dict | None:

    if not dates:
        return None

    if len(dates) == 1:
        return dates[0]

    anchor = normalize_text(normalized_anchor)

    if (
        "tu ngay" in anchor
        or "vao vien" in anchor
        or "nhap vien" in anchor
    ):
        return dates[0]

    if (
        "den ngay" in anchor
        or "ra vien" in anchor
        or "xuat vien" in anchor
    ):
        return dates[-1]

    return dates[0]


def find_leave_date_candidates(
    entities: list[dict],
    regions: list[dict],
    label: str,
    debug: bool = False,
) -> list[dict]:

    if label == "LEAVE_START_DATE":
        anchor_scores = START_DATE_ANCHOR_SCORES
    elif label == "LEAVE_END_DATE":
        anchor_scores = END_DATE_ANCHOR_SCORES
    else:
        return []

    existing = get_existing_entity(entities, label)
    existing_regions = set(existing.get("region_indexes", [])) if existing else set()
    existing_confidence = float(existing.get("confidence", 0.0)) if existing else 0.0
    template_type = get_template_type(entities)

    candidates = []

    for region in regions:
        raw_text = region["text"]
        normalized = normalize_text(raw_text)
        dates = extract_dates(raw_text)

        if not dates:
            continue

        for anchor, anchor_score in anchor_scores.items():
            if anchor not in normalized:
                continue

            selected = choose_date_after_anchor(
                raw_text,
                normalized,
                anchor,
                dates,
            )

            if not selected:
                continue

            score = anchor_score

            if region["index"] in existing_regions:
                score += 0.08 * existing_confidence

            if "giay ra vien" in template_type:
                if (
                    label == "LEAVE_START_DATE"
                    and ("vao vien" in anchor or "nhap vien" in anchor)
                ):
                    score += 0.05

                if (
                    label == "LEAVE_END_DATE"
                    and ("ra vien" in anchor or "xuat vien" in anchor)
                ):
                    score += 0.05

            score = min(score, 1.0)

            candidates.append(
                {
                    "label": label,
                    "text": selected["text"],
                    "confidence": round(score, 4),
                    "source": "semantic_date_postprocess",
                    "region_indexes": [region["index"]],
                    "_anchor": anchor,
                    "_score": score,
                    "_vertical_distance": 0.0,
                }
            )

    candidates.sort(
        key=lambda item: (
            -item["_score"],
            item["_vertical_distance"],
        )
    )

    return candidates


def find_spatial_leave_date_candidates(
    entities: list[dict],
    regions: list[dict],
    label: str,
    debug: bool = False,
) -> list[dict]:
    """
    Recover a date when anchor/time/date are separate OCR boxes.

    Example:
        Vào viện lúc:   18 giờ 03 phút   Ngày 29 tháng 07 năm 2026
        Ra viện lúc:    10 giờ 30 phút   Ngày 31 tháng 07 năm 2026
    """

    if label == "LEAVE_START_DATE":
        anchor_scores = START_DATE_ANCHOR_SCORES
    elif label == "LEAVE_END_DATE":
        anchor_scores = END_DATE_ANCHOR_SCORES
    else:
        return []

    template_type = get_template_type(entities)
    candidates = []

    for anchor_region in regions:
        anchor_text = normalize_text(anchor_region["text"])
        matched_anchor = None
        matched_score = 0.0

        for anchor_pattern, anchor_score in anchor_scores.items():
            if (
                anchor_pattern in anchor_text
                and anchor_score > matched_score
            ):
                matched_anchor = anchor_pattern
                matched_score = anchor_score

        if matched_anchor is None:
            continue

        anchor_box = anchor_region["bbox"]
        anchor_cy = bbox_center_y(anchor_box)
        anchor_h = bbox_height(anchor_box)

        for date_region in regions:
            if date_region["index"] == anchor_region["index"]:
                continue

            dates = extract_dates(date_region["text"])
            if not dates:
                continue

            date_box = date_region["bbox"]
            date_cy = bbox_center_y(date_box)
            date_h = bbox_height(date_box)

            vertical_distance = abs(date_cy - anchor_cy)

            # Main rule:
            # same row / very close row.
            allowed_vertical_distance = max(
                anchor_h,
                date_h,
            ) * 2.0

            if vertical_distance > allowed_vertical_distance:
                continue

            anchor_x1, _, anchor_x2, _ = anchor_box
            date_x1 = date_box[0]

            # Do not limit large right-side gaps.
            if date_x1 >= anchor_x2:
                horizontal_penalty = 0.0
            elif date_x1 >= anchor_x1:
                horizontal_penalty = 0.02
            else:
                horizontal_penalty = 0.08

            relative_vertical = (
                vertical_distance
                / max(anchor_h, date_h, 1.0)
            )

            vertical_penalty = min(
                relative_vertical * 0.04,
                0.12,
            )

            score = (
                matched_score
                - vertical_penalty
                - horizontal_penalty
            )

            if "giay ra vien" in template_type:
                if (
                    label == "LEAVE_START_DATE"
                    and (
                        "vao vien" in matched_anchor
                        or "nhap vien" in matched_anchor
                    )
                ):
                    score += 0.05

                if (
                    label == "LEAVE_END_DATE"
                    and (
                        "ra vien" in matched_anchor
                        or "xuat vien" in matched_anchor
                    )
                ):
                    score += 0.05

            score = max(0.0, min(score, 1.0))

            for extracted_date in dates:
                candidates.append(
                    {
                        "label": label,
                        "text": extracted_date["text"],
                        "confidence": round(score, 4),
                        "source": "spatial_leave_date_postprocess",
                        "region_indexes": [
                            anchor_region["index"],
                            date_region["index"],
                        ],
                        "_anchor": matched_anchor,
                        "_score": score,
                        "_vertical_distance": vertical_distance,
                    }
                )

    candidates.sort(
        key=lambda item: (
            -item["_score"],
            item["_vertical_distance"],
        )
    )

    if debug and candidates:
        print(f"[POSTPROCESS] {label} spatial candidates:")

        for candidate in candidates[:5]:
            print(
                "  ",
                candidate["text"],
                f"score={candidate['_score']:.4f}",
                f"anchor={candidate['_anchor']}",
                f"regions={candidate['region_indexes']}",
                f"y_gap={candidate['_vertical_distance']:.1f}",
            )

    return candidates


def replace_label_entities(
    entities: list[dict],
    label: str,
    replacement: dict,
) -> list[dict]:

    cleaned = [
        entity
        for entity in entities
        if entity.get("label") != label
    ]

    replacement = {
        key: value
        for key, value in replacement.items()
        if not key.startswith("_")
    }

    cleaned.append(replacement)
    return cleaned


def resolve_leave_dates(
    entities: list[dict],
    regions: list[dict],
    debug: bool = False,
    strong_threshold: float = 0.90,
) -> list[dict]:

    start_candidates = find_leave_date_candidates(
        entities,
        regions,
        "LEAVE_START_DATE",
        debug,
    )

    end_candidates = find_leave_date_candidates(
        entities,
        regions,
        "LEAVE_END_DATE",
        debug,
    )

    # Spatial fallback:
    # anchor and date can be separate OCR regions.
    start_candidates.extend(
        find_spatial_leave_date_candidates(
            entities,
            regions,
            "LEAVE_START_DATE",
            debug,
        )
    )

    end_candidates.extend(
        find_spatial_leave_date_candidates(
            entities,
            regions,
            "LEAVE_END_DATE",
            debug,
        )
    )

    start_candidates.sort(
        key=lambda item: (
            -item["_score"],
            item.get("_vertical_distance", 0),
        )
    )

    end_candidates.sort(
        key=lambda item: (
            -item["_score"],
            item.get("_vertical_distance", 0),
        )
    )

    best_start = start_candidates[0] if start_candidates else None
    best_end = end_candidates[0] if end_candidates else None

    # Validate chronology.
    if (
        best_start
        and best_end
        and best_start["_score"] >= strong_threshold
        and best_end["_score"] >= strong_threshold
    ):
        try:
            sd, sm, sy = map(int, best_start["text"].split("/"))
            ed, em, ey = map(int, best_end["text"].split("/"))

            if (sy, sm, sd) > (ey, em, ed):
                if debug:
                    print(
                        "[POSTPROCESS] Reject leave dates:",
                        best_start["text"],
                        ">",
                        best_end["text"],
                    )

                best_start = None
                best_end = None

        except ValueError:
            pass

    if best_start and best_start["_score"] >= strong_threshold:
        entities = replace_label_entities(
            entities,
            "LEAVE_START_DATE",
            best_start,
        )

    if best_end and best_end["_score"] >= strong_threshold:
        entities = replace_label_entities(
            entities,
            "LEAVE_END_DATE",
            best_end,
        )

    return entities


# ============================================================
# Anchor-based recovery
# ============================================================

def _entity_exists(entities: list[dict], label: str) -> bool:
    return any(entity.get("label") == label for entity in entities)


def _clean_leading_field_punctuation(text: str) -> str:
    return re.sub(
        r"^[\s:;,.\-–—|]+",
        "",
        str(text or ""),
    ).strip()


def _is_reasonable_person_name(text: str) -> bool:
    value = _clean_leading_field_punctuation(text)

    if not value:
        return False

    normalized = normalize_text(value)

    forbidden = (
        "ngay sinh",
        "ngay/thang/nam",
        "gioi tinh",
        "dan toc",
        "nghe nghiep",
        "dia chi",
        "cccd",
        "cmnd",
        "bhxh",
        "bhyt",
        "chan doan",
        "vao vien",
        "ra vien",
    )

    if any(token in normalized for token in forbidden):
        return False

    words = value.split()

    if not 2 <= len(words) <= 7:
        return False

    letters = sum(c.isalpha() for c in value)
    visible = sum(not c.isspace() for c in value)

    return visible > 0 and letters / visible >= 0.65


def _nearby_value_candidates(
    anchor: dict,
    regions: list[dict],
    max_right_gap: int = 900,
    max_vertical_gap: int = 90,
) -> list[dict]:

    ax1, ay1, ax2, ay2 = anchor["bbox"]
    anchor_cy = (ay1 + ay2) / 2
    anchor_height = max(1, ay2 - ay1)

    candidates = []

    for region in regions:
        if region["index"] == anchor["index"]:
            continue

        x1, y1, x2, y2 = region["bbox"]
        cy = (y1 + y2) / 2

        same_line = is_same_text_line(
            anchor["bbox"],
            region["bbox"],
            min_vertical_overlap=0.25,
            max_center_distance=max(
                20,
                int(anchor_height * 0.8),
            ),
        )

        if same_line:
            if x1 >= ax2:
                gap = x1 - ax2

                if gap <= max_right_gap:
                    candidates.append(
                        (0, gap, abs(cy - anchor_cy), region)
                    )

            elif horizontal_distance(
                anchor["bbox"],
                region["bbox"],
            ) == 0:
                candidates.append(
                    (1, 0, abs(cy - anchor_cy), region)
                )

            continue

        vertical_gap = y1 - ay2

        if (
            0 <= vertical_gap <= max_vertical_gap
            and x2 >= ax1 - 80
            and x1 <= ax2 + max_right_gap
        ):
            candidates.append(
                (2, vertical_gap, abs(x1 - ax1), region)
            )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
            item[3]["bbox"][0],
        )
    )

    return [item[3] for item in candidates]


def recover_patient_name_from_ocr(
    entities: list[dict],
    regions: list[dict],
    debug: bool = False,
) -> list[dict]:

    if _entity_exists(entities, "PATIENT_NAME"):
        return entities

    anchor_patterns = (
        "ho ten nguoi benh",
        "ho ten benh nhan",
        "ho va ten nguoi benh",
        "ho va ten benh nhan",
    )

    for anchor in regions:
        normalized = normalize_text(anchor["text"])

        if not any(pattern in normalized for pattern in anchor_patterns):
            continue

        if ":" in anchor["text"]:
            value = _clean_leading_field_punctuation(
                anchor["text"].split(":", 1)[1]
            )

            if _is_reasonable_person_name(value):
                return [
                    *entities,
                    {
                        "label": "PATIENT_NAME",
                        "text": value,
                        "confidence": 1.0,
                        "source": "patient_name_anchor_postprocess",
                        "region_indexes": [anchor["index"]],
                    },
                ]

        for candidate in _nearby_value_candidates(anchor, regions):
            value = _clean_leading_field_punctuation(candidate["text"])

            if not _is_reasonable_person_name(value):
                continue

            return [
                *entities,
                {
                    "label": "PATIENT_NAME",
                    "text": value,
                    "confidence": 1.0,
                    "source": "patient_name_anchor_postprocess",
                    "region_indexes": [candidate["index"]],
                },
            ]

    return entities


def recover_citizen_id_from_ocr(
    entities: list[dict],
    regions: list[dict],
    debug: bool = False,
) -> list[dict]:

    if _entity_exists(entities, "CITIZEN_ID"):
        return entities

    anchor_patterns = (
        "cccd",
        "cmnd",
        "dinh danh cong dan",
        "can cuoc cong dan",
        "ho chieu",
    )

    for anchor in regions:
        normalized = normalize_text(anchor["text"])

        if not any(pattern in normalized for pattern in anchor_patterns):
            continue

        inline_digits = re.sub(r"\D", "", anchor["text"])

        if len(inline_digits) in {9, 12}:
            return [
                *entities,
                {
                    "label": "CITIZEN_ID",
                    "text": inline_digits,
                    "confidence": 1.0,
                    "source": "citizen_id_anchor_postprocess",
                    "region_indexes": [anchor["index"]],
                },
            ]

        for candidate in _nearby_value_candidates(anchor, regions):
            value = re.sub(r"\D", "", candidate["text"])

            if len(value) not in {9, 12}:
                continue

            return [
                *entities,
                {
                    "label": "CITIZEN_ID",
                    "text": value,
                    "confidence": 1.0,
                    "source": "citizen_id_anchor_postprocess",
                    "region_indexes": [candidate["index"]],
                },
            ]

    return entities


def remove_invalid_date_entities(
    entities: list[dict],
    debug: bool = False,
) -> list[dict]:

    result = []

    for entity in entities:
        label = entity.get("label")

        if label not in DATE_LABELS:
            result.append(entity)
            continue

        original = str(entity.get("text", "")).strip()
        normalized = normalize_date_value(original)

        if re.fullmatch(r"\d{2}/\d{2}/\d{4}", normalized):
            entity = entity.copy()
            entity["text"] = normalized
            result.append(entity)

        elif debug:
            print(
                "[POSTPROCESS] Dropped invalid date:",
                label,
                repr(original),
            )

    return result


# ============================================================
# Diagnosis
# ============================================================

def _find_prepared_region(
    regions: list[dict],
    region_index: int,
) -> dict | None:

    for region in regions:
        if region["index"] == region_index:
            return region

    if 0 <= region_index < len(regions):
        return regions[region_index]

    return None


def merge_diagnosis_entities(
    entities: list[dict],
    regions: list[dict],
    debug: bool = False,
) -> list[dict]:

    diagnosis_entities = [
        entity
        for entity in entities
        if entity.get("label") == "DIAGNOSIS"
    ]

    if not diagnosis_entities:
        return entities

    recovered_regions = {}

    for entity in diagnosis_entities:
        confidence = float(entity.get("confidence", 0.0) or 0.0)

        for region_index in entity.get("region_indexes", []):
            if not isinstance(region_index, int):
                continue

            region = _find_prepared_region(
                regions,
                region_index,
            )

            if not region:
                continue

            text = str(region.get("text", "")).strip()
            bbox = region.get("bbox")

            if not text or not bbox:
                continue

            index = int(region.get("index", region_index))
            previous = recovered_regions.get(index)

            if previous is None or confidence > previous["confidence"]:
                recovered_regions[index] = {
                    "index": index,
                    "text": text,
                    "bbox": bbox,
                    "confidence": confidence,
                }

    if not recovered_regions:
        return entities

    diagnosis_regions = list(recovered_regions.values())

    diagnosis_regions.sort(
        key=lambda item: (
            (item["bbox"][1] + item["bbox"][3]) / 2,
            item["bbox"][0],
        )
    )

    merged_text = re.sub(
        r"\s+",
        " ",
        " ".join(item["text"] for item in diagnosis_regions),
    ).strip()

    total_weight = 0
    weighted_sum = 0.0

    for item in diagnosis_regions:
        weight = max(1, len(item["text"]))
        total_weight += weight
        weighted_sum += item["confidence"] * weight

    merged_confidence = (
        weighted_sum / total_weight
        if total_weight
        else 0.0
    )

    merged = {
        "label": "DIAGNOSIS",
        "text": merged_text,
        "confidence": round(merged_confidence, 4),
        "source": "diagnosis_ocr_region_postprocess",
        "region_indexes": [
            item["index"]
            for item in diagnosis_regions
        ],
    }

    result = []
    inserted = False

    for entity in entities:
        if entity.get("label") == "DIAGNOSIS":
            if not inserted:
                result.append(merged)
                inserted = True
            continue

        result.append(entity)

    return result


# ============================================================
# Validation notifications
# ============================================================

def build_validation_notifications(
    entities: list[dict],
) -> list[dict]:

    notifications = []

    for entity_index, entity in enumerate(entities):
        label = entity.get("label")
        rule = FIELD_LENGTH_RULES.get(label)

        if not rule:
            continue

        text = str(entity.get("text", ""))
        value_type, expected_lengths = rule

        if value_type == "digits":
            value = re.sub(r"\D", "", text)
        else:
            value = re.sub(r"[^A-Za-z0-9]", "", text)

        actual_length = len(value)

        if actual_length in expected_lengths:
            continue

        expected_text = " or ".join(
            str(length)
            for length in expected_lengths
        )

        notifications.append(
            {
                "type": "review_required",
                "field": label,
                "entity_index": entity_index,
                "text": text,
                "actual_length": actual_length,
                "expected_lengths": list(expected_lengths),
                "message": (
                    f"Review {label}: expected {expected_text} "
                    f"characters, found {actual_length}."
                ),
            }
        )

    return notifications


# ============================================================
# Main
# ============================================================

def merge_related_entities(
    entities: list[dict],
    ocr_regions: list[dict],
    debug: bool = False,
) -> list[dict]:

    regions = prepare_regions(ocr_regions)

    # --------------------------------------------------------
    # 1. Insurance reconstruction
    # --------------------------------------------------------

    health_card_anchors = [
        region
        for region in regions
        if is_health_card_anchor(region["text"])
    ]

    consumed_region_indexes = set()
    new_entities = []

    for anchor in health_card_anchors:
        field_regions = find_health_card_regions(
            anchor,
            regions,
        )

        health_card_value = build_health_card_value(
            anchor,
            field_regions,
        )

        bhxh_value = extract_bhxh_value(
            anchor["text"]
        )

        if not health_card_value and not bhxh_value:
            continue

        region_indexes = []

        for region in field_regions:
            if region["index"] == anchor["index"]:
                consumed_region_indexes.add(region["index"])
                region_indexes.append(region["index"])
                continue

            if is_bhxh_anchor(region["text"]):
                continue

            if looks_like_health_card_fragment(region["text"]):
                consumed_region_indexes.add(region["index"])
                region_indexes.append(region["index"])

        region_indexes = sorted(set(region_indexes))

        if bhxh_value:
            new_entities.append(
                {
                    "label": "SOCIAL_INSURANCE_NUMBER",
                    "text": bhxh_value,
                    "confidence": 1.0,
                    "source": "spatial_postprocess",
                    "region_indexes": [anchor["index"]],
                }
            )

        if health_card_value:
            health_card_value = compact_code(
                health_card_value
            )

            new_entities.append(
                {
                    "label": classify_insurance_value(
                        health_card_value
                    ),
                    "text": health_card_value,
                    "confidence": 1.0,
                    "source": "spatial_postprocess",
                    "region_indexes": region_indexes,
                }
            )

    if consumed_region_indexes:
        entities = remove_consumed_entities(
            entities,
            consumed_region_indexes,
        )

    entities.extend(new_entities)

    # --------------------------------------------------------
    # 2. Diagnosis
    # --------------------------------------------------------

    entities = merge_diagnosis_entities(
        entities,
        regions,
        debug,
    )

    # --------------------------------------------------------
    # 3. Leave dates
    #
    # Handles both:
    #
    #   "Vào viện lúc: Ngày 29..."
    #
    # and:
    #
    #   "Vào viện lúc:"      "Ngày 29..."
    # --------------------------------------------------------

    entities = resolve_leave_dates(
        entities,
        regions,
        debug,
    )

    # --------------------------------------------------------
    # 4. Recover missed entities
    # --------------------------------------------------------

    entities = recover_patient_name_from_ocr(
        entities,
        regions,
        debug,
    )

    entities = recover_citizen_id_from_ocr(
        entities,
        regions,
        debug,
    )

    # --------------------------------------------------------
    # 5. Remove invalid dates
    # --------------------------------------------------------

    entities = remove_invalid_date_entities(
        entities,
        debug,
    )

    # --------------------------------------------------------
    # 6. Hospital name cleanup
    # --------------------------------------------------------



    entities = clean_hospital_name_entities(entities)

    # --------------------------------------------------------
    # 6. Final normalization
    # --------------------------------------------------------

    entities = normalize_entities(
        entities,
        debug,
    )

    return entities