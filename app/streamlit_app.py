# app/streamlit_app.py

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.device import get_device
from src.pipeline import DocumentPipeline
from app.utils import (
    TEMPLATE_TYPES,
    build_entity_df,
    build_processing_items,
    create_excel,
    create_json,
    display_name,
    get_detected_template,
    get_missing_required,
    get_quality,
    save_temp_image,
    validate_review,
)


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Social Insurance Documents OCR + KIE",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    return DocumentPipeline(
        device=get_device(),
        score_thresh=0.3,
        recognition_batch_size=32,
        enable_orientation=True,
    )


pipeline = load_pipeline()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "results": [],
    "reviewed_documents": {},
    "edited_entities": {},
    "notifications": [],
    "final_template_types": {},
    "review_notes": {},
}


def reset_state():
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value.copy()


for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy()


# ============================================================
# HEADER
# ============================================================

st.title("📄 Social Insurance Documents OCR + KIE")
st.caption("Upload → OCR + KIE → Review → Commit → Export")

uploaded_files = st.file_uploader(
    "Upload Images or PDFs",
    type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)


# ============================================================
# METRICS
# ============================================================

need_review = sum(
    bool(get_missing_required(item["result"]))
    for item in st.session_state.results
    if item["document_id"] not in st.session_state.reviewed_documents
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Uploaded", len(uploaded_files or []))
c2.metric("Processed", len(st.session_state.results))
c3.metric("Need Review", need_review)
c4.metric("Completed", len(st.session_state.reviewed_documents))


# ============================================================
# UPLOAD PREVIEW
# ============================================================

if uploaded_files:
    st.subheader("Uploaded Documents")

    for file in uploaded_files:
        with st.expander(file.name):
            if Path(file.name).suffix.lower() == ".pdf":
                st.info("PDF will be converted into page images when processed.")

            else:
                try:
                    image = Image.open(io.BytesIO(file.getvalue()))
                    st.image(image, width ='stretch')

                except Exception:
                    st.warning("Preview unavailable.")


# ============================================================
# PROCESS
# ============================================================

st.divider()

c1, c2 = st.columns([1, 5])

process_clicked = c1.button(
    "🚀 Process",
    type="primary",
    disabled=not uploaded_files,
    width ='stretch',
)

clear_clicked = c2.button(
    "Clear Results",
    disabled=not st.session_state.results,
)


if clear_clicked:
    reset_state()
    st.rerun()


if process_clicked:
    reset_state()

    try:
        items = build_processing_items(uploaded_files)

    except Exception as e:
        items = []
        st.error(f"Cannot prepare files: {e}")


    if not items:
        st.error("No valid document pages found.")

    else:
        progress = st.progress(0)
        status = st.empty()

        for i, item in enumerate(items, 1):

            status.info(
                f"Processing {i}/{len(items)}: {display_name(item)}"
            )

            temp = save_temp_image(
                item["image_bytes"],
                item["extension"],
            )

            try:
                result = pipeline.process(str(temp))

                st.session_state.results.append({
                    **item,
                    "result": result,
                })

                missing = get_missing_required(result)

                if missing:
                    st.session_state.notifications.append(
                        (
                            "warning",
                            display_name(item),
                            f"Missing: {', '.join(missing)}",
                        )
                    )

                quality, message = get_quality(result)

                if quality in {"warning", "review", "poor"}:
                    st.session_state.notifications.append(
                        (
                            "warning",
                            display_name(item),
                            message or "Please review document quality.",
                        )
                    )

            except Exception as e:
                result = {
                    "entities": [],
                    "quality": {
                        "status": "error",
                        "notification": str(e),
                    },
                }

                st.session_state.results.append({
                    **item,
                    "result": result,
                })

                st.session_state.notifications.append(
                    (
                        "error",
                        display_name(item),
                        str(e),
                    )
                )

            finally:
                temp.unlink(missing_ok=True)

            progress.progress(i / len(items))

        status.success(
            f"Processed {len(items)} page(s)."
        )


# ============================================================
# NOTIFICATIONS
# ============================================================

if st.session_state.results:
    st.divider()
    st.subheader("Notifications")

    if not st.session_state.notifications:
        st.success("No issues detected.")

    for level, filename, message in st.session_state.notifications:

        text = f"**{filename}** — {message}"

        if level == "error":
            st.error(text)

        else:
            st.warning(text)


# ============================================================
# REVIEW
# ============================================================

if st.session_state.results:
    st.divider()
    st.subheader("Review Documents")


    # ========================================================
    # TEMPLATE FILTER
    # ========================================================

    detected_templates = sorted({
        get_detected_template(item["result"])
        for item in st.session_state.results
        if get_detected_template(item["result"])
    })

    filter_col, document_col = st.columns([1, 3])


    with filter_col:
        template_filter = st.selectbox(
            "Template Type",
            ["All"] + detected_templates,
        )


    filtered_items = [
        item
        for item in st.session_state.results
        if template_filter == "All"
        or get_detected_template(item["result"]) == template_filter
    ]


    options = {
        display_name(item): item["document_id"]
        for item in filtered_items
    }


    with document_col:
        selected_name = st.selectbox(
            "Document",
            list(options.keys()),
        )


    doc_id = options[selected_name]

    item = next(
        item
        for item in filtered_items
        if item["document_id"] == doc_id
    )

    result = item["result"]


    # ========================================================
    # MISSING / QUALITY WARNING
    # ========================================================

    missing = get_missing_required(result)

    if missing:
        st.warning(
            f"Missing required fields: {', '.join(missing)}"
        )

    else:
        st.success(
            "All required fields detected."
        )


    quality, quality_message = get_quality(result)

    if quality == "error":
        st.error(quality_message)

    elif quality in {"warning", "review", "poor"}:
        st.warning(
            quality_message or "Please review this document."
        )


    # ========================================================
    # REVIEW LAYOUT
    # ========================================================

    left, right = st.columns([1, 1.5])


    # ========================================================
    # DOCUMENT
    # ========================================================

    with left:
        st.markdown("### Document")

        try:
            image = Image.open(
                io.BytesIO(item["image_bytes"])
            )

            st.image(
                image,
                caption=selected_name,
                width ='stretch',
            )

        except Exception:
            st.warning("Preview unavailable.")


        st.caption(
            f"Source: {item['source_filename']}"
        )

        if item.get("page_number"):
            st.caption(
                f"Page: {item['page_number']}"
            )


    # ========================================================
    # EXTRACTED INFORMATION
    # ========================================================

    with right:
        st.markdown("### Extracted Information")


        # ====================================================
        # FINAL TEMPLATE TYPE
        # ====================================================

        detected_template = get_detected_template(result)

        saved_template = (
            st.session_state.final_template_types.get(
                doc_id,
                detected_template,
            )
        )

        templates = TEMPLATE_TYPES.copy()

        for template in [detected_template, saved_template]:

            if template and template not in templates:
                templates.insert(1, template)


        final_template = st.selectbox(
            "Final Template Type",
            templates,
            index=(
                templates.index(saved_template)
                if saved_template in templates
                else 0
            ),
            key=f"template_{doc_id}",
        )


        if detected_template:
            st.caption(
                f"Model detected: {detected_template}"
            )

        else:
            st.warning(
                "Template type was not detected."
            )


        # ====================================================
        # ENTITY EDITOR
        # ====================================================

        df = build_entity_df(
            result,
            st.session_state.edited_entities.get(
                doc_id,
                {},
            ),
        )


        edited_df = st.data_editor(
            df,
            hide_index=True,
            width ='stretch',
            disabled=[
                "Field",
                "Confidence",
                "Detected",
                "Required",
            ],
            column_config={
                "Value": st.column_config.TextColumn(
                    "Reviewed Value"
                ),

                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                ),
            },
            key=f"editor_{doc_id}",
        )


        # ====================================================
        # NOTE
        # ====================================================

        note = st.text_area(
            "Note",
            value=st.session_state.review_notes.get(
                doc_id,
                "",
            ),
            placeholder="Add reviewer note...",
            key=f"note_{doc_id}",
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        missing_review = validate_review(
            edited_df
        )


        if missing_review:
            st.info(
                f"Still missing: {', '.join(missing_review)}"
            )

        else:
            st.success(
                "All required fields complete."
            )


        # ====================================================
        # COMMIT
        # ====================================================

        c1, c2 = st.columns([1.5, 2])


        commit_clicked = c1.button(
            "✅ Commit",
            type="primary",
            key=f"commit_{doc_id}",
            width ='stretch',
        )


        if doc_id in st.session_state.reviewed_documents:
            c2.success("Completed")

        else:
            c2.info("Not committed")


        if commit_clicked:

            if not final_template:
                st.error(
                    "Select Final Template Type."
                )


            elif missing_review:
                st.error(
                    f"Complete required fields: "
                    f"{', '.join(missing_review)}"
                )


            else:
                values = {
                    row["Field"]: str(
                        row["Value"]
                        if pd.notna(row["Value"])
                        else ""
                    ).strip()
                    for _, row in edited_df.iterrows()
                }

                values["TEMPLATE_TYPE"] = final_template


                st.session_state.edited_entities[
                    doc_id
                ] = values

                st.session_state.final_template_types[
                    doc_id
                ] = final_template

                st.session_state.review_notes[
                    doc_id
                ] = note.strip()

                st.session_state.reviewed_documents[
                    doc_id
                ] = True

                st.rerun()


# ============================================================
# REVIEW STATUS
# ============================================================

if st.session_state.results:
    st.divider()
    st.subheader("Review Status")

    rows = []

    for item in st.session_state.results:

        doc_id = item["document_id"]

        missing = get_missing_required(
            item["result"]
        )

        completed = (
            doc_id
            in st.session_state.reviewed_documents
        )

        rows.append({
            "Document": display_name(item),

            "Detected Template":
                get_detected_template(
                    item["result"]
                ),

            "Final Template":
                st.session_state.final_template_types.get(
                    doc_id,
                    "",
                ),

            "Status":
                (
                    "Completed"
                    if completed
                    else (
                        "Needs Review"
                        if missing
                        else "Ready"
                    )
                ),

            "Missing Fields":
                ", ".join(missing),

            "Note":
                st.session_state.review_notes.get(
                    doc_id,
                    "",
                ),
        })


    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width ='stretch',
    )


# ============================================================
# COMPLETED DOCUMENTS
# ============================================================

if st.session_state.reviewed_documents:
    st.subheader("Completed Documents")

    rows = []

    for item in st.session_state.results:

        doc_id = item["document_id"]

        if doc_id not in st.session_state.reviewed_documents:
            continue


        row = {
            "Document": display_name(item),

            "Final Template":
                st.session_state.final_template_types.get(
                    doc_id,
                    "",
                ),

            "Note":
                st.session_state.review_notes.get(
                    doc_id,
                    "",
                ),
        }


        row.update(
            st.session_state.edited_entities.get(
                doc_id,
                {},
            )
        )

        rows.append(row)


    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width ='stretch',
    )


# ============================================================
# EXPORT
# ============================================================

if st.session_state.results:
    st.divider()
    st.subheader("Export")


    export_args = (
        st.session_state.results,
        st.session_state.reviewed_documents,
        st.session_state.edited_entities,
        st.session_state.final_template_types,
        st.session_state.review_notes,
    )


    c1, c2 = st.columns(2)


    c1.download_button(
        "📊 Export Excel",
        create_excel(*export_args),
        "document_results.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width ='stretch',
    )


    c2.download_button(
        "📄 Export JSON",
        create_json(*export_args),
        "document_results.json",
        "application/json",
        width ='stretch',
    )