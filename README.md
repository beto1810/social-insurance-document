# Social Insurance Document OCR + KIE

An end-to-end document understanding system for processing social insurance documents such as medical leave certificates, hospital discharge forms, and related administrative paperwork. The project combines document OCR, layout analysis, key information extraction (KIE), and a review workflow to convert scanned documents into structured records.

## Overview

This project is designed to:

- ingest PDF and image files
- detect document orientation and page layout
- extract text using OCR
- identify key fields from social insurance forms
- validate required values and quality checks
- let users review and correct extracted data in a Streamlit UI
- export the final results to Excel/JSON for downstream processing

The application is built around a custom document pipeline and a review interface for human verification before final export.

## Supported use case

The system currently targets document fields such as:

- `TEMPLATE_TYPE`
- `HOSPITAL_NAME`
- `PATIENT_NAME`
- `DATE_OF_BIRTH`
- `GENDER`
- `SOCIAL_INSURANCE_NUMBER`
- `HEALTH_INSURANCE_CARD`
- `MEDICAL_CERTIFICATE_NUMBER`
- `CITIZEN_ID`
- `DIAGNOSIS`
- `LEAVE_DURATION`
- `LEAVE_START_DATE`
- `LEAVE_END_DATE`
- `PARENT_NAME`

The project includes required/optional field validation and can flag documents that need review when critical information is missing.

## Main features

- PDF page rendering to images
- image and PDF upload through a web interface
- OCR text extraction and region detection
- document template detection
- structured entity extraction for Vietnamese documents
- validation of required fields and quality checks
- review workflow for correcting extracted values
- export of final output to Excel and JSON
- Docker support for local deployment

## Architecture

```text
Upload document
      │
      ▼
PDF/image preprocessing
      │
      ▼
Orientation and layout analysis
      │
      ▼
OCR extraction
      │
      ▼
Key Information Extraction (KIE)
      │
      ▼
Validation and quality review
      │
      ├──► Streamlit review UI
      │
      └──► Export results (Excel / JSON)
```

## Repository structure

```text
Social-Insurance-document/
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py
│   └── utils.py
├── src/
│   ├── __init__.py
│   ├── device.py
│   ├── pipeline.py
│   ├── kie/
│   └── ocr/
├── third_party/
│   └── vietocr/
├── Dockerfile
├── pyproject.toml
├── README.md
├── vgg-transformer.yml
└── .
```

## Tech stack

- Python 3.11
- PyTorch + torchvision
- OCR and document processing libraries
- PaddleOCR / LayoutXLM based workflow
- Streamlit for the review UI
- pandas / openpyxl for structured export
- Docker for packaging

## Local setup

### 1) Clone the repository

```bash
git clone <repository-url>
cd Social-Insurance-document
```

### 2) Create a virtual environment

Using `uv` is recommended because the project includes local dependency configuration in `pyproject.toml`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install uv
uv pip install -e .
```

### 3) Run the app

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

## Docker

Build and run the container:

```bash
docker build -t social-insurance-doc-ai .
docker run -p 8501:8501 social-insurance-doc-ai
```

The app runs on port `8501` and exposes the Streamlit interface.

## Usage

1. Open the Streamlit application.
2. Upload one or more PDF/image files.
3. Click the processing button to run OCR and KIE.
4. Review the extracted entities and fix any missing or incorrect values.
5. Export completed records to Excel or use the extracted JSON output.

## Notes on data quality

The pipeline includes validation and review checks to help highlight:

- missing required fields
- low-quality OCR output
- document template mismatch
- review-needed records before export

This makes the system useful as a semi-automated document processing workflow rather than a fully silent extraction pipeline.

## Development notes

The project is organized around a modular processing pipeline:

- `src/pipeline.py` orchestrates the end-to-end pipeline
- `src/ocr/` handles OCR and preprocessing
- `src/kie/` handles field extraction and validation
- `app/streamlit_app.py` provides the front-end UI
- `app/utils.py` contains metadata, validation, export, and review logic

## License

This project is provided as-is for research and internal business use. Please check the project repository and any third-party dependencies for their respective licenses, especially the code under `third_party/vietocr`.

## Future improvements

Possible next steps include:

- stronger template-specific validation
- batch processing for large document sets
- API service layer for programmatic integration
- database storage and audit history
- better model tuning for Vietnamese document variations

