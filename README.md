# AI-Invoice-Understanding

## Overview

AI Invoice Understanding Platform is an end-to-end document AI system designed to automatically process invoice documents.

The platform accepts invoice files such as PDF, PNG, and JPEG, extracts text and document layout information, identifies important invoice fields, validates extracted values, and returns structured data through an API.

The goal of this project is to build a production-ready invoice processing pipeline that can reduce manual data entry and support integration with accounting, finance, and enterprise systems.

## Main Features

* Upload invoice documents in PDF or image format
* Convert PDF pages into processable images
* Detect document layout and invoice regions
* Extract text using Optical Character Recognition
* Extract key invoice fields using Key Information Extraction
* Normalize dates, currencies, addresses, and numerical values
* Validate extracted invoice information
* Store document metadata and extraction results
* Provide REST API endpoints for document processing
* Display processing results through a web interface
* Package and deploy the application using Docker
* Test individual modules and the complete processing pipeline


## Project Architecture

```text
Invoice Document
        │
        ▼
Document Upload API
        │
        ▼
PDF and Image Preprocessing
        │
        ▼
Layout Detection
        │
        ▼
OCR Text Extraction
        │
        ▼
Key Information Extraction
        │
        ▼
Data Normalization and Validation
        │
        ▼
Database Storage
        │
        ├──► REST API Response
        │
        └──► Frontend Review Interface
```

## Repository Structure

```text
AI-document-understanding/
├── data/
├── notebooks/
├── src/
│   ├── layout/
│   ├── ocr/
│   ├── kie/
│   ├── api/
│   ├── services/
│   ├── database/
│   └── utils/
├── frontend/
├── docker/
├── tests/
├── README.md
└── requirements.txt
```

