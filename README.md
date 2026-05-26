# PDF OCR & Image Extraction Pipeline

A modular Python-based pipeline for extracting structured text, images, OCR captions, and metadata from PDF documents using AI-powered OCR and computer vision techniques.

## Overview

This project automates the processing of PDF textbooks or scanned documents by:

- Extracting text using deep learning OCR
- Detecting and extracting images from pages
- Generating structured metadata
- Organizing outputs by chapter and page
- Building a searchable master index

Designed for large academic or technical books such as medical, radiology, and educational references.

---

# Features

- AI-powered OCR using DocTR
- Automatic chapter detection from filenames
- Structured text extraction per page
- Image extraction using OpenCV contour detection
- OCR-based image caption generation
- Metadata indexing system
- JSON + TXT export formats
- Batch PDF processing pipeline
- Safe PyTorch model loading implementation
- Memory cleanup for large-scale processing

---

# Technologies Used

- Python
- PyTorch
- DocTR OCR
- OpenCV
- PyMuPDF (fitz)
- NumPy
- PIL
- tqdm

---

# Processing Pipeline

1. Validate and load PDF files
2. Detect chapter information from filenames
3. Extract OCR text page-by-page
4. Clean and structure extracted text
5. Detect images using contour analysis
6. Generate image metadata and OCR captions
7. Build master metadata index
8. Export organized outputs

---

# Extraction Strategy

## OCR Strategy
The system uses DocTR's pretrained OCR predictor to extract high-quality text from scanned or digital PDFs.

Techniques used:
- Structured block and line parsing
- Bullet and numbered list preservation
- OCR text cleanup and normalization
- Caption harvesting for image metadata

---

## Image Detection Strategy
Images are detected using computer vision contour analysis.

Techniques used:
- Edge detection with Canny filtering
- Contour extraction
- Bounding box filtering
- Aspect ratio validation
- High-resolution page rendering for better accuracy

---

## Metadata Strategy
The pipeline automatically generates:
- Chapter metadata
- Section indexing
- Page references
- Image metadata
- Master JSON index for all chapters

---

# Output Structure

```bash
Extracted_Data/
│
├── Text_Extraction/
│   └── Chapter_x/
│       └── Page_x/
│           ├── .txt
│           └── .json
│
├── Images/
│   └── Chapter_x/
│       └── Page_x/
│           ├── image files
│           └── metadata.json
│
└── Master_Index.json
```

---

# Purpose

This project was built to create a scalable and automated document digitization pipeline capable of transforming large educational PDF collections into structured datasets suitable for:

- Knowledge extraction
- AI training datasets
- Search systems
- RAG pipelines
- Medical documentation indexing
- Digital archiving

---

# Key Design Principles

- Modular architecture
- Scalable processing
- Fault tolerance
- Structured data generation
- Automation-first workflow
- Maintainable object-oriented design

---

# Example Use Cases

- Medical textbook digitization
- Academic archive processing
- OCR dataset generation
- AI knowledge base preparation
- Research indexing systems
- Educational content extraction

---

# Installation

```bash
pip install doctr pymupdf opencv-python pillow numpy tqdm torch
```

---

# Run

```bash
python main.py
```

---

# Author

Built for advanced PDF intelligence extraction, OCR automation, and structured document processing workflows.
