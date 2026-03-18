# AI-Powered Structural Compliance Auditor 🏗️🤖

> An offline Retrieval-Augmented Generation (RAG) system with custom Computer Vision integration for automated IS Code structural auditing.

**Inventors/Lead Engineers:** Arshan Ali Khan

---

## 📖 About the Project
Structural auditing in civil engineering relies heavily on manual defect measurement and tedious cross-referencing with massive regulatory documents like the 114-page IS 456:2000 codebook. This process is slow, linear, and prone to human error.

This project introduces a **fully offline, closed-loop cyber-physical system**. By bridging a custom-trained Computer Vision model with a deterministic Natural Language Processing (NLP) database, the system acts as an automated site inspector. It mathematically measures structural defects (such as cracks) and instantly queries a local vector database to verify if the defect falls within the legal permissible limits, outputting an automated `PASS/FAIL` alert.

## 🧠 System Architecture

The architecture is divided into two primary engines communicating via a local API bridge:

1. **The Vision Auditor (Hardware/Math Layer):** A custom YOLOv8 Instance Segmentation model isolates the exact polygonal pixel boundary of a structural defect. OpenCV extracts the extreme coordinates and calculates the Euclidean distance, dynamically translating pixel math into physical millimeters.
2. **The NLP Engine (Regulatory Layer):** An offline RAG pipeline. Engineering rules are extracted from raw PDFs, embedded using a Sentence-BERT transformer, and indexed in FAISS for millisecond-latency semantic retrieval (Cosine Similarity). 

## 🛠️ Tech Stack

* **Computer Vision:** YOLOv8 (Instance Segmentation), OpenCV, PyTorch, SciPy
* **NLP & Vector Search:** `all-MiniLM-L6-v2` (Sentence-Transformers), FAISS (Facebook AI Similarity Search)
* **Data Extraction:** PyMuPDF (`fitz`), Regular Expressions (Regex)
* **Backend & Frontend:** Python, Flask, HTML/CSS/JavaScript

## ⚙️ Repository Structure

* `extract_data.py`: Data mining script that uses Regex and PyMuPDF to parse the double-column IS 456 PDF and build the `is456_data.json` database.
* `app.py`: The Flask server and offline FAISS RAG backend. Hosts the NLP search engine and serves the interactive Web UI.
* `vision_auditor.py`: The computer vision bridge. Loads the YOLOv8 model, calculates defect dimensions via matrix math, and POSTs queries to the Flask server for compliance verification.
* `is456_data.json`: The cleaned, vectorized regulatory database.
* `best.pt`: Custom-trained YOLOv8 weights for concrete defect segmentation.

## 🚀 Installation & Local Setup

**Prerequisites:** Python 3.8+ and a C++ build environment (for FAISS).

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/structural-vision-rag.git](https://github.com/yourusername/structural-vision-rag.git)
   cd structural-vision-rag

2. **Install dependencies:**

```bash
pip install flask faiss-cpu sentence-transformers torch ultralytics opencv-python scipy pymupdf

