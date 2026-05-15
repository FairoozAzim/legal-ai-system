# Legal AI System (Grounded Document Understanding)

## 1. Overview
This project builds a grounded legal document AI system that:
- Extracts text from PDFs (including OCR for scanned documents)
- Performs retrieval over document chunks
- Generates grounded summaries using LLMs
- Learns from user feedback (using a small LLM) to improve future outputs

---

## 2. Features
- PDF text + OCR extraction
- Chunk-based retrieval (FAISS)
- Embedding-based semantic search
- Grounded LLM generation
- Feedback loop with learned writing rules

---

## 3. Architecture

PDF → OCR/Text Extraction → Chunking → Embeddings → Vector DB (FAISS)
→ Retrieval → LLM Generation → Feedback Store → Rule Learning -> Improved Draft Generation

---

## 4. Setup Instructions

### Install dependencies
```bash
pip install -r requirements.txt
```
### Install Tesseract (for OCR)
https://github.com/UB-Mannheim/tesseract/wiki

### Set GROQ API KEY(Free Usage Available)
 ```bash
 export GROQ_API_KEY=your_key
 ```
 ### Test Pipeline
 ```bash
 python test.py
```

### Assumptions
- Documents are legal/structured text or scanned PDFs
- OCR quality may vary for noisy scans
- Retrieval is based on semantic similarity only

### Tradeoffs
- FAISS used instead of vector DB (simplicity > scalability)
- Lightweight open-source models used for cost efficiency
- Heuristic feedback learning instead of full fine-tuning due to time-constraints