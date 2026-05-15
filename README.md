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
- A simple UI with streamlit to test the system 

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
 ### Test Pipeline with demo PDF added
 ```bash
 python test.py

```

### Test Pipeline with Own PDF
 ```bash
    streamlit run streamlit_app.py

```

### Assumptions
- Documents are legal/structured text or scanned PDFs
- OCR quality may vary for noisy scans
- Retrieval is based on semantic similarity only

### Tradeoffs
- FAISS used instead of vector DB (simplicity > scalability)
- Lightweight open-source models used for cost efficiency
- Heuristic feedback learning instead of full fine-tuning due to time-constraints

### Qualitative Evaluation
- System struggles a little with very blurry images, but still manages to capture some information
- The references are grounding properly, though a thorough evaluation couldn't be conducted due to lack of legal documents and time   constrains
- The feedback learning is now heuristic approach rather than more sophisticated techniques like Reinforcement Learning or fine-tuning.

### Future Work Needed
- System stores the embeddings in the vector database but generates the embedding every time instead of looking up in the db. 
- The operator cannot edit in the streamlit app yet, needs to be done inside the ide by running the main.py
- Even though the feedback learning partially works, it still needs some major evaluation and improvements. 

