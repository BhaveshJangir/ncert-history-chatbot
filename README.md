# NCERT Class 10 History Study Chatbot

A highly optimized Retrieval-Augmented Generation (RAG) chatbot designed to answer Class 10 student questions based strictly on the NCERT textbook, complete with precise chapter and page citations.

## Features
- **Strict Grounding:** Refuses out-of-syllabus and non-history questions.
- **Precise Citations:** Outputs exactly which chapter and page the information comes from.
- **Multi-turn Chat:** Supports coreference resolution (e.g. "Who was Gandhi? -> What did he do?")
- **OCR Fallback:** Uses Tesseract to extract text from scanned images.
- **Hybrid Search:** Uses Qdrant for dense (semantic) and sparse (keyword/BM25) search.

## Prerequisites
- A [Groq API Key](https://console.groq.com/keys) (Free)
- Python 3.10+

---

## 📁 Project Structure
```text
ncert-history-chatbot/
├── app/                        # Main application source code
│   ├── main.py                 # FastAPI server & Chat Engine
│   ├── ingest.py               # Data Ingestion Script
│   ├── eval.py                 # Ragas evaluation script
│   ├── config.py               # Application configuration
│   └── ui.py                   # Streamlit frontend UI
├── data/                       # Local data storage
│   ├── ncert_history.pdf       # The textbook
│   └── qdrant_data/            # Local SQLite database (if run locally)
├── requirements.txt            # Python dependencies
├── golden_dataset.json         # ~50-100 evaluation Q&A pairs template
└── sample_transcripts.md       # Examples of bot interactions
```

---

## 🚀 Quick Start
Run the chatbot directly on your local machine using Python. The Qdrant vector database will automatically use local disk storage (`./data/qdrant_data`).

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: For the OCR fallback to work locally, you must install Tesseract-OCR on your host machine).*

2. **Add your API Key:** Create a `.env` file in the root directory with `GROQ_API_KEY`.

3. **Run the Ingestion Pipeline:**
   ```bash
   python app/ingest.py data/ncert_history.pdf
   ```

4. **Start the API Server:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

5. **Start the Streamlit UI (in a new terminal):**
   ```bash
   streamlit run app/ui.py
   ```
   The UI will be available at `http://localhost:8501`.

## Usage
You can test the API using `curl` or Postman:
```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "Who wrote Hind Swaraj?", "history": []}'
```

## Architecture Notes
- **Ingestion:** Uses PyMuPDF to extract text. Uses heuristics to detect chapters, and falls back to PyTesseract for scanned image pages.
- **Vector DB:** Qdrant supports Hybrid Search out of the box. Automatically connects to local disk storage (`./data/qdrant_data`).
- **LLM:** Groq provides blazing-fast inference for query execution.
