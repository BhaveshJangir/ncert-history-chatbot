# NCERT Class 10 History Study Chatbot

A highly optimized Retrieval-Augmented Generation (RAG) chatbot designed to answer Class 10 student questions based strictly on the NCERT textbook, complete with precise chapter and page citations.

## Features
- **Strict Grounding:** Refuses out-of-syllabus and non-history questions.
- **Precise Citations:** Outputs exactly which chapter and page the information comes from.
- **Hybrid Search:** Uses Qdrant for dense (semantic) and sparse (keyword/BM25) search.
- **Age-Appropriate:** Explains concepts at a 15-year-old's level.
- **Production-Ready:** Built on FastAPI with LlamaIndex and Groq Cloud for ultra-fast, zero-cost LLM inference.

## Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/keys) (Free)

## Setup Instructions

1. **Clone & Environment Setup:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Data Ingestion:**
   Place the official NCERT History textbook PDF (e.g., `ncert_history.pdf`) in the project directory, then run the ingestion script:
   ```bash
   python ingest.py ncert_history.pdf
   ```
   *This will parse the PDF, create chunks, generate embeddings (locally using FastEmbed), and index them into a local Qdrant database.*

4. **Run the API:**
   Start the FastAPI server:
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`. You can access the Swagger UI documentation at `http://localhost:8000/docs`.

## Usage
You can test the API using `curl` or Postman:
```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "Who wrote Hind Swaraj?"}'
```

## Architecture Notes
- **Ingestion:** Uses PyMuPDF to extract text with page metadata. Employs heuristics to separate glossaries and source boxes.
- **Vector DB:** Qdrant is used natively via `qdrant-client` to avoid Docker requirements, supporting Hybrid Search out of the box.
- **LLM:** Groq provides blazing-fast inference using Llama-3-70B for the main generation and Llama-3-8B for guardrails.
