import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging to save to a file and output to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from qdrant_client import QdrantClient
from llama_index.core.prompts import PromptTemplate

load_dotenv()

app = FastAPI(title="NCERT History Chatbot API")

# Initialize models
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# We use groq cloud for fast inference
# Make sure GROQ_API_KEY is set in your environment
api_key = os.getenv("GROQ_API_KEY")
if api_key:
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=api_key)
else:
    print("WARNING: GROQ_API_KEY not found. Please set it in .env file.")

# Connect to local Qdrant
try:
    client = QdrantClient(path="./qdrant_data")
    vector_store = QdrantVectorStore(client=client, collection_name="ncert_history", enable_hybrid=True)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
except Exception as e:
    index = None
    print(f"Warning: Could not load index. Did you run ingest.py? Error: {e}")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

qa_prompt_tmpl_str = (
    "You are an expert history tutor for a 15-year-old Class 10 student. \n"
    "Your goal is to answer questions strictly based on the provided NCERT textbook context.\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Rules:\n"
    "1. You MUST answer the question using ONLY the provided context.\n"
    "2. If the context does not contain the answer, or if it is an out-of-syllabus/non-history question (like math, science, current affairs), you MUST refuse to answer and say you can only help with the NCERT History syllabus.\n"
    "3. Explain concepts clearly at a 15-year-old's reading level. Define hard terms if necessary.\n"
    "4. For every factual claim, you MUST cite the chapter and page number at the end of the sentence or paragraph, using the metadata provided in the context (e.g., [Chapter 1, Page 12]).\n"
    "5. Do NOT hallucinate or guess. \n\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

def check_guardrails(query: str, llm) -> bool:
    """Pre-generation guardrail to filter out obvious off-topic questions."""
    guardrail_prompt = (
        f"Analyze this query: '{query}'.\n"
        "Is this a question that could potentially be related to Class 10 History or the NCERT textbook? "
        "Answer with just 'YES' or 'NO'."
    )
    try:
        response = llm.complete(guardrail_prompt).text.strip().upper()
        return "YES" in response
    except Exception:
        # If guardrail fails, let it pass to the main generation which also has rules
        return True

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not index:
        raise HTTPException(status_code=500, detail="Index not loaded. Please ingest data first.")
    
    if not hasattr(Settings, 'llm') or not Settings.llm:
         raise HTTPException(status_code=500, detail="LLM not initialized. Check GROQ_API_KEY.")
         
    # 1. Pre-generation guardrail
    # If it's a completely irrelevant query, reject early.
    is_valid = check_guardrails(request.query, Settings.llm)
    if not is_valid:
        return ChatResponse(answer="I can only answer questions related to your Class 10 NCERT History syllabus. I cannot help with that.")

    # 2. Retrieval & Logging
    # Setting vector_store_query_mode to "hybrid" uses both Dense and BM25
    retriever = index.as_retriever(similarity_top_k=5, vector_store_query_mode="hybrid")
    
    logger.info("="*50)
    logger.info(f"🧐 USER QUERY: {request.query}")
    logger.info("="*50)
    
    # Manually retrieve to show logs
    retrieved_nodes = retriever.retrieve(request.query)
    
    logger.info("🔎 RETRIEVED DATA (Hybrid Search + Qdrant Fusion):")
    for i, node in enumerate(retrieved_nodes):
        chapter = node.metadata.get("chapter", "Unknown")
        page = node.metadata.get("page_number", "Unknown")
        score = node.score if node.score else 0.0
        logger.info(f"--- Result {i+1} | Score: {score:.4f} | Chapter: {chapter} | Page: {page} ---")
        content_preview = node.get_content()[:300].replace('\n', ' ')
        logger.info(content_preview + "..." if len(node.get_content()) > 300 else content_preview)
        
    logger.info("="*50)
    logger.info("🧠 GENERATING RESPONSE (Sending to Groq LLM...)")
    
    # Construct the Query Engine using the retriever
    query_engine = index.as_query_engine(
        similarity_top_k=5, 
        vector_store_query_mode="hybrid",
        text_qa_template=qa_prompt_tmpl
    )
    
    # 3. Generation
    response = query_engine.query(request.query)
    
    logger.info("✅ FINAL ANSWER:")
    logger.info(str(response))
    logger.info("="*50)
    
    return ChatResponse(answer=str(response))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
