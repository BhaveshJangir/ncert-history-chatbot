import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from qdrant_client import QdrantClient
from llama_index.core.prompts import PromptTemplate

import sys
import os
# Add the project root to sys.path so 'from app import config' works when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config

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

load_dotenv()

app = FastAPI(title="NCERT History Chatbot API")

# Initialize models
logger.info("Initializing embeddings...")
embed_model = FastEmbedEmbedding(model_name=config.EMBED_MODEL_NAME)
Settings.embed_model = embed_model

# We use groq cloud for fast inference
# Make sure GROQ_API_KEY is set in your environment
api_key = os.getenv("GROQ_API_KEY")
if api_key:
    Settings.llm = Groq(model=config.LLM_MODEL, api_key=api_key)
else:
    print("WARNING: GROQ_API_KEY not found. Please set it in .env file.")

# Connect to local Qdrant
try:
    # Initialize the Qdrant Client (supports Docker or Local)
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        client = QdrantClient(url=qdrant_url)
    else:
        client = QdrantClient(path="./data/qdrant_data")

    vector_store = QdrantVectorStore(client=client, collection_name=config.COLLECTION_NAME, enable_hybrid=True)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
except Exception as e:
    index = None
    print(f"Warning: Could not load index. Did you run ingest.py? Error: {e}")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = []

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
    "2. If the user is just saying 'Hi', 'Hello', or greeting you, respond with a friendly greeting and ask how you can help them with their Class 10 NCERT History studies.\n"
    "3. If the context does not contain the answer, or if it is an out-of-syllabus/non-history question (like math, science, current affairs), you MUST refuse to answer and say you can only help with the NCERT History syllabus.\n"
    "4. Explain concepts clearly at a 15-year-old's reading level. Define hard terms if necessary.\n"
    "5. For every factual claim, you MUST cite the chapter and page number exactly as they appear in the metadata of the specific chunk you are using. Do not mix metadata between different chunks! Format it exactly as [Chapter: <chapter_name>, Page: <page_number>].\n"
    "6. Do NOT hallucinate or guess.\n"
    "7. Do NOT start your answer with phrases like 'According to the provided context' or 'Based on the text'. Just answer the question directly in a natural tone.\n\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

def check_guardrails(query: str, history, llm) -> str:
    """Pre-generation guardrail to filter out obvious off-topic questions."""
    # Provide the last few messages of context so the LLM understands follow-up questions like "Why?"
    history_str = ""
    if history:
        history_str = "Previous conversation:\n" + "\n".join([f"{msg.role}: {msg.content}" for msg in history[-3:]])
        
    guardrail_prompt = (
        f"{history_str}\n\n"
        f"Analyze the user's latest query: '{query}'.\n"
        "Classify the query into one of these categories:\n"
        "1. GREETING (friendly greetings like hi, hello, hey, heyy, wassup, good morning, etc.)\n"
        "2. RELEVANT (potentially related to Class 10 History, the NCERT textbook, or a natural continuation of the history discussion)\n"
        "3. IRRELEVANT (completely unrelated topics like math, science, programming, general chat)\n"
        "Answer with just the category name: 'GREETING', 'RELEVANT', or 'IRRELEVANT'."
    )
    try:
        response = llm.complete(guardrail_prompt).text.strip().upper()
        if "GREETING" in response:
            return "GREETING"
        elif "RELEVANT" in response or "YES" in response:
            return "RELEVANT"
        else:
            return "IRRELEVANT"
    except Exception:
        # If guardrail fails, let it pass to the main generation which also has rules
        return "RELEVANT"



@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not index:
        raise HTTPException(status_code=500, detail="Index not loaded. Please ingest data first.")
    
    if not hasattr(Settings, 'llm') or not Settings.llm:
         raise HTTPException(status_code=500, detail="LLM not initialized. Check GROQ_API_KEY.")
         
    # 1. Pre-generation guardrail
    # If it's a completely irrelevant query, reject early.
    guardrail_result = check_guardrails(request.query, request.history, Settings.llm)
    if guardrail_result == "IRRELEVANT":
        return ChatResponse(answer="I can only answer questions related to your Class 10 NCERT History syllabus. I cannot help with that.")
    elif guardrail_result == "GREETING":
        return ChatResponse(answer="Hello! How can I help you with your Class 10 NCERT History studies?")
         


    # 2. Retrieval & Logging
    # Setting vector_store_query_mode to "hybrid" uses both Dense and BM25
    retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K, vector_store_query_mode="hybrid")
    
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
        
    if not retrieved_nodes:
        logger.warning("No context retrieved. Database might be empty or query is unrelated.")
        return ChatResponse(answer="I couldn't find any relevant information in the textbook to answer your question. Make sure you've successfully ingested the PDF data!")
        
    logger.info("="*50)
    logger.info("🧠 GENERATING RESPONSE (Sending to Groq LLM...)")
    
    # Construct the Query Engine using the retriever
    query_engine = index.as_query_engine(
        similarity_top_k=config.SIMILARITY_TOP_K, 
        sparse_top_k=config.SPARSE_TOP_K, 
        vector_store_query_mode="hybrid",
        text_qa_template=qa_prompt_tmpl
    )
    
    # 3. Generation (Multi-turn using CondenseQuestionChatEngine)
    from llama_index.core.chat_engine import CondenseQuestionChatEngine
    from llama_index.core.llms import ChatMessage as LlamaChatMessage
    
    # Convert incoming history to LlamaIndex format
    chat_history = []
    if request.history:
        chat_history = [LlamaChatMessage(role=msg.role, content=msg.content) for msg in request.history]
        
    condense_prompt_str = (
        "Given the following conversation and a follow up question, "
        "rewrite the follow up question to be a standalone question.\n"
        "IMPORTANT: Do NOT hallucinate extra constraints like 'relate to the NCERT syllabus'. Just resolve pronouns (e.g. 'he', 'it').\n"
        "If the Follow Up Input is just a greeting, return it exactly as it is.\n"
        "Chat History:\n"
        "{chat_history}\n"
        "Follow Up Input: {question}\n"
        "Standalone question: "
    )
        
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        chat_history=chat_history,
        condense_question_prompt=PromptTemplate(condense_prompt_str),
        verbose=True
    )
    
    response = chat_engine.chat(request.query)
    
    logger.info("✅ FINAL ANSWER:")
    logger.info(str(response))
    logger.info("="*50)
    
    return ChatResponse(answer=str(response))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
