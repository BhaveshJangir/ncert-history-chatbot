import os
import fitz  # PyMuPDF
from typing import List
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings

load_dotenv()

def parse_pdf(pdf_path: str) -> List[Document]:
    """
    Parses the PDF and extracts text with metadata, 
    separating body text from glossaries/source boxes using basic heuristics.
    """
    doc = fitz.open(pdf_path)
    documents = []
    current_chapter = "Unknown Chapter"
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict").get("blocks", [])
        
        for b in blocks:
            if "lines" in b:
                text = ""
                is_bold = False
                header_font = False
                for l in b["lines"]:
                    for s in l["spans"]:
                        text += s["text"] + " "
                        font_name = s.get("font", "").lower()
                        font_size = s.get("size", 0)
                        if "bold" in font_name or "black" in font_name:
                            is_bold = True
                        if "rotissemiserif" in font_name and 9.5 <= font_size <= 10.5:
                            header_font = True
                
                text = text.strip()
                if not text or len(text) < 10: # Skip very short useless text
                    continue
                
                # Heuristics for tagging
                element_type = "body"
                lower_text = text.lower()
                
                # NCERT specific Chapter header detection
                if header_font and "india and the contemporary world" not in lower_text:
                    if len(text) > 4:
                        current_chapter = text
                    continue
                elif lower_text.startswith("chapter"):
                    current_chapter = text.split("\n")[0]
                    element_type = "chapter_heading"
                elif lower_text.startswith("source") or lower_text.startswith("box"):
                    element_type = "source_box"
                elif "new words" in lower_text or lower_text.startswith("glossary"):
                    element_type = "glossary"
                elif len(text.split()) < 15 and is_bold:
                    element_type = "heading"
                elif "?" in text and len(text.split()) < 30 and ("exercises" in lower_text or "discuss" in lower_text):
                    element_type = "question"
                
                # Create a LlamaIndex document for each meaningful block
                doc_obj = Document(
                    text=text,
                    metadata={
                        "page_number": page_num + 1,
                        "chapter": current_chapter,
                        "element_type": element_type,
                    },
                    excluded_embed_metadata_keys=["element_type"],
                    excluded_llm_metadata_keys=["element_type"]
                )
                documents.append(doc_obj)
                
    return documents

def ingest_data(pdf_path: str, collection_name: str = "ncert_history"):
    logger.info(f"Parsing PDF: {pdf_path}...")
    documents = parse_pdf(pdf_path)
    logger.info(f"Extracted {len(documents)} blocks from PDF.")

    # 1. Deliberate chunking strategy
    # We use a SentenceSplitter to ensure chunks respect sentence boundaries and aren't too large
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    logger.info(f"Created {len(nodes)} nodes/chunks.")

    # 2. Embeddings setup
    # Using FastEmbed for dense vectors (runs locally)
    logger.info("Initializing embeddings...")
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    
    # We will use Qdrant for both dense and sparse (hybrid) search
    # Running Qdrant locally in memory/disk mode via client
    logger.info("Connecting to local Qdrant...")
    client = QdrantClient(path="./qdrant_data")
    
    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' already exists. Overwriting...")
        client.delete_collection(collection_name)
    
    # Setup the Qdrant vector store
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=collection_name,
        enable_hybrid=True # Crucial for proper noun search (BM25 + Dense)
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Create Index and Index the nodes
    logger.info("Indexing nodes into Qdrant (this may take a minute depending on PDF size)...")
    index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context,
        show_progress=True,
        insert_batch_size=32
    )
    
    logger.info("Ingestion complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        logger.error("Usage: python ingest.py <path_to_ncert_history.pdf>")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    if not os.path.exists(pdf_file):
        logger.error(f"File not found: {pdf_file}")
        sys.exit(1)
        
    ingest_data(pdf_file)
