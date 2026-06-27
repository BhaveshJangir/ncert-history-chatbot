# Configuration Settings for NCERT History Chatbot

# Qdrant Database Settings
COLLECTION_NAME = "ncert_history"

# Embedding Model Settings
# We use FastEmbed (BAAI/bge-small-en-v1.5) which runs locally and supports dense + sparse embeddings for Hybrid Search
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Ingestion Settings
CHUNK_SIZE = 512             # Number of tokens per chunk
CHUNK_OVERLAP = 50           # Number of overlapping tokens between chunks
INSERT_BATCH_SIZE = 64       # Lower batch size prevents local Qdrant from freezing during ingestion

# Retrieval Settings
SIMILARITY_TOP_K = 15        # Number of dense vector chunks to retrieve
SPARSE_TOP_K = 15            # Number of sparse vector chunks to retrieve (BM25 keyword search)

# LLM Settings
LLM_MODEL = "llama-3.1-8b-instant" 
