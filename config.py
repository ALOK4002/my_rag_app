# config.py

# --- Paths ---
KNOWLEDGE_BASE_DIR = "knowledge_base"    # Folder with your documents
CHROMA_DB_DIR = "chroma_db"              # Where vectors will be saved
COLLECTION_NAME = "my_knowledge_base"   # Like a table name in ChromaDB

# --- Chunking Settings ---
CHUNK_SIZE = 500         # Max tokens per chunk
CHUNK_OVERLAP = 50       # Overlap between chunks (preserves context at boundaries)

# --- Embedding Model ---
# This runs 100% locally, no API key needed
# Downloads ~90MB model on first run (cached after that)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval Settings ---
TOP_K_RESULTS = 3                  # How many chunks to retrieve
SIMILARITY_THRESHOLD = 0.4         # Minimum score to trust local results
                                   # ChromaDB returns "distance" not similarity
                                   # Lower distance = more similar
                                   # Distance < 0.4 means good match

# --- LLM Settings ---
LLM_MODEL = "llama3-8b-8192"       # Groq's free fast model
# LLM_MODEL = "gpt-4o-mini"        # Use this if using OpenAI

# --- Web Search Settings ---
MAX_WEB_RESULTS = 3                # How many web results to fetch as fallback