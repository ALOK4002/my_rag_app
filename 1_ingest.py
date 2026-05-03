# 1_ingest.py

import os
from dotenv import load_dotenv

# LangChain document loaders - handles reading different file types
from langchain_community.document_loaders import (
    DirectoryLoader,     # Loads all files from a folder
    TextLoader,          # Loads .txt files
    PyPDFLoader          # Loads .pdf files
)

# Text splitter - breaks documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding model - converts text to vectors (runs locally)
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Vector store - stores and searches vectors
from langchain_community.vectorstores import Chroma

import config   # Your settings file

load_dotenv()   # Load .env file


def load_documents():
    """
    Loads all .txt and .pdf files from the knowledge_base folder.
    Returns a list of Document objects.
    Each Document has: page_content (string) + metadata (dict with filename, page, etc.)
    """

    all_docs = []

    # --- Load TXT files ---
    txt_loader = DirectoryLoader(
        path=config.KNOWLEDGE_BASE_DIR,
        glob="**/*.txt",             # Match all .txt files recursively
        loader_cls=TextLoader,       # Use TextLoader for each file
        loader_kwargs={"encoding": "utf-8"}
    )
    all_docs.extend(txt_loader.load())

    # --- Load PDF files ---
    pdf_loader = DirectoryLoader(
        path=config.KNOWLEDGE_BASE_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    all_docs.extend(pdf_loader.load())

    print(f"✅ Loaded {len(all_docs)} document(s)")
    return all_docs

def chunk_documents(documents):
    """
    Splits large documents into smaller overlapping chunks.

    Why RecursiveCharacterTextSplitter?
    It tries to split on paragraphs first, then sentences, then words.
    Smarter than a simple character split — preserves meaning better.

    Example:
    Original: 2000-word article
    After chunking: ~4 chunks of 500 tokens each, with 50-token overlap at edges
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,          # Use character count (not token count here)
        separators=["\n\n", "\n", ". ", " ", ""]
        # ↑ Try to split on paragraph breaks first, then newlines, then sentences
    )

    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunk(s) from {len(documents)} document(s)")
    return chunks


def create_vector_store(chunks):
    """
    Converts each chunk to an embedding vector and saves to ChromaDB.

    SentenceTransformerEmbeddings:
    - Downloads model on first run (~90MB, cached at ~/.cache/huggingface)
    - All subsequent runs are instant (uses local cache)
    - No API key required!

    Chroma.from_documents():
    - Loops through chunks
    - Calls embedding model on each chunk's text
    - Stores (vector, original_text, metadata) in ChromaDB
    - Persists to disk at chroma_db/ folder
    """

    print("⏳ Loading embedding model (may download on first run)...")

    embedding_model = SentenceTransformerEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )

    print("⏳ Embedding chunks and saving to ChromaDB...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DB_DIR    # Save to disk
    )

    print(f"✅ Saved {len(chunks)} chunks to ChromaDB at '{config.CHROMA_DB_DIR}/'")
    return vector_store

def main():
    print("=== INGESTION PIPELINE STARTING ===\n")

    # Step 1: Load raw documents
    documents = load_documents()

    if not documents:
        print("❌ No documents found! Add .txt or .pdf files to knowledge_base/ folder")
        return

    # Step 2: Split into chunks
    chunks = chunk_documents(documents)

    # Step 3: Embed and store
    create_vector_store(chunks)

    print("\n=== INGESTION COMPLETE ✅ ===")
    print("You can now run 2_query.py to search your knowledge base")


if __name__ == "__main__":
    main()