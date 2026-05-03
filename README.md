# 🤖 RAG Chatbot — Local + Web Search

A **Retrieval-Augmented Generation (RAG)** chatbot that combines local knowledge base search with web search fallback. Uses **LangChain**, **ChromaDB**, and **Groq API** for fast, intelligent answers.

## ✨ Features

- 📚 **Local Knowledge Base**: Stores and searches documents (PDF, TXT) using vector embeddings
- 🌐 **Web Search Fallback**: Automatically searches the web if local knowledge has no good matches
- ⚡ **Fast & Free**: Uses Groq's free LLM API (no expensive OpenAI costs)
- 🧠 **Smart Embeddings**: Uses locally-run `SentenceTransformer` (no internet needed after first run)
- 💾 **Persistent Storage**: ChromaDB stores embeddings locally for instant retrieval
- 🔍 **Similarity Scoring**: Filters results by relevance threshold to ensure quality

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+
- pip

### 2. Installation

```bash
# Clone the repo
git clone <YOUR_REPO_URL>
cd my_rag_app

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # On Windows
# source venv/bin/activate   # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
# Optional - if using OpenAI instead:
# OPENAI_API_KEY=your_openai_api_key_here
```

Get your free Groq API key from [console.groq.com](https://console.groq.com)

### 4. Add Your Documents

Place your `.txt` or `.pdf` files in the `knowledge_base/` folder:

```
knowledge_base/
├── document1.txt
├── document2.pdf
└── document3.txt
```

### 5. Ingest Documents

This processes your documents and creates vector embeddings:

```bash
python ingest.py
```

**What happens:**
- Reads all `.txt` and `.pdf` files from `knowledge_base/`
- Splits them into chunks (500 characters with 50-char overlap)
- Embeds each chunk using `SentenceTransformer`
- Stores in ChromaDB (local SQLite database)
- First run downloads ~90MB embedding model (then cached locally)

### 6. Run the Chatbot

```bash
python app.py
```

Type your questions. The bot will:
1. Search local knowledge base first
2. Fall back to web search if needed
3. Return answers with source information

Type `quit`, `exit`, or `bye` to close.

## 📁 Project Structure

```
my_rag_app/
├── ingest.py           # Load docs, create embeddings, save to ChromaDB
├── query.py            # Search local + web, call LLM for answers
├── app.py              # Chat interface
├── config.py           # Settings (chunk size, model names, etc.)
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (API keys)
├── knowledge_base/    # Your documents (add PDFs/TXTs here)
├── chroma_db/         # Vector database (auto-created)
└── README.md          # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
CHUNK_SIZE = 500              # Chunk size for splitting documents
CHUNK_OVERLAP = 50            # Overlap between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local embedding model
TOP_K_RESULTS = 3             # How many chunks to retrieve
SIMILARITY_THRESHOLD = 0.4    # Minimum relevance score
LLM_MODEL = "llama3-8b-8192"  # Groq model (free & fast)
MAX_WEB_RESULTS = 3           # Web search results as fallback
```

### Switch to OpenAI (paid)

In `config.py`:
```python
LLM_MODEL = "gpt-4o-mini"
```

In `query.py`:
```python
# Uncomment this:
from langchain_openai import ChatOpenAI

# Comment out this:
# from langchain_groq import ChatGroq
```

## 🔄 Workflow

### Pipeline Overview

```
Documents (PDF/TXT)
    ↓
[1. ingest.py] → Split into chunks
    ↓
[Embedding Model] → Convert to vectors
    ↓
[ChromaDB] → Store vectors + text
    ↓
User Question
    ↓
[query.py] → Search ChromaDB for similar chunks
    ↓
    ├─ Found good matches? → Use them as context
    └─ No matches? → Search web for context
    ↓
[Build Prompt] → Combine context + question
    ↓
[LLM (Groq)] → Generate answer
    ↓
User sees answer + source info
```

## 📊 Example Usage

```
You: What is machine learning?
🔍 Searching for: 'What is machine learning?'
📚 Found 2 relevant chunk(s) in local knowledge base
🤖 Generating answer...

🤖 Answer (via local knowledge base):
----------------------------------------
Machine learning is a subset of artificial intelligence...
[From: knowledge_base/ml_basics.txt]
Machine learning enables systems to learn from data...
----------------------------------------
```

## 🛠️ Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'langchain_core'
```
**Solution:** Run `pip install -r requirements.txt` again

### ChromaDB Path Error
```
PersistentClient() missing some required arguments
```
**Solution:** Ensure `chroma_db/` folder exists (auto-created by `ingest.py`)

### No Results from Web Search
Check internet connection and DuckDuckGo availability (no API key needed)

### LLM Returns Empty Response
- Verify `GROQ_API_KEY` is set in `.env`
- Check Groq API status at [status.groq.com](https://status.groq.com)
- Try a simpler question

## 📦 Dependencies

- **langchain** - RAG framework
- **langchain-groq** - Groq LLM integration
- **langchain-openai** - OpenAI integration (optional)
- **chromadb** - Vector database
- **sentence-transformers** - Local embeddings
- **duckduckgo-search** - Free web search
- **pypdf** - PDF reading
- **python-dotenv** - Environment variables

## 🔐 Security Notes

- Never commit `.env` file (contains API keys)
- `.gitignore` already excludes it
- Large ChromaDB files are not tracked (only first ingestion needed locally)

## 🚀 Performance Tips

1. **Adjust chunk size** - Larger chunks = faster but less precise
2. **Lower similarity threshold** - Gets more results (may include noise)
3. **Use better embedding model** - Trade-off: larger download, better accuracy
   - Try `sentence-transformers/all-mpnet-base-v2` (best quality, ~420MB)
   - Current: `all-MiniLM-L6-v2` (fast, ~90MB)

## 📝 License

MIT License - feel free to use and modify

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues, questions, or suggestions:
- Check existing issues on GitHub
- Create a new issue with clear description
- Include error messages and steps to reproduce

---

**Happy querying! 🚀**
