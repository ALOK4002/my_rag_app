# 2_query.py

import os
from dotenv import load_dotenv

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun  # Free web search

# LLM options (uncomment the one you want)
from langchain_groq import ChatGroq                        # Free Groq
# from langchain_openai import ChatOpenAI                  # Paid OpenAI

from langchain.schema import HumanMessage, SystemMessage

import config

load_dotenv()

def load_vector_store():
    """
    Loads the EXISTING ChromaDB from disk.
    Must use the SAME embedding model used during ingestion!
    (If you change models, you must re-run 1_ingest.py)
    """

    embedding_model = SentenceTransformerEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=config.CHROMA_DB_DIR
    )

    return vector_store

def search_local(vector_store, query):
    """
    Searches ChromaDB for chunks similar to the query.

    Returns: list of (Document, score) tuples
    Score is a DISTANCE value:
        0.0 = perfect match
        1.0 = completely unrelated
        < 0.4 = good match (our threshold)

    Why similarity_search_with_score?
    Because we need the score to decide whether to trust local results
    or fall back to web search.
    """

    results = vector_store.similarity_search_with_score(
        query=query,
        k=config.TOP_K_RESULTS
    )

    # Filter by threshold — only keep good matches
    good_results = [
        (doc, score)
        for doc, score in results
        if score < config.SIMILARITY_THRESHOLD    # Lower = better in distance metric
    ]

    return good_results

def search_web(query):
    """
    Falls back to DuckDuckGo if local knowledge base has no good results.
    DuckDuckGo is free and requires no API key.

    Returns: raw string of search results (summaries from top pages)
    """

    print("🌐 No local results found. Searching the web...")

    search_tool = DuckDuckGoSearchRun()
    web_results = search_tool.run(query)

    return web_results

def build_prompt(query, context, source):
    """
    Constructs the prompt that will be sent to the LLM.

    This is the most important part of RAG!
    You are explicitly telling the LLM:
    - What context to use
    - What question to answer
    - How to behave when it doesn't know

    source: "local" or "web" — helps the LLM frame its answer
    """

    system_prompt = """You are a helpful assistant. 
    Answer the user's question using ONLY the provided context.
    If the context doesn't have enough information, say so honestly.
    Do not make up information."""

    user_prompt = f"""
    Context (from {source} search):
    ================================
    {context}
    ================================

    Question: {query}

    Please provide a clear, accurate answer based on the context above.
    """

    return system_prompt, user_prompt



def get_llm_answer(system_prompt, user_prompt):
    """Sends the prompt to the LLM and returns the answer.
    Using Groq (free) — swap to ChatOpenAI if you prefer.
    """

    llm = ChatGroq(
        model=config.LLM_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2    # Low temp = more factual, less creative
                           # Range: 0.0 (deterministic) to 1.0 (creative)
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    return response.content


def query_rag(vector_store, user_question):
    """
    Full RAG pipeline for one question:
    1. Search local knowledge base
    2. If no good local results → search web
    3. Build prompt with context
    4. Get LLM answer
    5. Return answer + source info
    """

    print(f"\n🔍 Searching for: '{user_question}'")

    # --- Step 1: Try local search ---
    local_results = search_local(vector_store, user_question)

    if local_results:
        # Found good local results!
        print(f"📚 Found {len(local_results)} relevant chunk(s) in local knowledge base")

        # Combine chunk texts into one context block
        context = "\n\n---\n\n".join([
            f"[From: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc, score in local_results
        ])
        source = "local knowledge base"

    else:
        # --- Step 2: Fallback to web search ---
        context = search_web(user_question)
        source = "web search"

    # --- Step 3: Build prompt ---
    system_prompt, user_prompt = build_prompt(user_question, context, source)

    # --- Step 4: Get LLM answer ---
    print("🤖 Generating answer...")
    answer = get_llm_answer(system_prompt, user_prompt)

    return answer, source