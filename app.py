import os
from query import load_vector_store, query_rag   # Import from query.py


def main():
    print("=" * 50)
    print("   🤖 RAG Chatbot — Local + Web Search")
    print("=" * 50)
    print("Type your question. Type 'quit' to exit.\n")

    # Load vector store ONCE at startup (expensive operation)
    print("⏳ Loading knowledge base...")
    vector_store = load_vector_store()
    print("✅ Ready!\n")

    # Chat loop
    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye! 👋")
            break

        # Get answer
        answer, source = query_rag(vector_store, user_input)

        print(f"\n🤖 Answer (via {source}):")
        print("-" * 40)
        print(answer)
        print("-" * 40 + "\n")


if __name__ == "__main__":
    main()