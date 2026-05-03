import chromadb
import config

print("🔍 Connecting to ChromaDB...")

# Connect to your existing ChromaDB
client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)

# List all collections
collections = client.list_collections()
print(f"📦 Total collections found: {len(collections)}")

if len(collections) == 0:
    print("❌ No collections found — ingestion may not have run successfully")
else:
    for col in collections:
        print(f"\n📁 Collection name: {col.name}")
        
        collection = client.get_collection(col.name)
        count = collection.count()
        print(f"   Total chunks stored: {count}")

        if count > 0:
            results = collection.get(limit=3)
            print(f"\n   Showing first {len(results['documents'])} chunk(s):")
            for i, doc in enumerate(results['documents']):
                print(f"\n   --- Chunk {i+1} ---")
                print(f"   {doc[:300]}")
        else:
            print("   ⚠️ Collection exists but has no documents")

print("\n✅ Done")