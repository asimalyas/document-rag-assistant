from src.data_loader import load_all_documents
from src.vectorstore import VectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":

    docs = load_all_documents("data")
    store = VectorStore()
    # Uncomment to rebuild the index from scratch:
    # store.build_from_documents(docs)

    rag_search = RAGSearch(vectorstore=store)
    query = "What is attention mechanism?"
    result = rag_search.search_with_trace(query, top_k=3)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
    print(f"Latency: {result['latency_ms']}ms")