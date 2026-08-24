def main():
    """Entry point for the yt-rag package."""
    from .src.search import RAGSearch

    rag = RAGSearch()
    query = "What is attention mechanism?"
    result = rag.search_with_trace(query, top_k=3)
    print("Answer:", result["answer"])
