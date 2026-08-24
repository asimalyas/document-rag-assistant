import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""

    _instance = None
    _model = None

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        """Singleton pattern to avoid reloading the model per call"""
        if cls._instance is None or cls._model is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager

        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        if EmbeddingManager._model is not None:
            return
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            EmbeddingManager._model = SentenceTransformer(self.model_name)
            dim = EmbeddingManager._model.get_embedding_dimension()
            print(f"Model loaded successfully. Embedding dimension: {dim}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    @property
    def model(self):
        return EmbeddingManager._model

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query text

        Args:
            query: Query text to embed

        Returns:
            1D numpy array embedding
        """
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.encode([query])[0]
