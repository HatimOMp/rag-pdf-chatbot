import faiss
import numpy as np
from mistralai import Mistral
import os


class VectorStore:
    """
    Handles embedding generation and similarity search using
    Mistral embeddings + FAISS vector index.
    """

    def __init__(self):
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.embedding_model = "mistral-embed"
        self.index = None
        self.chunks = []
        self.dimension = 1024  # Mistral embedding dimension

    def embed_texts(self, texts):
        """Generate embeddings for a list of texts."""
        embeddings = []
        # Process in batches of 10 to avoid rate limits
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                inputs=batch
            )
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings, dtype=np.float32)

    def build_index(self, chunks):
        """Build FAISS index from document chunks."""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        # Generate embeddings
        embeddings = self.embed_texts(texts)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build flat index (exact search)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        return len(chunks)

    def search(self, query, top_k=4):
        """Find most relevant chunks for a query."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Embed query
        query_embedding = self.embed_texts([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score)
                })

        return results