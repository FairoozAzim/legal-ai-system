import faiss
import json
import numpy as np


class FAISSVectorStore:

    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = []

    def add_documents(self, embeddings, chunks):
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) must match "
                f"chunks count ({len(chunks)})"
            )

        embeddings_np = np.array(embeddings).astype("float32")

        # Guard: ensure shape is (n, dim)
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)

        self.index.add(embeddings_np)
        self.chunks.extend(chunks)

    def search(self, query_embedding, k=5):
        query_np = np.array(query_embedding).astype("float32")

        # Ensure shape is (1, dim) for a single query
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        distances, indices = self.index.search(query_np, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:           # FAISS returns -1 for unfilled slots
                continue
            results.append({
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx]["metadata"],
                "score": float(dist)
            })

        return results

    def save(self, path):
        faiss.write_index(self.index, f"{path}/faiss.index")

        # Store chunks as JSON instead of pickle
        with open(f"{path}/chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load(self, path):
        self.index = faiss.read_index(f"{path}/faiss.index")

        with open(f"{path}/chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)