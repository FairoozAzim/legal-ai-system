import faiss
import numpy as np
import pickle


class FAISSVectorStore:

    def __init__(self, embedding_dim):

        self.index = faiss.IndexFlatL2(embedding_dim)

        self.chunks = []

    def add_documents(self, embeddings, chunks):

        embeddings = np.array(
            embeddings
        ).astype("float32")

        self.index.add(embeddings)

        self.chunks.extend(chunks)

    def search(self, query_embedding, k=5):

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        retrieved_chunks = [
            self.chunks[i]
            for i in indices[0]
        ]

        return retrieved_chunks

    def save(self, path):

        faiss.write_index(
            self.index,
            f"{path}/faiss.index"
        )

        with open(
            f"{path}/chunks.pkl",
            "wb"
        ) as f:

            pickle.dump(self.chunks, f)

    def load(self, path):

        self.index = faiss.read_index(
            f"{path}/faiss.index"
        )

        with open(
            f"{path}/chunks.pkl",
            "rb"
        ) as f:

            self.chunks = pickle.load(f)